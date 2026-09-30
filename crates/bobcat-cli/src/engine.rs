//! A model on the Metal GPU that renders conversations through the model's chat template and
//! generates replies.
//!
//! The engine keeps the GPU state of the tokens it has run. A prompt that starts with those tokens
//! runs only the tokens that follow them, so a conversation that grows by a turn, or an agent that
//! resends its history with one more message, costs only the new tokens.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use bobcat::metal::Metal;
use bobcat::{Checkpoint, Lfm2Metal, Model};
use minijinja::{Environment, context};
use tokenizers::Tokenizer;

use crate::Error;
use crate::sampler::Sampler;
use crate::tools::{self, ToolCall};

/// The part of a reply a piece of text belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Part {
    /// The model's reasoning before its answer.
    Thinking,
    /// The answer itself.
    Answer,
}

/// The author of a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    System,
    User,
    Assistant,
    /// The result of a tool call.
    Tool,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// One message of a conversation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Message {
    pub(crate) role: Role,
    pub(crate) content: String,
    /// The tools an assistant message called.
    pub(crate) tool_calls: Vec<ToolCall>,
}

impl Message {
    /// Return a message from `role` holding `content` and no tool calls.
    pub(crate) fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
        }
    }
}

/// Why a reply ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Finish {
    /// The model ended its turn or wrote a stop sequence.
    Stop,
    /// The reply reached its token limit.
    Length,
    /// The model called tools and waits for their results.
    ToolCalls,
    /// The caller cancelled the reply.
    Cancelled,
}

/// What a reply produced.
#[derive(Debug, Clone)]
pub(crate) struct Reply {
    pub(crate) answer: String,
    pub(crate) finish: Finish,
    pub(crate) prompt_tokens: usize,
    /// The prompt tokens that an earlier reply had already run.
    pub(crate) cached_tokens: usize,
    pub(crate) completion_tokens: usize,
    /// The time from the start of the reply to its first token.
    pub(crate) first_token: Duration,
    /// The time the whole reply took.
    pub(crate) elapsed: Duration,
}

/// Something a reply produced, passed to the caller as it decodes.
#[derive(Debug)]
pub(crate) enum Event<'e> {
    /// The prompt holds this many tokens. The event comes before any text.
    Prompt(usize),
    /// A piece of text.
    Text(Part, &'e str),
    /// A complete tool call.
    ToolCall(&'e ToolCall),
}

/// What to generate.
pub(crate) struct Request<'r> {
    pub(crate) messages: &'r [Message],
    /// The tools the model may call, as JSON objects with a name, a description, and parameters.
    pub(crate) tools: &'r [serde_json::Value],
    pub(crate) sampler: &'r mut Sampler,
    /// The most tokens in the reply.
    pub(crate) max_tokens: usize,
    /// Text that ends the reply when the answer contains it. The reply leaves the text out.
    pub(crate) stop: &'r [String],
}

/// A model on the Metal GPU.
pub(crate) struct Engine<'a> {
    model: &'a Model,
    gpu: Lfm2Metal<'a>,
    /// The tokens the GPU state has run, in order.
    consumed: Vec<u32>,
    /// The GPU state after the latest prompt's history, before the turn of the reply, with the
    /// history's tokens. The next request's prompt repeats the history but none of the reply's
    /// thinking, so the engine restarts from here.
    history: Option<(Vec<u32>, Checkpoint)>,
    tokenizer: &'a Tokenizer,
    template: Environment<'static>,
    bos_token: String,
    context: u32,
    special: Special,
}

/// The ids of the tokens that change how the engine reads the model's output.
struct Special {
    stop: u32,
    think_open: Option<u32>,
    think_close: Option<u32>,
    call_open: Option<u32>,
    call_close: Option<u32>,
    turn_open: Option<u32>,
}

/// The tokens that wrap a tool call in LFM2's output.
const CALL_OPEN: &str = "<|tool_call_start|>";
const CALL_CLOSE: &str = "<|tool_call_end|>";

impl<'a> Engine<'a> {
    /// Load `model` onto `metal` for conversations of up to `context` tokens.
    pub(crate) fn new(
        model: &'a Model,
        metal: &'a mut Metal,
        tokenizer: &'a Tokenizer,
        context: u32,
    ) -> Result<Self, Error> {
        let gguf = model.gguf();
        let source = gguf
            .string("tokenizer.chat_template")
            .ok_or("the model file holds no chat template")?;
        let source = std::str::from_utf8(source)?;
        let stop = gguf
            .u32("tokenizer.ggml.eos_token_id")
            .ok_or("the model file names no end-of-turn token")?;
        let bos_token = gguf
            .u32("tokenizer.ggml.bos_token_id")
            .and_then(|id| tokenizer.id_to_token(id))
            .unwrap_or_default();

        let mut template = Environment::new();
        // Hugging Face renders chat templates with Python's Jinja2, so templates call Python
        // string and dictionary methods, which pycompat provides.
        template.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        template.add_function("raise_exception", |message: String| -> Result<(), _> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                message,
            ))
        });
        // The generation tag marks assistant text for training masks in transformers and has no
        // effect on the rendered text. MiniJinja does not know the tag.
        let source = source
            .replace("{%- generation -%}", "")
            .replace("{%- endgeneration -%}", "");
        template.add_template_owned("chat", source)?;

        Ok(Self {
            model,
            gpu: Lfm2Metal::new(model, metal, context, true)?,
            consumed: Vec::new(),
            history: None,
            tokenizer,
            template,
            bos_token,
            context,
            special: Special {
                stop,
                think_open: tokenizer.token_to_id("<think>"),
                think_close: tokenizer.token_to_id("</think>"),
                call_open: tokenizer.token_to_id(CALL_OPEN),
                call_close: tokenizer.token_to_id(CALL_CLOSE),
                turn_open: tokenizer.token_to_id("<|im_start|>"),
            },
        })
    }

    /// Return the recommended sampling settings of the model.
    pub(crate) fn recommended_sampling(&self) -> bobcat::Sampling {
        self.model.recommended_sampling()
    }

    /// Return the prompt that `messages` and `tools` render to, ending where the reply begins.
    fn render(&self, messages: &[Message], tools: &[serde_json::Value]) -> Result<String, Error> {
        let messages: Vec<minijinja::Value> = messages
            .iter()
            .map(|message| {
                // The template knows no tool calls, so earlier calls go back into the text in the
                // form the model writes them.
                let content = if message.tool_calls.is_empty() {
                    message.content.clone()
                } else {
                    format!(
                        "{}{CALL_OPEN}{}{CALL_CLOSE}",
                        message.content,
                        tools::format(&message.tool_calls)
                    )
                };
                context! { role => message.role.as_str(), content => content }
            })
            .collect();
        let tools = (!tools.is_empty()).then(|| minijinja::Value::from_serialize(tools));
        let prompt = self.template.get_template("chat")?.render(context! {
            messages => messages,
            tools => tools,
            bos_token => &self.bos_token,
            add_generation_prompt => true,
        })?;
        Ok(prompt)
    }

    /// Return the tokens of the prompt that `messages` and `tools` render to.
    fn encode(&self, messages: &[Message], tools: &[serde_json::Value]) -> Result<Vec<u32>, Error> {
        let prompt = self.render(messages, tools)?;
        // The template writes the special tokens as text, including the one that begins the
        // sequence, so the tokenizer adds none of its own.
        let encoding = self.tokenizer.encode(prompt.as_str(), false)?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Return the number of tokens in the prompt that `messages` and `tools` render to.
    pub(crate) fn count(
        &self,
        messages: &[Message],
        tools: &[serde_json::Value],
    ) -> Result<usize, Error> {
        Ok(self.encode(messages, tools)?.len())
    }

    /// Generate the reply to `request`, passing each event to `emit` as it decodes.
    ///
    /// The reply ends early when `cancel` becomes true. A failed reply leaves the engine ready for
    /// the next one.
    pub(crate) fn generate(
        &mut self,
        request: &mut Request<'_>,
        cancel: &AtomicBool,
        emit: impl FnMut(Event<'_>) -> Result<(), Error>,
    ) -> Result<Reply, Error> {
        let result = self.generate_inner(request, cancel, emit);
        if result.is_err() {
            // The GPU state may hold part of the failed reply, so the next reply starts over.
            self.consumed.clear();
            self.history = None;
        }
        result
    }

    fn generate_inner(
        &mut self,
        request: &mut Request<'_>,
        cancel: &AtomicBool,
        mut emit: impl FnMut(Event<'_>) -> Result<(), Error>,
    ) -> Result<Reply, Error> {
        let started = Instant::now();
        let tokens = self.encode(request.messages, request.tools)?;
        let prompt_tokens = tokens.len();
        let prompt_len = u32::try_from(prompt_tokens)?;
        if prompt_len >= self.context {
            return Err(format!(
                "the conversation holds {prompt_len} tokens, which fills the context of {}",
                self.context
            )
            .into());
        }
        emit(Event::Prompt(prompt_tokens))?;

        // Some templates open the reply's thinking in the prompt itself.
        let part = if tokens.last().copied() == self.special.think_open {
            Part::Thinking
        } else {
            Part::Answer
        };
        let mut reader = Reader {
            special: &self.special,
            decoder: self.tokenizer.decode_stream(false),
            part,
            thinking: String::new(),
            answer: String::new(),
            call_text: String::new(),
            in_call: false,
            tool_calls: Vec::new(),
            stop: request.stop,
            generated: 0,
            first_token: None,
            started,
        };

        let cached_tokens = if !self.consumed.is_empty() && tokens.starts_with(&self.consumed) {
            self.consumed.len()
        } else if let Some((history, checkpoint)) = &self.history
            && tokens.starts_with(history)
        {
            self.gpu.restore(checkpoint)?;
            history.len()
        } else {
            self.gpu.reset()?;
            self.history = None;
            0
        };
        // The reply's turn starts at the prompt's last turn token.
        let turn = tokens
            .iter()
            .rposition(|&token| Some(token) == self.special.turn_open)
            .unwrap_or(0);
        let start = if turn > cached_tokens {
            self.gpu.prefill(&tokens[cached_tokens..turn], None, None)?;
            self.history = Some((tokens[..turn].to_vec(), self.gpu.checkpoint()?));
            turn
        } else {
            cached_tokens
        };

        let room = (self.context - prompt_len) as usize;
        let mut remaining = request.max_tokens.min(room);
        let mut finish = None;
        if request.sampler.is_greedy() {
            // The GPU picks each most likely token itself, several steps ahead of the CPU, and
            // each token streams out as soon as its step finishes. The steps already submitted
            // when the reply ends still run, and their tokens join the consumed history.
            self.gpu.prefill(&tokens[start..], None, None)?;
            self.consumed = tokens;
            let mut failure = None;
            let gained =
                self.gpu
                    .generate_stream(u32::try_from(remaining)?, |token| {
                        match reader.accept(token, cancel, &mut emit) {
                            Ok(None) => true,
                            Ok(Some(reason)) => {
                                finish = Some(reason);
                                false
                            }
                            Err(error) => {
                                failure = Some(error);
                                false
                            }
                        }
                    })?;
            self.consumed.extend_from_slice(&gained);
            if let Some(error) = failure {
                return Err(error);
            }
        } else {
            // Sampling reads each step's logits on the CPU before the next step can start.
            let mut logits = vec![0.0; self.model.hyperparameters().n_vocab as usize];
            self.gpu
                .prefill(&tokens[start..], Some(&mut logits), None)?;
            self.consumed = tokens;
            while remaining > 0 {
                let token = request.sampler.sample(&mut logits, &self.consumed);
                remaining -= 1;
                finish = reader.accept(token, cancel, &mut emit)?;
                if finish.is_some() || remaining == 0 {
                    break;
                }
                self.gpu.step(token, Some(&mut logits), None)?;
                self.consumed.push(token);
            }
        }

        let finish = match finish {
            Some(Finish::Stop) if !reader.tool_calls.is_empty() => Finish::ToolCalls,
            Some(finish) => finish,
            None => Finish::Length,
        };
        let elapsed = started.elapsed();
        Ok(Reply {
            answer: reader.answer.trim_end().to_owned(),
            finish,
            prompt_tokens,
            cached_tokens,
            completion_tokens: reader.generated,
            first_token: reader.first_token.unwrap_or(elapsed),
            elapsed,
        })
    }
}

/// Reads the model's output token by token into text, thinking, and tool calls.
struct Reader<'r> {
    special: &'r Special,
    decoder: tokenizers::DecodeStream<
        'r,
        tokenizers::ModelWrapper,
        tokenizers::NormalizerWrapper,
        tokenizers::PreTokenizerWrapper,
        tokenizers::PostProcessorWrapper,
        tokenizers::DecoderWrapper,
    >,
    part: Part,
    thinking: String,
    answer: String,
    /// The text of the tool call being written.
    call_text: String,
    in_call: bool,
    tool_calls: Vec<ToolCall>,
    stop: &'r [String],
    generated: usize,
    first_token: Option<Duration>,
    started: Instant,
}

impl Reader<'_> {
    /// Take one generated token and return why the reply ended, if it has.
    fn accept(
        &mut self,
        token: u32,
        cancel: &AtomicBool,
        emit: &mut impl FnMut(Event<'_>) -> Result<(), Error>,
    ) -> Result<Option<Finish>, Error> {
        if cancel.load(Ordering::Relaxed) {
            return Ok(Some(Finish::Cancelled));
        }
        self.first_token
            .get_or_insert_with(|| self.started.elapsed());
        self.generated += 1;
        let special = self.special;
        if token == special.stop {
            return Ok(Some(Finish::Stop));
        }
        if Some(token) == special.think_open {
            self.part = Part::Thinking;
            return Ok(None);
        }
        if Some(token) == special.think_close {
            self.part = Part::Answer;
            return Ok(None);
        }
        if Some(token) == special.call_open {
            self.in_call = true;
            self.call_text.clear();
            return Ok(None);
        }
        if Some(token) == special.call_close {
            self.in_call = false;
            let calls = tools::parse(&self.call_text)
                .map_err(|error| format!("the model wrote a malformed tool call: {error}"))?;
            for call in calls {
                emit(Event::ToolCall(&call))?;
                self.tool_calls.push(call);
            }
            return Ok(None);
        }
        let Some(piece) = self.decoder.step(token)? else {
            return Ok(None);
        };
        if self.in_call {
            self.call_text.push_str(&piece);
            return Ok(None);
        }
        let text = match self.part {
            Part::Thinking => &mut self.thinking,
            Part::Answer => &mut self.answer,
        };
        // The model separates its thinking from its answer with blank lines.
        let piece = if text.is_empty() {
            piece.trim_start()
        } else {
            &piece
        };
        if piece.is_empty() {
            return Ok(None);
        }
        let before = text.len();
        text.push_str(piece);
        if self.part == Part::Answer
            && let Some(at) = self
                .stop
                .iter()
                .filter_map(|stop| text.find(stop.as_str()))
                .min()
        {
            // A stop sequence ends the reply, and the reply leaves out the sequence and all after.
            if at > before {
                emit(Event::Text(Part::Answer, &text[before..at]))?;
            }
            text.truncate(at);
            return Ok(Some(Finish::Stop));
        }
        emit(Event::Text(self.part, piece))?;
        Ok(None)
    }
}
