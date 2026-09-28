//! A conversation with a model: the chat template, the tokenizer, and the messages so far.

use gip::Model;
use gip::metal::Metal;
use minijinja::{Environment, Value, context};
use tokenizers::Tokenizer;

use crate::Error;

/// Decode this many tokens per GPU call. Each call pipelines its steps, and text streams out
/// after each call, so the count trades pipelining against output latency.
const DECODE_CHUNK: usize = 8;

/// The part of a reply a piece of text belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Part {
    /// The model's reasoning before its answer.
    Thinking,
    /// The answer itself.
    Answer,
}

/// The limits of a conversation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Limits {
    /// The most tokens in one reply.
    pub(crate) max_tokens: usize,
    /// The most tokens in the whole conversation.
    pub(crate) context: u32,
}

/// A conversation with a model on the Metal GPU.
pub(crate) struct Conversation<'a> {
    model: &'a Model,
    tokenizer: &'a Tokenizer,
    template: Environment<'static>,
    bos_token: String,
    stop_token: u32,
    think_open: Option<u32>,
    think_close: Option<u32>,
    /// Every message so far.
    messages: Vec<Value>,
    limits: Limits,
}

impl<'a> Conversation<'a> {
    /// Start a conversation with `model`, opened by the system message `system` when given.
    pub(crate) fn new(
        model: &'a Model,
        tokenizer: &'a Tokenizer,
        system: Option<&str>,
        limits: Limits,
    ) -> Result<Self, Error> {
        let source = model
            .chat_template()
            .ok_or("the model file holds no chat template")?;
        let stop_token = model
            .eos_token()
            .ok_or("the model file names no end-of-turn token")?;
        let bos_token = model
            .bos_token()
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
            tokenizer,
            template,
            bos_token,
            stop_token,
            think_open: tokenizer.token_to_id("<think>"),
            think_close: tokenizer.token_to_id("</think>"),
            messages: system
                .map(|content| context! { role => "system", content => content })
                .into_iter()
                .collect(),
            limits,
        })
    }

    /// Reply to the user message `text`, passing each piece of the reply to `emit` as it decodes.
    ///
    /// The template may render earlier turns differently once a new turn follows, for example by
    /// dropping an earlier answer's thinking. LFM2's convolution state cannot rewind to a shared
    /// prefix, so each reply prefills the whole conversation from a fresh state.
    pub(crate) fn reply(
        &mut self,
        metal: &mut Metal,
        text: &str,
        mut emit: impl FnMut(Part, &str) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.messages
            .push(context! { role => "user", content => text });
        let prompt = self.template.get_template("chat")?.render(context! {
            messages => &self.messages,
            bos_token => &self.bos_token,
            add_generation_prompt => true,
        })?;
        // The template writes the special tokens as text, including the one that begins the
        // sequence, so the tokenizer adds none of its own.
        let tokens = self
            .tokenizer
            .encode(prompt.as_str(), false)?
            .get_ids()
            .to_vec();
        let prompt_len = u32::try_from(tokens.len())?;
        if prompt_len >= self.limits.context {
            return Err(format!(
                "the conversation holds {prompt_len} tokens, which fills the context of {}",
                self.limits.context
            )
            .into());
        }

        let mut gpu = gip::Lfm2Metal::new(self.model, metal, self.limits.context, true)?;
        gpu.prefill(&tokens, None, None)?;

        // Some templates open the reply's thinking in the prompt itself.
        let mut part = if prompt.ends_with("<think>") {
            Part::Thinking
        } else {
            Part::Answer
        };
        let mut thinking = String::new();
        let mut answer = String::new();
        let mut decoder = self.tokenizer.decode_stream(false);
        let room = (self.limits.context - prompt_len) as usize;
        let mut remaining = self.limits.max_tokens.min(room);
        let mut chunk = [0; DECODE_CHUNK];
        'decode: while remaining > 0 {
            let count = DECODE_CHUNK.min(remaining);
            gpu.generate(&mut chunk[..count])?;
            remaining -= count;
            for &token in &chunk[..count] {
                if token == self.stop_token {
                    break 'decode;
                }
                if Some(token) == self.think_open {
                    part = Part::Thinking;
                    continue;
                }
                if Some(token) == self.think_close {
                    part = Part::Answer;
                    continue;
                }
                let Some(piece) = decoder.step(token)? else {
                    continue;
                };
                let text = match part {
                    Part::Thinking => &mut thinking,
                    Part::Answer => &mut answer,
                };
                // The model separates its thinking from its answer with blank lines.
                let piece = if text.is_empty() {
                    piece.trim_start()
                } else {
                    &piece
                };
                if !piece.is_empty() {
                    text.push_str(piece);
                    emit(part, piece)?;
                }
            }
        }

        self.messages.push(context! {
            role => "assistant",
            content => answer.trim_end(),
            thinking => thinking.trim_end(),
        });
        Ok(())
    }
}
