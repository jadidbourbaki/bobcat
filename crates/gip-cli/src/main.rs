//! gip chats with a model on the Metal GPU.
//!
//! With `--prompt`, gip answers one message and exits. Otherwise gip reads one message per line
//! from standard input and answers each in turn until the input ends. The model's own chat template
//! formats the conversation, and Hugging Face's tokenizer converts text to tokens and back.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

/// Decode this many tokens per GPU call. Each call pipelines its steps, and text streams out after
/// each call, so the count trades pipelining against output latency.
#[cfg(target_os = "macos")]
const DECODE_CHUNK: usize = 8;

/// Chat with a model on the Metal GPU.
#[derive(Debug, Parser)]
#[command(version)]
struct Options {
    /// The GGUF model file.
    model: PathBuf,
    /// The model's Hugging Face `tokenizer.json`.
    #[arg(short, long)]
    tokenizer: PathBuf,
    /// Answer this message and exit.
    #[arg(short, long)]
    prompt: Option<String>,
    /// The system message that opens the conversation.
    #[arg(short, long)]
    system: Option<String>,
    /// The most tokens in one answer.
    #[arg(short = 'n', long, default_value_t = 2048)]
    max_tokens: usize,
    /// The most tokens in the whole conversation.
    #[arg(short, long, default_value_t = 8192)]
    context: u32,
}

fn main() -> ExitCode {
    let options = Options::parse();
    match run(&options) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            #[expect(clippy::print_stderr, reason = "gip reports errors on stderr")]
            {
                eprintln!("gip: {error}");
            }
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn run(_: &Options) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    Err("gip runs models on the Metal GPU, which needs macOS".into())
}

/// Load the model and hold the conversation `options` asks for.
#[cfg(target_os = "macos")]
fn run(options: &Options) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let model = gip::Model::load(&options.model)?;
    let tokenizer = tokenizers::Tokenizer::from_file(&options.tokenizer)?;
    let mut chat = Chat::new(&model, &tokenizer, options)?;
    let mut metal = gip::metal::Metal::open()?;
    let mut out = io::stdout().lock();

    if let Some(prompt) = &options.prompt {
        return chat.answer(&mut metal, prompt, &mut out);
    }
    for line in io::stdin().lock().lines() {
        let line = line?;
        if !line.trim().is_empty() {
            chat.answer(&mut metal, &line, &mut out)?;
        }
    }
    Ok(())
}

/// A conversation with a model.
#[cfg(target_os = "macos")]
struct Chat<'a> {
    model: &'a gip::Model,
    tokenizer: &'a tokenizers::Tokenizer,
    template: minijinja::Environment<'static>,
    bos_token: String,
    stop_token: u32,
    /// Every message so far, as maps with a `role` and a `content`.
    messages: Vec<minijinja::Value>,
    max_tokens: usize,
    context: u32,
}

#[cfg(target_os = "macos")]
impl<'a> Chat<'a> {
    fn new(
        model: &'a gip::Model,
        tokenizer: &'a tokenizers::Tokenizer,
        options: &Options,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
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

        let mut template = minijinja::Environment::new();
        // Hugging Face templates call Python string and dictionary methods, which pycompat
        // provides.
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

        let messages = options
            .system
            .iter()
            .map(|content| message("system", content))
            .collect();
        Ok(Self {
            model,
            tokenizer,
            template,
            bos_token,
            stop_token,
            messages,
            max_tokens: options.max_tokens,
            context: options.context,
        })
    }

    /// Answer the user message `text`, streaming the answer to `out`.
    ///
    /// The template may render earlier turns differently once a new turn follows, for example
    /// by dropping an earlier answer's thinking. LFM2's convolution state cannot rewind to a
    /// shared prefix, so each answer prefills the whole conversation from a fresh state.
    fn answer(
        &mut self,
        metal: &mut gip::metal::Metal,
        text: &str,
        out: &mut impl Write,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.messages.push(message("user", text));
        let prompt = self
            .template
            .get_template("chat")?
            .render(minijinja::context! {
                messages => &self.messages,
                bos_token => &self.bos_token,
                add_generation_prompt => true,
            })?;
        // The template writes the special tokens as text, including the one that begins the
        // sequence, so the tokenizer adds none of its own.
        let tokens = self.tokenizer.encode(prompt, false)?.get_ids().to_vec();

        let mut gpu = gip::Lfm2Metal::new(self.model, metal, self.context, true)?;
        gpu.prefill(&tokens, None, None)?;

        let mut decoder = self.tokenizer.decode_stream(false);
        let mut answer = String::new();
        let mut chunk = [0; DECODE_CHUNK];
        let mut remaining = self.max_tokens;
        'decode: while remaining > 0 {
            let room = self.context as usize - tokens.len() - (self.max_tokens - remaining);
            let count = DECODE_CHUNK.min(remaining).min(room);
            if count == 0 {
                break;
            }
            gpu.generate(&mut chunk[..count])?;
            for &token in &chunk[..count] {
                if token == self.stop_token {
                    break 'decode;
                }
                if let Some(piece) = decoder.step(token)? {
                    out.write_all(piece.as_bytes())?;
                    answer.push_str(&piece);
                }
            }
            out.flush()?;
            remaining -= count;
        }
        writeln!(out)?;
        self.messages.push(message("assistant", &answer));
        Ok(())
    }
}

/// Return a chat message from `role` holding `content`.
#[cfg(target_os = "macos")]
fn message(role: &str, content: &str) -> minijinja::Value {
    minijinja::context! { role => role, content => content }
}
