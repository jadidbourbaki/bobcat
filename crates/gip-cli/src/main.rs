//! gip runs language models on the Metal GPU.
//!
//! `gip respond` answers one prompt and writes only the answer to standard output, so it composes
//! with pipes. `gip chat` holds an interactive conversation in the terminal. `gip pull`, `gip
//! list`, and `gip rm` manage models in the Hugging Face cache.

use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};

#[cfg(target_os = "macos")]
mod conversation;
mod models;
#[cfg(target_os = "macos")]
mod sampler;
#[cfg(target_os = "macos")]
mod tokenizer;

type Error = Box<dyn std::error::Error + Send + Sync>;

/// Run language models on the Metal GPU.
#[derive(Debug, Parser)]
#[command(name = "gip", version)]
struct Options {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Answer one prompt and write the answer to standard output.
    ///
    /// Text on standard input follows the prompt, so `cat notes.md | gip respond -m MODEL
    /// "Summarize this."` answers about the file.
    Respond {
        #[command(flatten)]
        model: ModelOptions,
        /// Write the model's thinking to standard error.
        #[arg(long)]
        think: bool,
        /// The prompt. Words join with spaces.
        prompt: Vec<String>,
    },
    /// Chat with the model in the terminal. Ctrl-D ends the chat.
    Chat {
        #[command(flatten)]
        model: ModelOptions,
    },
    /// Download a model from Hugging Face and print the path of its file.
    Pull {
        /// The model, as an alias such as `lfm2.5:2.6b` or a Hugging Face name such as
        /// `LiquidAI/LFM2.5-2.6B-GGUF:Q8_0`. Aliases default to Q4_K_M and Hugging Face names to
        /// Q8_0.
        name: String,
    },
    /// List the downloaded models and their sizes.
    List,
    /// Remove a downloaded model.
    Rm {
        /// The model, as `gip list` shows it.
        name: String,
    },
}

/// The options that running a model takes.
#[derive(Debug, Args)]
struct ModelOptions {
    /// A model alias such as `lfm2.5:2.6b`, a Hugging Face name such as
    /// `LiquidAI/LFM2.5-2.6B-GGUF:Q8_0`, or a GGUF file. A named model downloads on first use.
    #[arg(short, long)]
    model: String,
    /// A Hugging Face `tokenizer.json` to use in place of the tokenizer inside the model file.
    #[arg(short, long)]
    tokenizer: Option<PathBuf>,
    /// The system message that opens the conversation.
    #[arg(short, long)]
    system: Option<String>,
    /// The most tokens in one reply.
    #[arg(short = 'n', long, default_value_t = 4096)]
    max_tokens: usize,
    /// The most tokens in the whole conversation.
    #[arg(short, long, default_value_t = 8192)]
    context: u32,
    #[command(flatten)]
    sampling: SamplingOptions,
}

/// Overrides of the sampling settings the model recommends, which come from the model file and
/// then from the model's authors.
#[derive(Debug, Args)]
#[command(next_help_heading = "Sampling")]
struct SamplingOptions {
    /// The softmax temperature. Zero always picks the most likely token.
    #[arg(long)]
    temperature: Option<f32>,
    /// Keep only this many most likely tokens. Zero keeps every token.
    #[arg(long)]
    top_k: Option<u32>,
    /// Keep the fewest tokens whose probabilities sum to this much.
    #[arg(long)]
    top_p: Option<f32>,
    /// Drop tokens less likely than this fraction of the most likely token.
    #[arg(long)]
    min_p: Option<f32>,
    /// Divide the logits of recently used tokens by this much. One turns the penalty off.
    #[arg(long)]
    repeat_penalty: Option<f32>,
    /// The seed of the random draws, for repeatable replies.
    #[arg(long)]
    seed: Option<u64>,
}

fn main() -> ExitCode {
    let options = Options::parse();
    match run(options.command) {
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

fn run(command: Command) -> Result<(), Error> {
    match command {
        Command::Respond {
            model,
            think,
            prompt,
        } => respond(&model, think, &prompt.join(" ")),
        Command::Chat { model } => chat(&model),
        Command::Pull { name } => {
            let path = models::pull(&models::Name::parse(&name)?)?;
            writeln!(io::stdout(), "{}", path.display())?;
            Ok(())
        }
        Command::List => {
            let mut stdout = io::stdout().lock();
            // A model with an alias lists the alias first and its full name last.
            for (name, size, alias) in models::list()? {
                let size = models::human_size(size);
                match alias {
                    Some(alias) => writeln!(stdout, "{alias}\t{size}\t{name}")?,
                    None => writeln!(stdout, "{name}\t{size}")?,
                }
            }
            Ok(())
        }
        Command::Rm { name } => models::remove(&models::Name::parse(&name)?),
    }
}

#[cfg(not(target_os = "macos"))]
fn respond(_: &ModelOptions, _: bool, _: &str) -> Result<(), Error> {
    Err("gip runs models on the Metal GPU, which needs macOS".into())
}

#[cfg(not(target_os = "macos"))]
fn chat(_: &ModelOptions) -> Result<(), Error> {
    Err("gip runs models on the Metal GPU, which needs macOS".into())
}

/// Return the prompt, followed by any text on standard input.
fn full_prompt(prompt: &str) -> Result<String, Error> {
    let mut stdin = io::stdin();
    let mut input = String::new();
    if !stdin.is_terminal() {
        stdin.read_to_string(&mut input)?;
    }
    let text = match (prompt.trim().is_empty(), input.trim().is_empty()) {
        (true, true) => {
            return Err("no prompt, so pass one as an argument or on standard input".into());
        }
        (false, true) => prompt.to_owned(),
        (true, false) => input,
        (false, false) => format!("{prompt}\n\n{input}"),
    };
    Ok(text)
}

/// Load the model and tokenizer that `options` name.
#[cfg(target_os = "macos")]
fn load(options: &ModelOptions) -> Result<(gip::Model, tokenizers::Tokenizer), Error> {
    let model = gip::Model::load(models::resolve(&options.model)?)?;
    let tokenizer = match &options.tokenizer {
        Some(path) => tokenizers::Tokenizer::from_file(path)?,
        None => tokenizer::from_gguf(model.gguf())?,
    };
    Ok((model, tokenizer))
}

/// Start the conversation with `model` on `metal` that `options` describe.
#[cfg(target_os = "macos")]
fn start<'a>(
    options: &ModelOptions,
    model: &'a gip::Model,
    metal: &'a mut gip::metal::Metal,
    tokenizer: &'a tokenizers::Tokenizer,
) -> Result<conversation::Conversation<'a>, Error> {
    let recommended = model.recommended_sampling();
    let flags = &options.sampling;
    let settings = gip::Sampling {
        temperature: flags.temperature.unwrap_or(recommended.temperature),
        top_k: flags.top_k.unwrap_or(recommended.top_k),
        top_p: flags.top_p.unwrap_or(recommended.top_p),
        min_p: flags.min_p.unwrap_or(recommended.min_p),
        repeat_penalty: flags.repeat_penalty.unwrap_or(recommended.repeat_penalty),
    };
    let seed = flags.seed.unwrap_or_else(|| fastrand::u64(..));
    let limits = conversation::Limits {
        max_tokens: options.max_tokens,
        context: options.context,
    };
    conversation::Conversation::new(
        model,
        metal,
        tokenizer,
        options.system.as_deref(),
        limits,
        sampler::Sampler::new(settings, seed),
    )
}

/// Answer `prompt` and stdin's text, writing the answer to stdout and, when `think` is true, the
/// thinking to stderr.
#[cfg(target_os = "macos")]
fn respond(options: &ModelOptions, think: bool, prompt: &str) -> Result<(), Error> {
    use conversation::Part;

    let text = full_prompt(prompt)?;
    let (model, tokenizer) = load(options)?;
    let mut metal = gip::metal::Metal::open()?;
    let mut conversation = start(options, &model, &mut metal, &tokenizer)?;

    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    let mut thought = false;
    conversation.reply(&text, |part, piece| {
        match part {
            Part::Thinking if think => {
                stderr.write_all(piece.as_bytes())?;
                stderr.flush()?;
                thought = true;
            }
            Part::Thinking => {}
            Part::Answer => {
                if thought {
                    // End the thinking's line before the answer starts.
                    stderr.write_all(b"\n")?;
                    thought = false;
                }
                stdout.write_all(piece.as_bytes())?;
                stdout.flush()?;
            }
        }
        Ok(())
    })?;
    writeln!(stdout)?;
    Ok(())
}

/// Hold an interactive conversation until the user ends the input.
#[cfg(target_os = "macos")]
fn chat(options: &ModelOptions) -> Result<(), Error> {
    use conversation::Part;
    use rustyline::error::ReadlineError;

    let (model, tokenizer) = load(options)?;
    let mut metal = gip::metal::Metal::open()?;
    let mut conversation = start(options, &model, &mut metal, &tokenizer)?;
    let mut editor = rustyline::DefaultEditor::new()?;
    let mut stdout = io::stdout().lock();
    // Dim text marks the thinking, unless the output goes elsewhere or the user asks for no
    // color, as https://no-color.org describes.
    let styled = stdout.is_terminal() && std::env::var_os("NO_COLOR").is_none();
    let (dim, reset) = if styled {
        ("\x1b[2m", "\x1b[0m")
    } else {
        ("", "")
    };

    loop {
        let line = match editor.readline("> ") {
            Ok(line) => line,
            Err(ReadlineError::Eof | ReadlineError::Interrupted) => return Ok(()),
            Err(error) => return Err(error.into()),
        };
        if line.trim().is_empty() {
            continue;
        }
        editor.add_history_entry(line.as_str())?;

        let mut last = None;
        let reply = conversation.reply(&line, |part, piece| {
            if last.is_none() && part == Part::Thinking {
                write!(stdout, "{dim}")?;
            }
            if last == Some(Part::Thinking) && part == Part::Answer {
                write!(stdout, "{reset}\n\n")?;
            }
            last = Some(part);
            stdout.write_all(piece.as_bytes())?;
            stdout.flush()?;
            Ok(())
        });
        if last.is_some() {
            write!(stdout, "{reset}\n\n")?;
        }
        // A failed reply, such as one that overflows the context, leaves the chat open for the
        // next message.
        if let Err(error) = reply {
            stdout.flush()?;
            #[expect(clippy::print_stderr, reason = "gip reports errors on stderr")]
            {
                eprintln!("gip: {error}\n");
            }
        }
    }
}
