//! gip runs language models on the Metal GPU.
//!
//! `gip respond` answers one prompt and writes only the answer to standard output, so it composes
//! with pipes. `gip chat` holds an interactive conversation in the terminal.

use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};

#[cfg(target_os = "macos")]
mod conversation;

type Error = Box<dyn std::error::Error + Send + Sync>;

/// Run language models on the Metal GPU.
#[derive(Debug, Parser)]
#[command(version)]
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
}

/// The options every command shares.
#[derive(Debug, Args)]
struct ModelOptions {
    /// The GGUF model file.
    #[arg(short, long)]
    model: PathBuf,
    /// The model's Hugging Face `tokenizer.json`. Defaults to `tokenizer.json` beside the model.
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
}

impl ModelOptions {
    /// Return the tokenizer file to load.
    fn tokenizer_path(&self) -> Result<PathBuf, Error> {
        if let Some(path) = &self.tokenizer {
            return Ok(path.clone());
        }
        let beside = self.model.with_file_name("tokenizer.json");
        if beside.exists() {
            Ok(beside)
        } else {
            Err(format!(
                "no tokenizer.json beside {}, so pass one with --tokenizer",
                self.model.display()
            )
            .into())
        }
    }
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

#[cfg(not(target_os = "macos"))]
fn run(_: Command) -> Result<(), Error> {
    Err("gip runs models on the Metal GPU, which needs macOS".into())
}

#[cfg(target_os = "macos")]
fn run(command: Command) -> Result<(), Error> {
    match command {
        Command::Respond {
            model,
            think,
            prompt,
        } => respond(&model, think, &prompt.join(" ")),
        Command::Chat { model } => chat(&model),
    }
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
    let tokenizer = tokenizers::Tokenizer::from_file(options.tokenizer_path()?)?;
    Ok((gip::Model::load(&options.model)?, tokenizer))
}

#[cfg(target_os = "macos")]
fn limits(options: &ModelOptions) -> conversation::Limits {
    conversation::Limits {
        max_tokens: options.max_tokens,
        context: options.context,
    }
}

/// Answer `prompt` and stdin's text, writing the answer to stdout and, when `think` is true, the
/// thinking to stderr.
#[cfg(target_os = "macos")]
fn respond(options: &ModelOptions, think: bool, prompt: &str) -> Result<(), Error> {
    use conversation::{Conversation, Part};

    let text = full_prompt(prompt)?;
    let (model, tokenizer) = load(options)?;
    let mut metal = gip::metal::Metal::open()?;
    let mut conversation = Conversation::new(
        &model,
        &tokenizer,
        options.system.as_deref(),
        limits(options),
    )?;

    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    let mut thought = false;
    conversation.reply(&mut metal, &text, |part, piece| {
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
    use conversation::{Conversation, Part};
    use rustyline::error::ReadlineError;

    let (model, tokenizer) = load(options)?;
    let mut metal = gip::metal::Metal::open()?;
    let mut conversation = Conversation::new(
        &model,
        &tokenizer,
        options.system.as_deref(),
        limits(options),
    )?;
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
        conversation.reply(&mut metal, &line, |part, piece| {
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
        })?;
        write!(stdout, "{reset}\n\n")?;
    }
}
