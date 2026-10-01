//! bobcat runs language models on the Metal GPU.
//!
//! `bobcat respond` answers one prompt and writes only the answer to standard output, so it
//! composes with pipes. `bobcat chat` holds an interactive conversation in the terminal. `bobcat
//! serve` answers OpenAI and Anthropic API requests over HTTP, for agents and other tools.
//! `bobcat pull`, `bobcat list`, and `bobcat rm` manage models in the Hugging Face cache.

use std::io::{self, IsTerminal, Read, Write};
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, CommandFactory, Parser, Subcommand};

#[cfg(target_os = "macos")]
mod backend;
#[cfg(target_os = "macos")]
mod conversation;
#[cfg(target_os = "macos")]
mod engine;
mod models;
#[cfg(target_os = "macos")]
mod sampler;
#[cfg(target_os = "macos")]
mod serve;
#[cfg(target_os = "macos")]
mod systemone;
#[cfg(target_os = "macos")]
mod tokenizer;
mod tools;

type Error = Box<dyn std::error::Error + Send + Sync>;

/// Run language models on the Metal GPU.
#[derive(Debug, Parser)]
#[command(name = "bobcat", version)]
struct Options {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Answer one prompt and write the answer to standard output.
    ///
    /// Text on standard input follows the prompt, so `cat notes.md | bobcat respond -m MODEL
    /// "Summarize this."` answers about the file.
    Respond {
        #[command(flatten)]
        model: ModelOptions,
        #[command(flatten)]
        reply: ReplyOptions,
        /// Write the model's thinking to standard error.
        #[arg(long)]
        think: bool,
        /// The prompt. Words join with spaces.
        prompt: Vec<String>,
    },
    /// Chat with the model in the terminal.
    ///
    /// Ctrl-C stops a reply, and Ctrl-D ends the chat. A line of three double quotes starts and
    /// ends a message of several lines. `/help` lists the chat commands.
    Chat {
        #[command(flatten)]
        model: ModelOptions,
        #[command(flatten)]
        reply: ReplyOptions,
    },
    /// Answer a Jev/SystemOne decision request with a Clef model and write the response.
    ///
    /// The request is a JSON object with `model`, `state`, and `questions`, given as an argument
    /// or on standard input.
    Decide {
        #[command(flatten)]
        model: ModelOptions,
        /// The request.
        request: Option<String>,
    },
    /// Answer OpenAI and Anthropic API requests over HTTP, for agents and other tools.
    ///
    /// The server answers `/v1/chat/completions`, `/v1/messages`, `/v1/messages/count_tokens`,
    /// and `/v1/models`. A Clef model also answers Jev/SystemOne decisions at `/v1/systemone`.
    Serve {
        #[command(flatten)]
        model: ModelOptions,
        /// The address to listen on.
        #[arg(long, default_value = "127.0.0.1")]
        host: IpAddr,
        /// The port to listen on.
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// The most tokens in one reply, for requests that name no limit.
        #[arg(short = 'n', long, default_value_t = 4096)]
        max_tokens: usize,
        /// The most tokens in one conversation. Agents send long system prompts, so the server
        /// defaults to a larger context than chat.
        #[arg(short, long, default_value_t = 32768)]
        context: u32,
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
        /// The model, as `bobcat list` shows it.
        name: String,
    },
    /// Print a shell completion script, as in `bobcat completions zsh > ~/.zfunc/_bobcat`.
    Completions {
        /// The shell to complete for.
        shell: clap_complete::Shell,
    },
}

/// The options that loading a model takes.
#[derive(Debug, Args)]
struct ModelOptions {
    /// A model alias such as `lfm2.5:2.6b`, a Hugging Face name such as
    /// `LiquidAI/LFM2.5-2.6B-GGUF:Q8_0`, or a GGUF file. A named model downloads on first use.
    #[arg(short, long, env = "BOBCAT_MODEL")]
    model: String,
    /// A Hugging Face `tokenizer.json` to use in place of the tokenizer inside the model file.
    #[arg(short, long)]
    tokenizer: Option<PathBuf>,
}

/// The options of the replies of `respond` and `chat`.
#[derive(Debug, Args)]
struct ReplyOptions {
    /// The system message that opens the conversation.
    #[arg(short, long)]
    system: Option<String>,
    /// The most tokens in one reply.
    #[arg(short = 'n', long, default_value_t = 4096)]
    max_tokens: usize,
    /// The most tokens in the whole conversation.
    #[arg(short, long, default_value_t = 8192)]
    context: u32,
    /// Write the time to the first token and the decode speed of each reply to standard error.
    #[arg(short, long)]
    verbose: bool,
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
            #[expect(clippy::print_stderr, reason = "bobcat reports errors on stderr")]
            {
                eprintln!("bobcat: {error}");
            }
            ExitCode::FAILURE
        }
    }
}

fn run(command: Command) -> Result<(), Error> {
    match command {
        Command::Respond {
            model,
            reply,
            think,
            prompt,
        } => respond(&model, &reply, think, &prompt.join(" ")),
        Command::Chat { model, reply } => chat(&model, &reply),
        Command::Decide { model, request } => decide(&model, request.as_deref().unwrap_or("")),
        Command::Serve {
            model,
            host,
            port,
            max_tokens,
            context,
        } => serve(&model, SocketAddr::new(host, port), max_tokens, context),
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
        Command::Completions { shell } => {
            clap_complete::generate(shell, &mut Options::command(), "bobcat", &mut io::stdout());
            Ok(())
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn respond(_: &ModelOptions, _: &ReplyOptions, _: bool, _: &str) -> Result<(), Error> {
    Err("bobcat runs models on the Metal GPU, which needs macOS".into())
}

#[cfg(not(target_os = "macos"))]
fn chat(_: &ModelOptions, _: &ReplyOptions) -> Result<(), Error> {
    Err("bobcat runs models on the Metal GPU, which needs macOS".into())
}

#[cfg(not(target_os = "macos"))]
fn decide(_: &ModelOptions, _: &str) -> Result<(), Error> {
    Err("bobcat runs models on the Metal GPU, which needs macOS".into())
}

#[cfg(not(target_os = "macos"))]
fn serve(_: &ModelOptions, _: SocketAddr, _: usize, _: u32) -> Result<(), Error> {
    Err("bobcat runs models on the Metal GPU, which needs macOS".into())
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
fn load(options: &ModelOptions) -> Result<(backend::Model, tokenizers::Tokenizer), Error> {
    let model = backend::Model::load(models::resolve(&options.model)?)?;
    let tokenizer = match &options.tokenizer {
        Some(path) => tokenizers::Tokenizer::from_file(path)?,
        None => model.tokenizer()?,
    };
    Ok((model, tokenizer))
}

/// Start the conversation with `model` on `metal` that `options` describe.
#[cfg(target_os = "macos")]
fn start<'a>(
    options: &ReplyOptions,
    model: &'a backend::Model,
    metal: &'a mut bobcat::metal::Metal,
    tokenizer: &'a tokenizers::Tokenizer,
) -> Result<conversation::Conversation<'a>, Error> {
    let recommended = model.recommended_sampling();
    let flags = &options.sampling;
    let settings = bobcat::Sampling {
        temperature: flags.temperature.unwrap_or(recommended.temperature),
        top_k: flags.top_k.unwrap_or(recommended.top_k),
        top_p: flags.top_p.unwrap_or(recommended.top_p),
        min_p: flags.min_p.unwrap_or(recommended.min_p),
        repeat_penalty: flags.repeat_penalty.unwrap_or(recommended.repeat_penalty),
    };
    let seed = flags.seed.unwrap_or_else(|| fastrand::u64(..));
    let engine = engine::Engine::new(model, metal, tokenizer, options.context)?;
    Ok(conversation::Conversation::new(
        engine,
        options.system.as_deref(),
        sampler::Sampler::new(settings, seed),
        options.max_tokens,
    ))
}

/// Write the time to the first token and the decode speed of `reply` to stderr.
#[cfg(target_os = "macos")]
fn report(reply: &engine::Reply) -> Result<(), Error> {
    let first_token_ms = reply.first_token.as_secs_f64() * 1000.0;
    let decode_seconds = reply
        .elapsed
        .saturating_sub(reply.first_token)
        .as_secs_f64();
    let decoded = reply.completion_tokens.saturating_sub(1);
    let speed = if decode_seconds > 0.0 {
        decoded as f64 / decode_seconds
    } else {
        0.0
    };
    writeln!(
        io::stderr(),
        "{} prompt tokens, first token after {first_token_ms:.0} ms, {} tokens at {speed:.1} \
         tokens/s",
        reply.prompt_tokens,
        reply.completion_tokens
    )?;
    Ok(())
}

/// Answer `prompt` and stdin's text, writing the answer to stdout and, when `think` is true, the
/// thinking to stderr.
#[cfg(target_os = "macos")]
fn respond(
    model_options: &ModelOptions,
    options: &ReplyOptions,
    think: bool,
    prompt: &str,
) -> Result<(), Error> {
    use engine::{Event, Part};

    let text = full_prompt(prompt)?;
    let (model, tokenizer) = load(model_options)?;
    let mut metal = bobcat::metal::Metal::open()?;
    let mut conversation = start(options, &model, &mut metal, &tokenizer)?;

    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    let mut thought = false;
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let reply = conversation.reply(&text, &cancel, |event| {
        match event {
            Event::Text(Part::Thinking, piece) if think => {
                stderr.write_all(piece.as_bytes())?;
                stderr.flush()?;
                thought = true;
            }
            Event::Text(Part::Answer, piece) => {
                if thought {
                    // End the thinking's line before the answer starts.
                    stderr.write_all(b"\n")?;
                    thought = false;
                }
                stdout.write_all(piece.as_bytes())?;
                stdout.flush()?;
            }
            Event::Text(Part::Thinking, _) | Event::Prompt(_) | Event::ToolCall(_) => {}
        }
        Ok(())
    })?;
    writeln!(stdout)?;
    if options.verbose {
        report(&reply)?;
    }
    Ok(())
}

/// Answer the decision `request`, or the request on stdin, and write the response to stdout.
#[cfg(target_os = "macos")]
fn decide(model_options: &ModelOptions, request: &str) -> Result<(), Error> {
    let text = full_prompt(request)?;
    let request: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("the request is not JSON: {error}"))?;
    let (model, tokenizer) = load(model_options)?;
    let mut metal = bobcat::metal::Metal::open()?;
    let context = u32::try_from(systemone::MAX_TOKENS)?;
    let mut engine = engine::Engine::new(&model, &mut metal, &tokenizer, context)?;
    let response = engine.decide(&request)?;
    writeln!(io::stdout(), "{response:#}")?;
    Ok(())
}

/// A line of the chat, after reading multi-line messages and chat commands.
#[cfg(target_os = "macos")]
enum Input {
    Message(String),
    /// A command that the chat handled itself.
    Handled,
    Exit,
}

/// The chat commands, as `/help` lists them.
#[cfg(target_os = "macos")]
const CHAT_HELP: &str = "\
/clear           forget the conversation
/system TEXT     set the system message, or remove it when TEXT is empty
/exit            end the chat
\"\"\"              start or end a message of several lines
Ctrl-C           stop a reply
Ctrl-D           end the chat";

/// Read the next message of the chat from `editor`, handling chat commands on `conversation`.
#[cfg(target_os = "macos")]
fn read_input(
    editor: &mut rustyline::DefaultEditor,
    conversation: &mut conversation::Conversation<'_>,
) -> Result<Input, Error> {
    use rustyline::error::ReadlineError;

    let line = match editor.readline("> ") {
        Ok(line) => line,
        // Ctrl-C at the prompt clears the line, as in a shell.
        Err(ReadlineError::Interrupted) => return Ok(Input::Handled),
        Err(ReadlineError::Eof) => return Ok(Input::Exit),
        Err(error) => return Err(error.into()),
    };
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(Input::Handled);
    }
    editor.add_history_entry(line.as_str())?;
    if trimmed == "\"\"\"" {
        let mut lines = Vec::new();
        loop {
            match editor.readline(". ") {
                Ok(line) if line.trim() == "\"\"\"" => break,
                Ok(line) => lines.push(line),
                Err(ReadlineError::Interrupted) => return Ok(Input::Handled),
                Err(ReadlineError::Eof) => break,
                Err(error) => return Err(error.into()),
            }
        }
        return Ok(Input::Message(lines.join("\n")));
    }
    let Some(command) = trimmed.strip_prefix('/') else {
        return Ok(Input::Message(line));
    };
    let (name, argument) = command.split_once(' ').unwrap_or((command, ""));
    let mut stderr = io::stderr();
    match name {
        "clear" => {
            conversation.clear();
            writeln!(stderr, "The conversation is cleared.")?;
        }
        "system" => {
            conversation.set_system(argument.trim());
            conversation.clear();
            writeln!(
                stderr,
                "The system message is set, and the conversation is cleared."
            )?;
        }
        "exit" | "bye" => return Ok(Input::Exit),
        "help" => writeln!(stderr, "{CHAT_HELP}")?,
        _ => writeln!(stderr, "Unknown command /{name}. /help lists the commands.")?,
    }
    Ok(Input::Handled)
}

/// Hold an interactive conversation until the user ends the input.
#[cfg(target_os = "macos")]
fn chat(model_options: &ModelOptions, options: &ReplyOptions) -> Result<(), Error> {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use engine::{Event, Finish, Part};

    let (model, tokenizer) = load(model_options)?;
    let mut metal = bobcat::metal::Metal::open()?;
    let mut conversation = start(options, &model, &mut metal, &tokenizer)?;
    let mut editor = rustyline::DefaultEditor::new()?;
    // Ctrl-C during a reply stops the reply. At the prompt, the line editor sees Ctrl-C itself.
    let cancel = Arc::new(AtomicBool::new(false));
    let handler = Arc::clone(&cancel);
    ctrlc::set_handler(move || handler.store(true, Ordering::Relaxed))?;
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
        let text = match read_input(&mut editor, &mut conversation)? {
            Input::Message(text) => text,
            Input::Handled => continue,
            Input::Exit => return Ok(()),
        };
        cancel.store(false, Ordering::Relaxed);
        let mut last = None;
        let reply = conversation.reply(&text, &cancel, |event| {
            let Event::Text(part, piece) = event else {
                return Ok(());
            };
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
        stdout.flush()?;
        match reply {
            Ok(reply) => {
                if reply.finish == Finish::Cancelled {
                    writeln!(io::stderr(), "Stopped.\n")?;
                }
                if options.verbose {
                    report(&reply)?;
                }
            }
            // A failed reply, such as one that overflows the context, leaves the chat open for
            // the next message.
            Err(error) => writeln!(io::stderr(), "bobcat: {error}\n")?,
        }
    }
}

/// Serve the model that `options` name at `address`.
#[cfg(target_os = "macos")]
fn serve(
    options: &ModelOptions,
    address: SocketAddr,
    max_tokens: usize,
    context: u32,
) -> Result<(), Error> {
    let (model, tokenizer) = load(options)?;
    let serve_options = serve::Options {
        name: options.model.clone(),
        context,
        max_tokens,
        address,
    };
    serve::run(&serve_options, model, tokenizer)
}
