//! The HTTP server of `bobcat serve`, which answers OpenAI Chat Completions and Anthropic
//! Messages requests with one model, and decision requests as [`crate::decisions`] describes.
//!
//! One worker thread owns the model and the GPU and answers requests one at a time. The HTTP
//! handlers translate each request into the engine's messages, queue it for the worker, and send
//! the worker's output back in the format of the API that asked. Agents resend their whole history
//! with every request, and the engine runs only the tokens past the history it has already run.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::{Map, Value, json};
use tokio::sync::{mpsc as channel, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;

use bobcat::clef::Head;

use crate::Error;
use crate::decisions::{self, Refusal};
use crate::engine::{Engine, Event, Finish, Message, Part, Request, Role};
use crate::sampler::Sampler;
use crate::tools::ToolCall;

/// How to serve a model.
#[derive(Debug, Clone)]
pub(crate) struct Options {
    /// The model's name in the API, as `-m` gave it.
    pub(crate) name: String,
    /// The most tokens in one conversation.
    pub(crate) context: u32,
    /// The most tokens in one reply when a request names no limit.
    pub(crate) max_tokens: usize,
    /// The address and port to listen on.
    pub(crate) address: SocketAddr,
}

/// Serve `model` with `tokenizer`, and with the decision head `head` when it is given, until the
/// process ends.
pub(crate) fn run(
    options: &Options,
    model: crate::backend::Model,
    tokenizer: tokenizers::Tokenizer,
    head: Option<Head>,
) -> Result<(), Error> {
    // Binding before the model loads reports a busy port at once.
    let listener = std::net::TcpListener::bind(options.address)
        .map_err(|error| format!("cannot listen on {}: {error}", options.address))?;
    listener.set_nonblocking(true)?;
    let (jobs, queue) = mpsc::channel();
    let (ready, started) = mpsc::channel();
    let worker_options = options.clone();
    std::thread::spawn(move || {
        worker(&model, &tokenizer, head, &worker_options, &queue, &ready);
    });
    started
        .recv()
        .map_err(|_| "the model worker stopped while loading")??;

    let shared = Arc::new(Shared {
        jobs,
        name: options.name.clone(),
        next_id: AtomicU64::new(1),
    });
    let app = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .route("/v1/decisions", post(decisions))
        .route("/v1/systemone", post(systemone))
        .route("/v1/score", post(score))
        .with_state(shared);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::from_std(listener)?;
        #[expect(clippy::print_stderr, reason = "bobcat reports its address on stderr")]
        {
            eprintln!(
                "bobcat: serving {} at http://{}",
                options.name, options.address
            );
        }
        axum::serve(listener, app).await?;
        Ok(())
    })
}

/// Work for the model worker.
enum Job {
    /// Generate a reply and send its updates.
    Generate {
        messages: Vec<Message>,
        tools: Vec<Value>,
        settings: Settings,
        updates: channel::UnboundedSender<Update>,
    },
    /// Answer a Jev/SystemOne decision request.
    Decide {
        route: Route,
        request: Value,
        reply: oneshot::Sender<Result<Value, Refusal>>,
    },
    /// Count the tokens of a prompt.
    Count {
        messages: Vec<Message>,
        tools: Vec<Value>,
        reply: oneshot::Sender<Result<usize, String>>,
    },
}

/// The sampling settings and limits of one request. Settings a request leaves out take the
/// model's recommended values.
#[derive(Default)]
struct Settings {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    seed: Option<u64>,
    max_tokens: Option<usize>,
    stop: Vec<String>,
}

/// What the worker reports about a reply as it decodes.
enum Update {
    /// The prompt holds this many tokens.
    Started(usize),
    Text(String),
    ToolCall(ToolCall),
    Done {
        finish: Finish,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Failed(String),
}

/// The decision routes, which share one job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    Decisions,
    SystemOne,
    Score,
}

/// Load `model` onto the GPU as `options` describe, report on `ready`, and answer the jobs on
/// `queue` one at a time.
fn worker(
    model: &crate::backend::Model,
    tokenizer: &tokenizers::Tokenizer,
    head: Option<Head>,
    options: &Options,
    queue: &mpsc::Receiver<Job>,
    ready: &mpsc::Sender<Result<(), Error>>,
) {
    let name = options.name.as_str();
    let max_tokens = options.max_tokens;
    let mut metal = match bobcat::metal::Metal::open() {
        Ok(metal) => metal,
        Err(error) => {
            let _ = ready.send(Err(error.into()));
            return;
        }
    };
    let mut engine = match Engine::new(model, &mut metal, tokenizer, options.context, head) {
        Ok(engine) => engine,
        Err(error) => {
            let _ = ready.send(Err(error));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        return;
    }
    for job in queue {
        match job {
            Job::Generate {
                messages,
                tools,
                settings,
                updates,
            } => generate(
                &mut engine,
                &messages,
                &tools,
                &settings,
                max_tokens,
                &updates,
            ),
            Job::Count {
                messages,
                tools,
                reply,
            } => {
                let count = engine
                    .count(&messages, &tools)
                    .map_err(|error| error.to_string());
                let _ = reply.send(count);
            }
            Job::Decide {
                route,
                request,
                reply,
            } => {
                let response = match route {
                    Route::Decisions => decisions::decisions(&mut engine, &request),
                    Route::SystemOne => decisions::systemone(&mut engine, &request, name),
                    Route::Score => decisions::score_request(&mut engine, &request),
                };
                let _ = reply.send(response);
            }
        }
    }
}

/// Generate the reply to one request and send its updates.
fn generate(
    engine: &mut Engine<'_>,
    messages: &[Message],
    tools: &[Value],
    settings: &Settings,
    default_max_tokens: usize,
    updates: &channel::UnboundedSender<Update>,
) {
    let recommended = engine.recommended_sampling();
    let sampling = bobcat::Sampling {
        temperature: settings.temperature.unwrap_or(recommended.temperature),
        top_k: settings.top_k.unwrap_or(recommended.top_k),
        top_p: settings.top_p.unwrap_or(recommended.top_p),
        min_p: recommended.min_p,
        repeat_penalty: recommended.repeat_penalty,
    };
    let seed = settings.seed.unwrap_or_else(|| fastrand::u64(..));
    let mut sampler = Sampler::new(sampling, seed);
    let mut request = Request {
        messages,
        tools,
        sampler: &mut sampler,
        max_tokens: settings.max_tokens.unwrap_or(default_max_tokens),
        stop: &settings.stop,
    };
    // A client that disconnects drops its receiver, and the reply stops at the next token.
    let cancel = AtomicBool::new(false);
    let result = engine.generate(&mut request, &cancel, |event| {
        let update = match event {
            Event::Prompt(tokens) => Update::Started(tokens),
            Event::Text(Part::Answer, text) => Update::Text(text.to_owned()),
            Event::Text(Part::Thinking, _) => return Ok(()),
            Event::ToolCall(call) => Update::ToolCall(call.clone()),
        };
        if updates.send(update).is_err() {
            cancel.store(true, Ordering::Relaxed);
        }
        Ok(())
    });
    let update = match result {
        Ok(reply) => {
            #[expect(clippy::print_stderr, reason = "the server logs each reply on stderr")]
            {
                eprintln!(
                    "bobcat: {} prompt tokens with {} cached, {} reply tokens in {:.1} s",
                    reply.prompt_tokens,
                    reply.cached_tokens,
                    reply.completion_tokens,
                    reply.elapsed.as_secs_f64()
                );
            }
            Update::Done {
                finish: reply.finish,
                prompt_tokens: reply.prompt_tokens,
                completion_tokens: reply.completion_tokens,
            }
        }
        Err(error) => {
            #[expect(clippy::print_stderr, reason = "the server logs each reply on stderr")]
            {
                eprintln!("bobcat: {error}");
            }
            Update::Failed(error.to_string())
        }
    };
    let _ = updates.send(update);
}

/// The state every handler shares.
struct Shared {
    jobs: mpsc::Sender<Job>,
    name: String,
    next_id: AtomicU64,
}

impl Shared {
    /// Return a new identifier that starts with `prefix`.
    fn id(&self, prefix: &str) -> String {
        format!("{prefix}{}", self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Queue a reply to `parsed` and return the receiver of its updates.
    fn submit(&self, parsed: Parsed) -> Result<channel::UnboundedReceiver<Update>, String> {
        let (updates, receiver) = channel::unbounded_channel();
        let job = Job::Generate {
            messages: parsed.messages,
            tools: parsed.tools,
            settings: parsed.settings,
            updates,
        };
        self.jobs
            .send(job)
            .map_err(|_| "the model worker has stopped".to_owned())?;
        Ok(receiver)
    }
}

/// A request translated into the engine's terms.
struct Parsed {
    messages: Vec<Message>,
    tools: Vec<Value>,
    settings: Settings,
}

/// The collected result of a reply, for a response that is not streamed.
struct Collected {
    text: String,
    calls: Vec<ToolCall>,
    finish: Finish,
    prompt_tokens: usize,
    completion_tokens: usize,
}

/// Wait for every update of a reply and return what it produced.
async fn collect(mut updates: channel::UnboundedReceiver<Update>) -> Result<Collected, String> {
    let mut text = String::new();
    let mut calls = Vec::new();
    while let Some(update) = updates.recv().await {
        match update {
            Update::Started(_) => {}
            Update::Text(piece) => text.push_str(&piece),
            Update::ToolCall(call) => calls.push(call),
            Update::Done {
                finish,
                prompt_tokens,
                completion_tokens,
            } => {
                return Ok(Collected {
                    text,
                    calls,
                    finish,
                    prompt_tokens,
                    completion_tokens,
                });
            }
            Update::Failed(error) => return Err(error),
        }
    }
    Err("the model worker stopped during the reply".to_owned())
}

/// Return a server-sent event stream that sends the events a task writes into its channel.
fn event_stream<F>(
    write: impl FnOnce(channel::UnboundedSender<Result<SseEvent, Infallible>>) -> F,
) -> Response
where
    F: Future<Output = ()> + Send + 'static,
{
    let (events, receiver) = channel::unbounded_channel();
    tokio::spawn(write(events));
    // Keep-alive comments hold the connection open through a long prefill.
    Sse::new(UnboundedReceiverStream::new(receiver))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Return the seconds since the Unix epoch.
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// Return the text of a message content, which is a string, a list of blocks with text, or null.
fn text_of(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Return `value` as a float setting.
fn float(value: &Value) -> Option<f32> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "sampling settings need only float precision"
    )]
    value.as_f64().map(|value| value as f32)
}

/// Return `value` as a count setting.
fn count(value: &Value) -> Option<usize> {
    value.as_u64().and_then(|value| usize::try_from(value).ok())
}

/// Return the stop sequences in `value`, a string or a list of strings.
fn stops(value: &Value) -> Vec<String> {
    match value {
        Value::String(stop) => vec![stop.clone()],
        Value::Array(stops) => stops
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

async fn models(State(shared): State<Arc<Shared>>) -> Response {
    let body = json!({
        "object": "list",
        "data": [{"id": shared.name, "object": "model", "created": 0, "owned_by": "bobcat"}],
    });
    axum::Json(body).into_response()
}

/// Return an OpenAI error response.
fn openai_error(status: StatusCode, message: &str) -> Response {
    let body = json!({"error": {"message": message, "type": "invalid_request_error"}});
    (status, axum::Json(body)).into_response()
}

/// Translate an OpenAI Chat Completions request.
fn parse_openai(request: &Value) -> Result<Parsed, String> {
    let mut messages = Vec::new();
    let list = request["messages"]
        .as_array()
        .ok_or("messages must be a list")?;
    for message in list {
        let role = match message["role"].as_str() {
            Some("system" | "developer") => Role::System,
            Some("user") => Role::User,
            Some("assistant") => Role::Assistant,
            Some("tool") => Role::Tool,
            other => return Err(format!("unknown message role {other:?}")),
        };
        let mut parsed = Message::new(role, text_of(&message["content"]));
        for call in message["tool_calls"].as_array().into_iter().flatten() {
            let function = &call["function"];
            let name = function["name"]
                .as_str()
                .ok_or("a tool call needs a name")?
                .to_owned();
            // Arguments arrive as a JSON string in OpenAI's format.
            let arguments = match &function["arguments"] {
                Value::String(text) if text.trim().is_empty() => Map::new(),
                Value::String(text) => serde_json::from_str(text).map_err(|error| {
                    format!("the arguments of {name} are no JSON object: {error}")
                })?,
                Value::Object(arguments) => arguments.clone(),
                _ => Map::new(),
            };
            parsed.tool_calls.push(ToolCall { name, arguments });
        }
        messages.push(parsed);
    }
    let tools = if request["tool_choice"].as_str() == Some("none") {
        Vec::new()
    } else {
        request["tools"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|tool| tool["function"].clone())
            .collect()
    };
    let settings = Settings {
        temperature: float(&request["temperature"]),
        top_p: float(&request["top_p"]),
        top_k: request["top_k"]
            .as_u64()
            .and_then(|k| u32::try_from(k).ok()),
        seed: request["seed"].as_u64(),
        max_tokens: count(&request["max_completion_tokens"]).or(count(&request["max_tokens"])),
        stop: stops(&request["stop"]),
    };
    Ok(Parsed {
        messages,
        tools,
        settings,
    })
}

fn openai_finish(finish: Finish) -> &'static str {
    match finish {
        Finish::Stop | Finish::Cancelled => "stop",
        Finish::Length => "length",
        Finish::ToolCalls => "tool_calls",
    }
}

/// Return the OpenAI form of `call` with the identifier `id`.
fn openai_call(call: &ToolCall, id: &str) -> Value {
    json!({
        "id": id,
        "type": "function",
        "function": {"name": call.name, "arguments": Value::Object(call.arguments.clone()).to_string()},
    })
}

async fn chat_completions(State(shared): State<Arc<Shared>>, body: Bytes) -> Response {
    let request: Value = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => return openai_error(StatusCode::BAD_REQUEST, &error.to_string()),
    };
    let parsed = match parse_openai(&request) {
        Ok(parsed) => parsed,
        Err(error) => return openai_error(StatusCode::BAD_REQUEST, &error),
    };
    let model = request["model"].as_str().unwrap_or(&shared.name).to_owned();
    let stream = request["stream"].as_bool() == Some(true);
    let include_usage = request["stream_options"]["include_usage"].as_bool() == Some(true);
    let updates = match shared.submit(parsed) {
        Ok(updates) => updates,
        Err(error) => return openai_error(StatusCode::SERVICE_UNAVAILABLE, &error),
    };
    let id = shared.id("chatcmpl-");
    if !stream {
        let collected = match collect(updates).await {
            Ok(collected) => collected,
            Err(error) => return openai_error(StatusCode::INTERNAL_SERVER_ERROR, &error),
        };
        let calls: Vec<Value> = collected
            .calls
            .iter()
            .map(|call| openai_call(call, &shared.id("call_")))
            .collect();
        let mut message = json!({"role": "assistant", "content": collected.text});
        if !calls.is_empty() {
            message["tool_calls"] = Value::Array(calls);
        }
        let body = json!({
            "id": id,
            "object": "chat.completion",
            "created": now(),
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": openai_finish(collected.finish)}],
            "usage": {
                "prompt_tokens": collected.prompt_tokens,
                "completion_tokens": collected.completion_tokens,
                "total_tokens": collected.prompt_tokens + collected.completion_tokens,
            },
        });
        return axum::Json(body).into_response();
    }
    event_stream(move |events| async move {
        let created = now();
        let chunk = |delta: Value, finish: Option<&str>| {
            let body = json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            });
            Ok(SseEvent::default().data(body.to_string()))
        };
        let mut updates = updates;
        let mut calls = 0;
        while let Some(update) = updates.recv().await {
            let sent = match update {
                Update::Started(_) => {
                    events.send(chunk(json!({"role": "assistant", "content": ""}), None))
                }
                Update::Text(text) => events.send(chunk(json!({"content": text}), None)),
                Update::ToolCall(call) => {
                    let mut value = openai_call(&call, &shared.id("call_"));
                    value["index"] = json!(calls);
                    calls += 1;
                    events.send(chunk(json!({"tool_calls": [value]}), None))
                }
                Update::Done {
                    finish,
                    prompt_tokens,
                    completion_tokens,
                } => {
                    let _ = events.send(chunk(json!({}), Some(openai_finish(finish))));
                    if include_usage {
                        let usage = json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens,
                            },
                        });
                        let _ = events.send(Ok(SseEvent::default().data(usage.to_string())));
                    }
                    break;
                }
                Update::Failed(error) => {
                    let body = json!({"error": {"message": error, "type": "server_error"}});
                    let _ = events.send(Ok(SseEvent::default().data(body.to_string())));
                    break;
                }
            };
            if sent.is_err() {
                return;
            }
        }
        let _ = events.send(Ok(SseEvent::default().data("[DONE]")));
    })
}

/// Return an Anthropic error response.
fn anthropic_error(status: StatusCode, message: &str) -> Response {
    let body =
        json!({"type": "error", "error": {"type": "invalid_request_error", "message": message}});
    (status, axum::Json(body)).into_response()
}

/// Translate an Anthropic Messages request.
fn parse_anthropic(request: &Value) -> Result<Parsed, String> {
    let mut messages = Vec::new();
    let system = text_of(&request["system"]);
    if !system.is_empty() {
        messages.push(Message::new(Role::System, system));
    }
    let list = request["messages"]
        .as_array()
        .ok_or("messages must be a list")?;
    for message in list {
        // A request may carry system messages inside the list as well as in `system`.
        let role = match message["role"].as_str() {
            Some("system") => Role::System,
            Some("user") => Role::User,
            Some("assistant") => Role::Assistant,
            other => return Err(format!("unknown message role {other:?}")),
        };
        let blocks = match &message["content"] {
            Value::String(text) => {
                messages.push(Message::new(role, text.clone()));
                continue;
            }
            Value::Array(blocks) => blocks,
            _ => return Err("message content must be a string or a list of blocks".to_owned()),
        };
        let mut parsed = Message::new(role, String::new());
        for block in blocks {
            match block["type"].as_str() {
                Some("text") => parsed
                    .content
                    .push_str(block["text"].as_str().unwrap_or("")),
                Some("tool_use") => {
                    let name = block["name"]
                        .as_str()
                        .ok_or("a tool_use block needs a name")?
                        .to_owned();
                    let arguments = block["input"].as_object().cloned().unwrap_or_default();
                    parsed.tool_calls.push(ToolCall { name, arguments });
                }
                // Each tool result becomes a message of its own, as the chat template expects.
                Some("tool_result") => {
                    messages.push(Message::new(Role::Tool, text_of(&block["content"])));
                }
                // Images, documents, and earlier thinking have no place in a text model's prompt.
                _ => {}
            }
        }
        if !parsed.content.is_empty() || !parsed.tool_calls.is_empty() {
            messages.push(parsed);
        }
    }
    let tools = if request["tool_choice"]["type"].as_str() == Some("none") {
        Vec::new()
    } else {
        request["tools"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|tool| {
                json!({
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                })
            })
            .collect()
    };
    let settings = Settings {
        temperature: float(&request["temperature"]),
        top_p: float(&request["top_p"]),
        top_k: request["top_k"]
            .as_u64()
            .and_then(|k| u32::try_from(k).ok()),
        seed: None,
        max_tokens: count(&request["max_tokens"]),
        stop: stops(&request["stop_sequences"]),
    };
    Ok(Parsed {
        messages,
        tools,
        settings,
    })
}

fn anthropic_stop(finish: Finish) -> &'static str {
    match finish {
        Finish::Stop | Finish::Cancelled => "end_turn",
        Finish::Length => "max_tokens",
        Finish::ToolCalls => "tool_use",
    }
}

/// Return a server-sent event in Anthropic's form, named after its type.
#[expect(
    clippy::unnecessary_wraps,
    reason = "the event channel carries results, as the SSE stream requires"
)]
fn anthropic_event(body: &Value) -> Result<SseEvent, Infallible> {
    let name = body["type"].as_str().unwrap_or("message");
    Ok(SseEvent::default().event(name).data(body.to_string()))
}

async fn messages(State(shared): State<Arc<Shared>>, body: Bytes) -> Response {
    let request: Value = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => return anthropic_error(StatusCode::BAD_REQUEST, &error.to_string()),
    };
    let parsed = match parse_anthropic(&request) {
        Ok(parsed) => parsed,
        Err(error) => return anthropic_error(StatusCode::BAD_REQUEST, &error),
    };
    let model = request["model"].as_str().unwrap_or(&shared.name).to_owned();
    let stream = request["stream"].as_bool() == Some(true);
    let updates = match shared.submit(parsed) {
        Ok(updates) => updates,
        Err(error) => return anthropic_error(StatusCode::SERVICE_UNAVAILABLE, &error),
    };
    let id = shared.id("msg_");
    if !stream {
        let collected = match collect(updates).await {
            Ok(collected) => collected,
            Err(error) => return anthropic_error(StatusCode::INTERNAL_SERVER_ERROR, &error),
        };
        let mut content = Vec::new();
        if !collected.text.is_empty() {
            content.push(json!({"type": "text", "text": collected.text}));
        }
        for call in &collected.calls {
            content.push(json!({
                "type": "tool_use",
                "id": shared.id("toolu_"),
                "name": call.name,
                "input": call.arguments,
            }));
        }
        let body = json!({
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": anthropic_stop(collected.finish),
            "stop_sequence": null,
            "usage": {"input_tokens": collected.prompt_tokens, "output_tokens": collected.completion_tokens},
        });
        return axum::Json(body).into_response();
    }
    event_stream(move |events| async move {
        let mut updates = updates;
        // The index of the next content block, and whether a text block is open.
        let mut index = 0;
        let mut text_open = false;
        while let Some(update) = updates.recv().await {
            let sent = match update {
                Update::Started(prompt_tokens) => events.send(anthropic_event(&json!({
                    "type": "message_start",
                    "message": {
                        "id": id,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": null,
                        "stop_sequence": null,
                        "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
                    },
                }))),
                Update::Text(text) => {
                    if !text_open {
                        let _ = events.send(anthropic_event(&json!({
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {"type": "text", "text": ""},
                        })));
                        text_open = true;
                    }
                    events.send(anthropic_event(&json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": text},
                    })))
                }
                Update::ToolCall(call) => {
                    if text_open {
                        let _ = events.send(anthropic_event(
                            &json!({"type": "content_block_stop", "index": index}),
                        ));
                        text_open = false;
                        index += 1;
                    }
                    let _ = events.send(anthropic_event(&json!({
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "tool_use", "id": shared.id("toolu_"), "name": call.name, "input": {}},
                    })));
                    let _ = events.send(anthropic_event(&json!({
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": Value::Object(call.arguments).to_string()},
                    })));
                    index += 1;
                    events.send(anthropic_event(
                        &json!({"type": "content_block_stop", "index": index - 1}),
                    ))
                }
                Update::Done {
                    finish,
                    completion_tokens,
                    ..
                } => {
                    if text_open {
                        let _ = events.send(anthropic_event(
                            &json!({"type": "content_block_stop", "index": index}),
                        ));
                    }
                    let _ = events.send(anthropic_event(&json!({
                        "type": "message_delta",
                        "delta": {"stop_reason": anthropic_stop(finish), "stop_sequence": null},
                        "usage": {"output_tokens": completion_tokens},
                    })));
                    let _ = events.send(anthropic_event(&json!({"type": "message_stop"})));
                    return;
                }
                Update::Failed(error) => {
                    let _ = events.send(anthropic_event(&json!({
                        "type": "error",
                        "error": {"type": "api_error", "message": error},
                    })));
                    return;
                }
            };
            if sent.is_err() {
                return;
            }
        }
    })
}

async fn decisions(State(shared): State<Arc<Shared>>, body: Bytes) -> Response {
    decide(&shared, &body, Route::Decisions).await
}

async fn systemone(State(shared): State<Arc<Shared>>, body: Bytes) -> Response {
    decide(&shared, &body, Route::SystemOne).await
}

async fn score(State(shared): State<Arc<Shared>>, body: Bytes) -> Response {
    decide(&shared, &body, Route::Score).await
}

/// Queue the decision request in `body` for the worker under `route` and return its response.
async fn decide(shared: &Shared, body: &Bytes, route: Route) -> Response {
    let request: Value = match serde_json::from_slice(body) {
        Ok(request) => request,
        Err(error) => return openai_error(StatusCode::BAD_REQUEST, &error.to_string()),
    };
    let (reply, receiver) = oneshot::channel();
    let job = Job::Decide {
        route,
        request,
        reply,
    };
    if shared.jobs.send(job).is_err() {
        return openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "the model worker has stopped",
        );
    }
    match receiver.await {
        Ok(Ok(response)) => axum::Json(response).into_response(),
        // SystemOne's SDKs expect 422 for a request that breaks the schema, as SGLang answers.
        Ok(Err(Refusal::Invalid(message))) if route == Route::SystemOne => {
            openai_error(StatusCode::UNPROCESSABLE_ENTITY, &message)
        }
        Ok(Err(Refusal::Invalid(message) | Refusal::Refused(message))) => {
            openai_error(StatusCode::BAD_REQUEST, &message)
        }
        Ok(Err(Refusal::Failed(error))) => {
            openai_error(StatusCode::INTERNAL_SERVER_ERROR, &error.to_string())
        }
        Err(_) => openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "the model worker has stopped",
        ),
    }
}

async fn count_tokens(State(shared): State<Arc<Shared>>, body: Bytes) -> Response {
    let request: Value = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => return anthropic_error(StatusCode::BAD_REQUEST, &error.to_string()),
    };
    let parsed = match parse_anthropic(&request) {
        Ok(parsed) => parsed,
        Err(error) => return anthropic_error(StatusCode::BAD_REQUEST, &error),
    };
    let (reply, receiver) = oneshot::channel();
    let job = Job::Count {
        messages: parsed.messages,
        tools: parsed.tools,
        reply,
    };
    if shared.jobs.send(job).is_err() {
        return anthropic_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "the model worker has stopped",
        );
    }
    match receiver.await {
        Ok(Ok(tokens)) => axum::Json(json!({"input_tokens": tokens})).into_response(),
        Ok(Err(error)) => anthropic_error(StatusCode::BAD_REQUEST, &error),
        Err(_) => anthropic_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "the model worker has stopped",
        ),
    }
}
