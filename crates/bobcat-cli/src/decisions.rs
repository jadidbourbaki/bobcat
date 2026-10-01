//! Decisions from language models: SGLang's `/v1/decisions` and `/v1/score`, and the SystemOne
//! API of `/v1/systemone`.
//!
//! A decision asks typed questions about an input and returns a probability for every option,
//! with no generated text. Each question becomes one user message through the model's chat
//! template, with thinking off. The options carry one-token labels, `A` to `Z` for choices, `0`
//! to `9` for score levels, and `yes` and `no`. One prefill reads the next-token log-probabilities
//! of the labels. The prompt wording, the labels, the checks, and the response shapes follow
//! SGLang's `serving_decisions.py` and `systemone/serving.py`, so clients of either server work
//! unchanged. A model file that holds a Clef decision head answers `/v1/systemone` with that
//! head, as [`crate::systemone`] describes.

use std::collections::HashSet;

use serde_json::{Map, Value, json};

use crate::Error;
use crate::engine::Engine;
use crate::systemone;

/// The version of the prompt wording and labels, which SGLang numbers the same way. A change to
/// any prompt a question can render needs a new version.
const PROMPT_FORMAT_VERSION: u64 = 1;

/// The most options of a choice question, labeled `A` to `Z`.
const MAX_OPTIONS: usize = 26;

/// The most levels of a score question, labeled `0` to `9`.
const MAX_LEVELS: usize = 10;

/// The answer text of a finished reply, rendered only to see what the template puts before an
/// answer.
const REPLY_SENTINEL: &str = "DECISION_ANSWER";

/// The chat template argument that turns thinking on and off in Qwen and Clef templates.
const THINKING_TOGGLE: &str = "enable_thinking";

/// The tags of a reasoning block.
const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// Why a decision request got no answer.
#[derive(Debug)]
pub(crate) enum Refusal {
    /// The request breaks the API's schema. SystemOne answers 422 and the other routes 400.
    Invalid(String),
    /// The request is well formed, but this model cannot answer it. Every route answers 400.
    Refused(String),
    /// The engine failed.
    Failed(Error),
}

impl From<Error> for Refusal {
    fn from(error: Error) -> Self {
        Self::Failed(error)
    }
}

/// The kind of a question.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Choice,
    Score,
    YesNo,
}

/// A question as the renderer and the scorer see it, from either API.
struct View {
    kind: Kind,
    /// The question's text, or null when it has none.
    question: Value,
    /// The option names, the level indices, or `yes` and `no`, in label order.
    names: Vec<String>,
    /// The option descriptions, the levels, or the `yes` and `no` descriptions.
    details: Vec<Value>,
}

/// A question encoded for scoring.
struct Encoded {
    prompt: Vec<u32>,
    labels: Vec<u32>,
}

/// Answer the `/v1/decisions` request `request`.
pub(crate) fn decisions(engine: &mut Engine<'_>, request: &Value) -> Result<Value, Refusal> {
    let object = request
        .as_object()
        .ok_or_else(|| invalid("the request must be a JSON object"))?;
    check_keys(
        object,
        &[
            "input",
            "questions",
            "temperature",
            "chat_template_kwargs",
            "prompt_format_version",
            "return_prompt_token_ids",
            "model",
        ],
    )
    .map_err(|error| invalid(format!("the request: {error}")))?;
    let input = &request["input"];
    text_field(input, "input", true).map_err(Refusal::Invalid)?;
    let questions = request["questions"]
        .as_array()
        .filter(|questions| !questions.is_empty())
        .ok_or_else(|| invalid("questions must be a list of at least one question"))?;
    let mut ids = Vec::new();
    let mut views = Vec::new();
    for (index, question) in questions.iter().enumerate() {
        let (id, view) = decisions_question(question)
            .map_err(|error| invalid(format!("questions[{index}]: {error}")))?;
        if ids.contains(&id) {
            return Err(invalid(format!(
                "question id {id:?} repeats another question"
            )));
        }
        ids.push(id);
        views.push(view);
    }
    let temperature = temperature(&request["temperature"])?;
    match &request["prompt_format_version"] {
        Value::Null => {}
        value if value.as_u64() == Some(PROMPT_FORMAT_VERSION) => {}
        Value::Number(version) => {
            return Err(Refusal::Refused(format!(
                "prompt_format_version {version} is not served, this server uses version \
                 {PROMPT_FORMAT_VERSION}"
            )));
        }
        _ => return Err(invalid("prompt_format_version must be an integer")),
    }
    let return_ids = match &request["return_prompt_token_ids"] {
        Value::Null => false,
        Value::Bool(value) => *value,
        _ => return Err(invalid("return_prompt_token_ids must be a boolean")),
    };
    let model = model_name(&request["model"])?;
    let kwargs = template_kwargs(&request["chat_template_kwargs"])?;

    let text = render_text(input);
    let encoded = views
        .iter()
        .zip(&ids)
        .map(|(view, id)| {
            encode_question(engine, &text, view, &kwargs)
                .map_err(|error| Refusal::Refused(format!("question {id:?}: {error}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let scores = score(engine, &encoded)?;

    let mut answers = Map::new();
    for (((id, view), encoded), logprobs) in ids.iter().zip(&views).zip(&encoded).zip(&scores) {
        let probabilities = softmax(logprobs, temperature);
        let mut answer = Map::new();
        answer.insert("type".to_owned(), json!(kind_name(view.kind)));
        answer.insert(
            "probabilities".to_owned(),
            Value::Object(by_name(view, &probabilities)),
        );
        answer.insert("label_mass".to_owned(), json!(label_mass(logprobs)));
        match view.kind {
            Kind::Choice => {
                answer.insert(
                    "choice".to_owned(),
                    json!(view.names[argmax(&probabilities)]),
                );
            }
            Kind::Score => {
                answer.insert("score".to_owned(), json!(mean_level(&probabilities)));
            }
            Kind::YesNo => {}
        }
        if return_ids {
            answer.insert("prompt_token_ids".to_owned(), json!(encoded.prompt));
            answer.insert("label_token_ids".to_owned(), json!(encoded.labels));
        }
        answers.insert(id.clone(), Value::Object(answer));
    }
    let prompt_tokens: usize = encoded.iter().map(|encoded| encoded.prompt.len()).sum();
    Ok(json!({
        "object": "decisions",
        "model": model,
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "answers": answers,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
            "completion_tokens": 0,
        },
    }))
}

/// Return the id and the view of the `/v1/decisions` question `question`.
fn decisions_question(question: &Value) -> Result<(String, View), String> {
    let object = question.as_object().ok_or("a question must be an object")?;
    let id = match &question["id"] {
        Value::String(id) if !id.trim().is_empty() => id.clone(),
        _ => return Err("id must be a nonblank string".to_owned()),
    };
    let kind = match question["type"].as_str() {
        Some("choice") => Kind::Choice,
        Some("score") => Kind::Score,
        Some("yes_no") => Kind::YesNo,
        _ => return Err("type must be choice, score, or yes_no".to_owned()),
    };
    text_field(&question["question"], "question", true)?;
    let view = match kind {
        Kind::Choice => {
            check_keys(object, &["id", "type", "question", "options"])?;
            let options = question["options"]
                .as_array()
                .filter(|options| (2..=MAX_OPTIONS).contains(&options.len()))
                .ok_or(format!("options must hold 2 to {MAX_OPTIONS} options"))?;
            let mut names = Vec::new();
            let mut details = Vec::new();
            for option in options {
                let option = option.as_object().ok_or("an option must be an object")?;
                check_keys(option, &["name", "description"])?;
                let name = option
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or("an option needs a name")?;
                let description = option.get("description").cloned().unwrap_or(Value::Null);
                text_field(&description, "description", false)?;
                names.push(name.to_owned());
                details.push(description);
            }
            check_option_names(&names)?;
            View {
                kind,
                question: question["question"].clone(),
                names,
                details,
            }
        }
        Kind::Score => {
            check_keys(object, &["id", "type", "question", "levels"])?;
            let levels = question["levels"]
                .as_array()
                .filter(|levels| (2..=MAX_LEVELS).contains(&levels.len()))
                .ok_or(format!("levels must hold 2 to {MAX_LEVELS} levels"))?;
            for level in levels {
                text_field(level, "a level", true)?;
            }
            View {
                kind,
                question: question["question"].clone(),
                names: (0..levels.len()).map(|level| level.to_string()).collect(),
                details: levels.clone(),
            }
        }
        Kind::YesNo => {
            check_keys(object, &["id", "type", "question", "yes", "no"])?;
            text_field(&question["yes"], "yes", false)?;
            text_field(&question["no"], "no", false)?;
            View {
                kind,
                question: question["question"].clone(),
                names: vec!["yes".to_owned(), "no".to_owned()],
                details: vec![question["yes"].clone(), question["no"].clone()],
            }
        }
    };
    Ok((id, view))
}

/// Answer the SystemOne request `request` as the model `name`. A model with a Clef head answers
/// with the head, and any other model answers through the scoring of `/v1/decisions`.
pub(crate) fn systemone(
    engine: &mut Engine<'_>,
    request: &Value,
    name: &str,
) -> Result<Value, Refusal> {
    if engine.has_decision_head() {
        systemone::validate(request).map_err(Refusal::Invalid)?;
        return Ok(engine.decide(request)?);
    }
    if !request.is_object() {
        return Err(invalid("the request must be a JSON object"));
    }
    if !request["model"].is_string() {
        return Err(invalid("model must be a string"));
    }
    let state = &request["state"];
    if request.get("state").is_none() {
        return Err(invalid("state is required"));
    }
    text_field(state, "state", false).map_err(Refusal::Invalid)?;
    for field in [
        "temperature",
        "prompt_format_version",
        "return_prompt_token_ids",
    ] {
        if !request[field].is_null() {
            return Err(invalid(format!(
                "{field} is not part of this API, use /v1/decisions for it"
            )));
        }
    }
    let questions = request["questions"]
        .as_object()
        .filter(|questions| !questions.is_empty())
        .ok_or_else(|| invalid("questions must map ids to at least one question"))?;
    let mut views = Vec::new();
    for (id, question) in questions {
        let view = systemone_question(question)
            .map_err(|error| invalid(format!("questions.{id}: {error}")))?;
        views.push(view);
    }
    let kwargs = template_kwargs(&request["chat_template_kwargs"])?;

    let text = render_text(state);
    let encoded = views
        .iter()
        .zip(questions.keys())
        .map(|(view, id)| {
            if view.names.len() > MAX_OPTIONS {
                return Err(Refusal::Refused(format!(
                    "question {id:?}: it has {} options, and bobcat labels at most {MAX_OPTIONS}",
                    view.names.len()
                )));
            }
            encode_question(engine, &text, view, &kwargs)
                .map_err(|error| Refusal::Refused(format!("question {id:?}: {error}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let scores = score(engine, &encoded)?;

    let mut answers = Map::new();
    for ((id, view), logprobs) in questions.keys().zip(&views).zip(&scores) {
        let probabilities = softmax(logprobs, 1.0);
        let mass = label_mass(logprobs);
        if !probabilities
            .iter()
            .chain([&mass])
            .all(|value| value.is_finite())
        {
            return Err(Refusal::Failed(
                format!("question {id:?} scored non-finite values").into(),
            ));
        }
        let normalized = normalize(&probabilities);
        let answer = match view.kind {
            Kind::YesNo => json!({"type": "noul", "noul": probabilities[0], "x_label_mass": mass}),
            Kind::Choice => json!({
                "type": "choice",
                "choice": view.names[argmax(&probabilities)],
                "confidence": choice_confidence(&normalized),
                "probabilities": by_name(view, &probabilities),
                "x_label_mass": mass,
            }),
            Kind::Score => {
                let legend: Map<String, Value> = view
                    .names
                    .iter()
                    .cloned()
                    .zip(view.details.iter().cloned())
                    .collect();
                json!({
                    "type": "score",
                    "score": mean_level(&probabilities),
                    "confidence": score_confidence(&normalized),
                    "legend": legend,
                    "probabilities": by_name(view, &probabilities),
                    "x_label_mass": mass,
                })
            }
        };
        answers.insert(id.clone(), answer);
    }
    let input_tokens: usize = encoded.iter().map(|encoded| encoded.prompt.len()).sum();
    Ok(json!({
        "model": name,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }))
}

/// Return the view of the SystemOne question `question`.
fn systemone_question(question: &Value) -> Result<View, String> {
    let object = question.as_object().ok_or("a question must be an object")?;
    check_keys(object, &["type", "instructions", "criteria"])?;
    let instructions = question["instructions"].clone();
    text_field(&instructions, "instructions", false)?;
    let criteria = &question["criteria"];
    match question["type"].as_str() {
        Some("noul") => {
            let (yes, no) = match criteria {
                Value::Null => (Value::Null, Value::Null),
                Value::Object(criteria) => {
                    check_keys(criteria, &["true", "false"])?;
                    let yes = criteria.get("true").cloned().unwrap_or(Value::Null);
                    let no = criteria.get("false").cloned().unwrap_or(Value::Null);
                    text_field(&yes, "criteria.true", false)?;
                    text_field(&no, "criteria.false", false)?;
                    (yes, no)
                }
                _ => return Err("criteria must be an object".to_owned()),
            };
            if [&instructions, &yes, &no].into_iter().all(blank) {
                return Err(
                    "a noul question needs instructions or a true or false description to \
                     decide on"
                        .to_owned(),
                );
            }
            Ok(View {
                kind: Kind::YesNo,
                question: instructions,
                names: vec!["yes".to_owned(), "no".to_owned()],
                details: vec![yes, no],
            })
        }
        Some("choice") => {
            let criteria = criteria
                .as_object()
                .filter(|criteria| (1..=255).contains(&criteria.len()))
                .ok_or("criteria must map 1 to 255 option names to descriptions")?;
            for description in criteria.values() {
                text_field(description, "a description", false)?;
            }
            let names: Vec<String> = criteria.keys().cloned().collect();
            check_option_names(&names)?;
            Ok(View {
                kind: Kind::Choice,
                question: instructions,
                names,
                details: criteria.values().cloned().collect(),
            })
        }
        Some("score") => {
            let levels = criteria
                .as_array()
                .filter(|levels| (1..=MAX_LEVELS).contains(&levels.len()))
                .ok_or(format!("criteria must list 1 to {MAX_LEVELS} levels"))?;
            for level in levels {
                text_field(level, "a level", true)?;
            }
            Ok(View {
                kind: Kind::Score,
                question: instructions,
                names: (0..levels.len()).map(|level| level.to_string()).collect(),
                details: levels.clone(),
            })
        }
        _ => Err("type must be noul, choice, or score".to_owned()),
    }
}

/// Answer the `/v1/score` request `request`. The response holds the probability of each label
/// token after each item joined to the query.
pub(crate) fn score_request(engine: &mut Engine<'_>, request: &Value) -> Result<Value, Refusal> {
    if !request.is_object() {
        return Err(invalid("the request must be a JSON object"));
    }
    for field in [
        "embed_override_token_id",
        "query_embed_overrides",
        "item_embed_overrides",
        "score_extraction_token",
    ] {
        if !request[field].is_null() {
            return Err(Refusal::Refused(format!("bobcat does not support {field}")));
        }
    }
    if request["return_pooled_hidden_states"] == json!(true) {
        return Err(Refusal::Refused(
            "bobcat does not support return_pooled_hidden_states".to_owned(),
        ));
    }
    let flag = |name: &str| match &request[name] {
        Value::Null => Ok(false),
        Value::Bool(value) => Ok(*value),
        _ => Err(invalid(format!("{name} must be a boolean"))),
    };
    let apply_softmax = flag("apply_softmax")?;
    let return_logprobs = flag("return_token_logprobs")?;
    let item_first = flag("item_first")?;
    let temperature = temperature(&request["temperature"])?;
    if (temperature - 1.0).abs() > 0.0 && !apply_softmax {
        return Err(invalid("temperature requires apply_softmax=True"));
    }
    let model = model_name(&request["model"])?;

    let tokenizer = engine.tokenizer();
    let encode = |text: &str| -> Result<Vec<u32>, Refusal> {
        Ok(tokenizer.encode(text, false)?.get_ids().to_vec())
    };
    let (query, text_query) = match &request["query"] {
        Value::Null => (Vec::new(), None),
        Value::String(text) => {
            // A text query starts a sequence, so it takes the token the model begins with.
            let mut ids: Vec<u32> = engine.added_bos().into_iter().collect();
            ids.extend(encode(text)?);
            (ids, Some(true))
        }
        Value::Array(_) => (token_ids(&request["query"], "query")?, Some(false)),
        _ => return Err(invalid("query must be text or a list of token ids")),
    };
    let items: Vec<Vec<u32>> = match &request["items"] {
        Value::String(text) if text_query != Some(false) => vec![encode(text)?],
        Value::Array(items) if items.iter().all(Value::is_string) && text_query != Some(false) => {
            items
                .iter()
                .map(|item| encode(item.as_str().unwrap_or_default()))
                .collect::<Result<_, _>>()?
        }
        Value::Array(items) if text_query != Some(true) => items
            .iter()
            .enumerate()
            .map(|(index, item)| token_ids(item, &format!("items[{index}]")))
            .collect::<Result<_, _>>()?,
        _ => {
            return Err(invalid(
                "items must be text or a list of texts with a text query, and lists of token \
                 ids with a token id query",
            ));
        }
    };
    if items.is_empty() {
        return Err(invalid("items must hold at least one item"));
    }
    let labels: Vec<Vec<u32>> = match &request["label_token_ids"] {
        Value::Array(labels) if labels.iter().all(Value::is_array) => {
            if labels.len() != items.len() {
                return Err(invalid("label_token_ids must hold one list per item"));
            }
            labels
                .iter()
                .map(|labels| token_ids(labels, "label_token_ids"))
                .collect::<Result<_, _>>()?
        }
        Value::Array(_) => {
            let shared = token_ids(&request["label_token_ids"], "label_token_ids")?;
            vec![shared; items.len()]
        }
        _ => return Err(invalid("label_token_ids must list token ids")),
    };
    let n_vocab = engine.n_vocab();
    if let Some(&label) = labels.iter().flatten().find(|&&label| label >= n_vocab) {
        return Err(Refusal::Refused(format!(
            "label token {label} lies outside the vocabulary of {n_vocab} tokens"
        )));
    }
    let prompts: Vec<Vec<u32>> = items
        .into_iter()
        .map(|item| {
            if item_first {
                [item, query.clone()].concat()
            } else {
                [query.clone(), item].concat()
            }
        })
        .collect();
    if let Some(prompt) = prompts
        .iter()
        .find(|prompt| prompt.is_empty() || prompt.len() >= engine.context() as usize)
    {
        return Err(Refusal::Refused(format!(
            "a scored sequence of {} tokens must hold at least one token and fit the context of \
             {} tokens",
            prompt.len(),
            engine.context()
        )));
    }

    let logprobs = engine.score(&prompts, &labels)?;
    let scores: Vec<Vec<f64>> = logprobs
        .iter()
        .map(|logprobs| {
            if apply_softmax {
                softmax(logprobs, temperature)
            } else {
                logprobs.iter().map(|&logprob| logprob.exp()).collect()
            }
        })
        .collect();
    let prompt_tokens: usize = prompts.iter().map(Vec::len).sum();
    let mut response = json!({
        "scores": scores,
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
            "completion_tokens": 0,
        },
        "object": "scoring",
    });
    if return_logprobs {
        response["token_logprobs"] = json!(logprobs);
    }
    Ok(response)
}

/// Return the request's `temperature`, which divides the label logits before their softmax and
/// defaults to 1.
fn temperature(value: &Value) -> Result<f64, Refusal> {
    match value {
        Value::Null => Ok(1.0),
        value => value
            .as_f64()
            .filter(|t| t.is_finite() && *t > 0.0)
            .ok_or_else(|| invalid("temperature must be a number above 0")),
    }
}

/// Return the request's `model`, which the response echoes, as SGLang's `default` when the
/// request names none.
fn model_name(value: &Value) -> Result<String, Refusal> {
    match value {
        Value::Null => Ok("default".to_owned()),
        Value::String(model) => Ok(model.clone()),
        _ => Err(invalid("model must be a string")),
    }
}

/// Return the token ids in `value`, a list of nonnegative integers.
fn token_ids(value: &Value, name: &str) -> Result<Vec<u32>, Refusal> {
    value
        .as_array()
        .ok_or_else(|| invalid(format!("{name} must be a list of token ids")))?
        .iter()
        .map(|id| {
            id.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .ok_or_else(|| invalid(format!("{name} must hold token ids")))
        })
        .collect()
}

/// Return the chat template arguments of a request: thinking off, then the request's
/// `chat_template_kwargs`, which may not turn thinking on.
fn template_kwargs(value: &Value) -> Result<Map<String, Value>, Refusal> {
    let mut kwargs = Map::new();
    kwargs.insert(THINKING_TOGGLE.to_owned(), Value::Bool(false));
    match value {
        Value::Null => {}
        Value::Object(request) => {
            if let Some(toggle) = request.get(THINKING_TOGGLE)
                && *toggle != Value::Bool(false)
            {
                return Err(Refusal::Refused(format!(
                    "chat_template_kwargs sets {THINKING_TOGGLE:?} to {toggle}, but decisions \
                     need it false or unset"
                )));
            }
            kwargs.extend(request.clone());
        }
        _ => return Err(invalid("chat_template_kwargs must be an object")),
    }
    Ok(kwargs)
}

/// Return the prompt and label tokens of `view` about the input `text`, after checking that the
/// answer position follows any reasoning block and that each label is one distinct token there.
fn encode_question(
    engine: &Engine<'_>,
    text: &str,
    view: &View,
    kwargs: &Map<String, Value>,
) -> Result<Encoded, String> {
    let labels = labels(view);
    let content = render_question(text, view, &labels);
    let prompt = engine
        .render_message(&content, None, kwargs)
        .map_err(|error| format!("the chat template failed: {error}"))?;

    // The message's last line is fixed text, so the generation prompt starts after it.
    let closing = content.rsplit('\n').next().unwrap_or_default();
    let generation = prompt
        .rfind(closing)
        .map_or(prompt.as_str(), |at| &prompt[at + closing.len()..]);
    if generation.rfind(THINK_OPEN) > generation.rfind(THINK_CLOSE) {
        return Err(
            "the chat template leaves a reasoning block open at the answer position, so this \
             model cannot answer decisions"
                .to_owned(),
        );
    }
    // The template's own finished reply shows whether answers start with a reasoning block that
    // the generation prompt leaves out.
    if let Ok(reply) = engine.render_message(closing, Some(REPLY_SENTINEL), kwargs)
        && let Some(begin) = reply.rfind(closing)
        && let Some(answer) = reply[begin..].find(REPLY_SENTINEL)
        && reply[begin..begin + answer].matches(THINK_OPEN).count()
            > generation.matches(THINK_OPEN).count()
    {
        return Err(
            "the chat template starts every answer with a reasoning block, so this model \
             cannot answer decisions"
                .to_owned(),
        );
    }

    let tokenizer = engine.tokenizer();
    let encode = |text: &str| -> Result<Vec<u32>, String> {
        Ok(tokenizer
            .encode(text, false)
            .map_err(|error| error.to_string())?
            .get_ids()
            .to_vec())
    };
    let prompt_ids = encode(&prompt)?;
    if prompt_ids.len() >= engine.context() as usize {
        return Err(format!(
            "the prompt has {} tokens, which does not fit the context of {} tokens",
            prompt_ids.len(),
            engine.context()
        ));
    }

    // Added tokens split the text before tokenization, so the text after the last one tokenizes
    // on its own, and the label check need not encode the whole input again.
    let added = tokenizer.get_added_tokens_decoder();
    let mut context = (prompt.as_str(), prompt_ids.as_slice());
    if let Some(last) = prompt_ids.iter().rposition(|id| added.contains_key(id))
        && let Some(token) = added.get(&prompt_ids[last])
        && let Some(start) = prompt.rfind(&token.content)
    {
        let suffix = &prompt[start + token.content.len()..];
        if encode(suffix)? == prompt_ids[last + 1..] {
            context = (suffix, &prompt_ids[last + 1..]);
        }
    }
    let (context_text, context_ids) = context;
    let mut label_ids = Vec::new();
    for label in &labels {
        let ids = encode(&format!("{context_text}{label}"))?;
        match ids.split_last() {
            Some((&id, rest)) if rest == context_ids && !label_ids.contains(&id) => {
                label_ids.push(id);
            }
            _ => {
                return Err(format!(
                    "the answer label {label:?} is not one distinct token after the chat \
                     prompt for this tokenizer, so this model cannot answer decisions"
                ));
            }
        }
    }
    Ok(Encoded {
        prompt: prompt_ids,
        labels: label_ids,
    })
}

/// Return the full-vocabulary log-probabilities of the labels of each encoded question.
fn score(engine: &mut Engine<'_>, encoded: &[Encoded]) -> Result<Vec<Vec<f64>>, Refusal> {
    let prompts: Vec<Vec<u32>> = encoded
        .iter()
        .map(|encoded| encoded.prompt.clone())
        .collect();
    let labels: Vec<Vec<u32>> = encoded
        .iter()
        .map(|encoded| encoded.labels.clone())
        .collect();
    Ok(engine.score(&prompts, &labels)?)
}

/// Return the labels of `view`: `A` onward for options, the level indices, or `yes` and `no`.
fn labels(view: &View) -> Vec<String> {
    match view.kind {
        Kind::Choice => ('A'..='Z')
            .take(view.names.len())
            .map(String::from)
            .collect(),
        Kind::Score | Kind::YesNo => view.names.clone(),
    }
}

/// Return the user message that asks `view` about `text`, in the wording of version 1.
fn render_question(text: &str, view: &View, labels: &[String]) -> String {
    let question = if blank(&view.question) {
        String::new()
    } else {
        render_text(&view.question)
    };
    let mut lines = Vec::new();
    match view.kind {
        Kind::Choice => {
            if !question.is_empty() {
                lines.push(format!("Question: {question}"));
            }
            for ((label, name), description) in labels.iter().zip(&view.names).zip(&view.details) {
                let detail = render_text(description);
                lines.push(if detail.is_empty() {
                    format!("{label}: {name}")
                } else {
                    format!("{label}: {name} - {detail}")
                });
            }
            lines.push("Answer with the letter of one option only.".to_owned());
        }
        Kind::Score => {
            if !question.is_empty() {
                lines.push(format!("Question: {question}"));
            }
            for (label, level) in labels.iter().zip(&view.details) {
                lines.push(format!("{label}: {}", render_text(level)));
            }
            lines.push("Answer with the number of one level only.".to_owned());
        }
        Kind::YesNo => {
            lines.push(if question.is_empty() {
                "Is the following true?".to_owned()
            } else {
                format!("Is the following true? {question}")
            });
            for (label, description) in labels.iter().zip(&view.details) {
                let detail = render_text(description);
                if !detail.is_empty() {
                    lines.push(format!("{label}: {detail}"));
                }
            }
            lines.push("Answer with yes or no only.".to_owned());
        }
    }
    [text.to_owned(), String::new()]
        .into_iter()
        .chain(lines)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Return `value` as prompt text: a string as itself, null as nothing, and anything else as
/// compact JSON.
fn render_text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        Value::Bool(_) | Value::Number(_) | Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

/// Return whether `value` is blank: whitespace, null, or empty, as Python's `not` sees it.
fn blank(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => true,
        Value::Bool(true) => false,
        Value::Number(number) => number.as_f64() == Some(0.0),
        Value::String(text) => text.trim().is_empty(),
        Value::Array(items) => items.is_empty(),
        Value::Object(items) => items.is_empty(),
    }
}

/// Check that `value` is text the prompt can hold: a string, an object, or a list. With
/// `required`, the text must not be blank. Without it, null stands for no text.
fn text_field(value: &Value, name: &str, required: bool) -> Result<(), String> {
    match value {
        Value::Null if !required => Ok(()),
        Value::String(_) | Value::Array(_) | Value::Object(_) if !(required && blank(value)) => {
            Ok(())
        }
        _ if required => Err(format!("{name} must be a nonblank string, object, or list")),
        _ => Err(format!("{name} must be a string, object, list, or null")),
    }
}

/// Refuse option names that would make the rendered option lines ambiguous.
fn check_option_names(names: &[String]) -> Result<(), String> {
    let mut seen = HashSet::new();
    for name in names {
        let key = name.trim().to_lowercase();
        if key.is_empty() {
            return Err("option names must be nonempty".to_owned());
        }
        // Each option is one line of the prompt.
        if name
            .chars()
            .any(|c| c.is_control() || c == '\u{2028}' || c == '\u{2029}')
        {
            return Err(format!(
                "option name {name:?} must not contain control or line break characters"
            ));
        }
        if !seen.insert(key) {
            return Err(format!("option name {name:?} repeats another option"));
        }
    }
    Ok(())
}

/// Refuse any key of `object` outside `allowed`.
fn check_keys(object: &Map<String, Value>, allowed: &[&str]) -> Result<(), String> {
    match object.keys().find(|key| !allowed.contains(&key.as_str())) {
        Some(key) => Err(format!("unknown field {key:?}")),
        None => Ok(()),
    }
}

fn invalid(message: impl Into<String>) -> Refusal {
    Refusal::Invalid(message.into())
}

fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Choice => "choice",
        Kind::Score => "score",
        Kind::YesNo => "yes_no",
    }
}

/// Return the probabilities keyed by the names of `view`.
fn by_name(view: &View, probabilities: &[f64]) -> Map<String, Value> {
    view.names
        .iter()
        .cloned()
        .zip(probabilities.iter().map(|&p| json!(p)))
        .collect()
}

/// Return the softmax of the log-probabilities `logprobs` divided by `temperature`. The
/// vocabulary normalizer cancels, so the result is the softmax of the label logits.
pub(crate) fn softmax(logprobs: &[f64], temperature: f64) -> Vec<f64> {
    let max = logprobs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logprobs
        .iter()
        .map(|&logprob| ((logprob - max) / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    weights.iter().map(|weight| weight / total).collect()
}

/// Return the full-vocabulary probability of all the labels.
fn label_mass(logprobs: &[f64]) -> f64 {
    logprobs.iter().map(|logprob| logprob.exp()).sum()
}

/// Return the index of the largest of `values`. Ties go to the first.
fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .fold(0, |best, (i, &v)| if v > values[best] { i } else { best })
}

/// Return the probability-weighted mean level index.
fn mean_level(probabilities: &[f64]) -> f64 {
    (0_u32..)
        .zip(probabilities)
        .map(|(level, p)| f64::from(level) * p)
        .sum()
}

/// Return `probabilities` scaled to sum to one, or uniform when they sum to nothing.
fn normalize(probabilities: &[f64]) -> Vec<f64> {
    let total: f64 = probabilities.iter().sum();
    if total > 0.0 {
        probabilities.iter().map(|p| p / total).collect()
    } else {
        vec![1.0 / probabilities.len() as f64; probabilities.len()]
    }
}

/// Return how far the top option stands above a uniform guess, from 0 to 1, as TypeSafe defines a
/// choice's confidence.
fn choice_confidence(q: &[f64]) -> f64 {
    let n = q.len() as f64;
    if q.len() == 1 {
        return 1.0;
    }
    let top = q.iter().copied().fold(0.0, f64::max);
    ((n * top - 1.0) / (n - 1.0)).clamp(0.0, 1.0)
}

/// Return one minus the spread of `q` around its top level relative to a uniform spread, floored
/// at 0, as TypeSafe defines a score's confidence.
fn score_confidence(q: &[f64]) -> f64 {
    let n = q.len();
    if n == 1 {
        return 1.0;
    }
    let top = argmax(q);
    let spread: f64 = (0..n).map(|i| q[i] * (i as f64 - top as f64).abs()).sum();
    let middle = (n - 1) as f64 / 2.0;
    let uniform: f64 = (0..n).map(|i| (i as f64 - middle).abs()).sum::<f64>() / n as f64;
    (1.0 - spread / uniform).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Check the wording of `PROMPT_FORMAT_VERSION` 1 against the messages SGLang renders for the
    /// questions of its documentation. A change to the wording needs a new version and new
    /// expected messages.
    #[test]
    fn questions_render_as_sglang_version_1() -> Result<(), String> {
        let request = json!([
            {"id": "team", "type": "choice", "question": "Which team?", "options": [
                {"name": "billing", "description": "Payments"}, {"name": "sales"}]},
            {"id": "mood", "type": "score", "question": {"ask": "How upset?"},
             "levels": ["Calm", "Angry"]},
            {"id": "urgent", "type": "yes_no", "question": "It is urgent.", "no": "It can wait."},
        ]);
        let want = [
            "Ticket text\n\nQuestion: Which team?\nA: billing - Payments\nB: sales\nAnswer with \
             the letter of one option only.",
            "Ticket text\n\nQuestion: {\"ask\":\"How upset?\"}\n0: Calm\n1: Angry\nAnswer with \
             the number of one level only.",
            "Ticket text\n\nIs the following true? It is urgent.\nno: It can wait.\nAnswer with \
             yes or no only.",
        ];
        for (question, want) in request
            .as_array()
            .ok_or("the questions form a list")?
            .iter()
            .zip(want)
        {
            let (_, view) = decisions_question(question)?;
            assert_eq!(render_question("Ticket text", &view, &labels(&view)), want);
        }
        Ok(())
    }
}
