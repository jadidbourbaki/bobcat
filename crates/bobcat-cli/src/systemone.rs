//! Jev/SystemOne decision requests for Cloudflare's Clef models.
//!
//! A request holds a state and a schema of questions. Each question is a yes-or-no proposition
//! (`noul`), a choice among named options (`choice`), or a score over ordered levels (`score`).
//! The encoding matches `encode_record` in the release's `joint_schema_model.py` token for token,
//! because the head reads the hidden states at the spans of each question and option.

use std::ops::Range;

use bobcat::clef::{Question, QuestionKind};
use serde_json::{Map, Value, json};
use tokenizers::Tokenizer;

use crate::Error;
use crate::decisions;

/// The longest request Clef reads, in tokens. Longer states lose their tail.
pub(crate) const MAX_TOKENS: usize = 16384;

/// The system prompt of every Clef request.
const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. \
                             Each answer must be exactly one of that field's allowed options.";

/// A request encoded for the model.
pub(crate) struct Encoded {
    /// The tokens of the whole request.
    pub(crate) tokens: Vec<u32>,
    /// Each question's kind and the spans of its instructions and options in `tokens`.
    pub(crate) questions: Vec<Question>,
    /// Each question's id and its options' ids, in the order of `questions`.
    ids: Vec<(String, Vec<String>)>,
}

/// Check that `request` is a decision request, with the errors the release's `systemone` gives.
pub(crate) fn validate(request: &Value) -> Result<(), String> {
    if !request["model"].is_string() || request.get("state").is_none() {
        return Err("model and state are required".to_owned());
    }
    let questions = request["questions"]
        .as_object()
        .filter(|questions| !questions.is_empty())
        .ok_or("at least one question is required")?;
    for (id, question) in questions {
        match question["type"].as_str() {
            Some("noul") => {}
            Some("choice" | "score") if !empty(&question["criteria"]) => {}
            Some("choice" | "score") => return Err(format!("{id}: criteria must not be empty")),
            _ => return Err(format!("{id}: type must be noul, choice, or score")),
        }
    }
    Ok(())
}

/// Return whether `value` is null, false, or an empty string, list, or object, as Python's `not`
/// sees it.
fn empty(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => true,
        Value::String(text) => text.is_empty(),
        Value::Array(items) => items.is_empty(),
        Value::Object(items) => items.is_empty(),
        Value::Bool(true) => false,
        Value::Number(number) => number.as_f64() == Some(0.0),
    }
}

/// Return the text of `value` in the request: a string as itself, any other value as compact JSON
/// with sorted keys.
fn render(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::Array(_) | Value::Object(_) => {
            sorted(value).to_string()
        }
    }
}

/// Return `value` with the keys of every object in sorted order.
fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut keys: Vec<&String> = object.keys().collect();
            keys.sort();
            Value::Object(
                keys.into_iter()
                    .map(|key| (key.clone(), sorted(&object[key])))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => value.clone(),
    }
}

/// Return the id and description of each option of `question`, in the order the model reads
/// them.
fn options(question: &Value) -> Vec<(String, Value)> {
    let criteria = &question["criteria"];
    match question["type"].as_str() {
        Some("noul") => [
            ("true", "The proposition is true or the answer is yes."),
            ("false", "The proposition is false or the answer is no."),
        ]
        .into_iter()
        .map(|(id, default)| {
            let description = criteria.get(id).cloned().unwrap_or_else(|| json!(default));
            (id.to_owned(), description)
        })
        .collect(),
        Some("choice") => {
            let mut options: Vec<(String, Value)> = criteria
                .as_object()
                .into_iter()
                .flatten()
                .map(|(id, description)| (id.clone(), description.clone()))
                .collect();
            options.sort_by(|a, b| a.0.cmp(&b.0));
            options
        }
        _ => criteria
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
            .map(|(index, description)| (index.to_string(), description.clone()))
            .collect(),
    }
}

/// Encode `request`, which [`validate`] accepted, in at most `max_tokens` tokens.
pub(crate) fn encode(
    tokenizer: &Tokenizer,
    request: &Value,
    max_tokens: usize,
) -> Result<Encoded, Error> {
    let tokens = |text: &str| -> Result<Vec<u32>, Error> {
        Ok(tokenizer.encode(text, false)?.get_ids().to_vec())
    };
    let mut schema = tokens("\n\nSCHEMA FIELDS:\n")?;
    let mut questions = Vec::new();
    let mut ids = Vec::new();
    let empty_schema = Map::new();
    let schema_questions = request["questions"].as_object().unwrap_or(&empty_schema);
    for (index, (id, question)) in schema_questions.iter().enumerate() {
        let kind_name = question["type"].as_str().unwrap_or_default();
        let kind = match kind_name {
            "noul" => QuestionKind::Noul,
            "choice" => QuestionKind::Choice,
            _ => QuestionKind::Score,
        };
        let number = index + 1;
        schema.extend(tokens(&format!(
            "\nFIELD {number}\nID: {id}\nTYPE: {kind_name}\nINSTRUCTION: "
        ))?);
        let start = schema.len();
        let instructions = match &question["instructions"] {
            Value::Null => id.clone(),
            Value::String(text) if text.is_empty() => id.clone(),
            other => render(other),
        };
        schema.extend(tokens(&instructions)?);
        let span = start..schema.len();
        schema.extend(tokens("\nALLOWED OPTIONS:\n")?);

        let mut option_spans = Vec::new();
        let mut option_ids = Vec::new();
        for (option_index, (option_id, description)) in options(question).into_iter().enumerate() {
            let number = option_index + 1;
            schema.extend(tokens(&format!("OPTION {number}: "))?);
            let start = schema.len();
            let mut semantics = Map::new();
            semantics.insert("option_id".to_owned(), json!(option_id));
            if !description.is_null() {
                semantics.insert("description".to_owned(), description);
            }
            schema.extend(tokens(&render(&Value::Object(semantics)))?);
            option_spans.push(start..schema.len());
            option_ids.push(option_id);
            schema.extend(tokens("\n")?);
        }
        schema.extend(tokens("END FIELD\n")?);
        questions.push(Question {
            kind,
            span,
            options: option_spans,
        });
        ids.push((id.clone(), option_ids));
    }

    let prefix = tokens(&format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
    ))?;
    let suffix = tokens(
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )?;
    let fixed = prefix.len() + schema.len() + suffix.len();
    if fixed > max_tokens {
        return Err(format!(
            "the schema needs {fixed} tokens before the state, and the limit is {max_tokens}"
        )
        .into());
    }
    let mut state = tokens(&render(&request["state"]))?;
    state.truncate(max_tokens - fixed);

    // The spans count from the start of the schema, which follows the prefix and the state.
    let offset = prefix.len() + state.len();
    let shift = |span: &Range<usize>| span.start + offset..span.end + offset;
    for question in &mut questions {
        question.span = shift(&question.span);
        question.options = question.options.iter().map(shift).collect();
    }
    let mut all = prefix;
    all.extend(state);
    all.extend(schema);
    all.extend(suffix);
    Ok(Encoded {
        tokens: all,
        questions,
        ids,
    })
}

/// Return the SystemOne response to `request` from the head's `logits` for `encoded`.
pub(crate) fn response(request: &Value, encoded: &Encoded, logits: &[Vec<f32>]) -> Value {
    let mut answers = Map::new();
    for ((id, option_ids), logits) in encoded.ids.iter().zip(logits) {
        let probabilities: Map<String, Value> = option_ids
            .iter()
            .cloned()
            .zip(decisions::softmax(
                &logits
                    .iter()
                    .map(|&logit| f64::from(logit))
                    .collect::<Vec<_>>(),
                1.0,
            ))
            .map(|(option, probability)| (option, json!(probability)))
            .collect();
        let probability = |option: &str| probabilities.get(option).and_then(Value::as_f64);
        let question = &request["questions"][id];
        let answer = match question["type"].as_str() {
            Some("noul") => {
                json!({"type": "noul", "noul": round(probability("true").unwrap_or(0.0))})
            }
            Some("choice") => {
                // The probabilities follow the order of the request's criteria.
                let ordered: Vec<(&String, f64)> = question["criteria"]
                    .as_object()
                    .into_iter()
                    .flatten()
                    .map(|(option, _)| (option, probability(option).unwrap_or(0.0)))
                    .collect();
                let (choice, confidence) = ordered
                    .iter()
                    .fold(
                        None,
                        |best: Option<(&String, f64)>, &(option, p)| match best {
                            Some((_, top)) if top >= p => best,
                            _ => Some((option, p)),
                        },
                    )
                    .unwrap_or((id, 0.0));
                let rounded: Map<String, Value> = ordered
                    .iter()
                    .map(|&(option, p)| (option.clone(), json!(round(p))))
                    .collect();
                json!({
                    "type": "choice",
                    "choice": choice,
                    "confidence": round(confidence),
                    "probabilities": rounded,
                })
            }
            _ => {
                let levels: Vec<(String, f64)> = option_ids
                    .iter()
                    .map(|level| (level.clone(), probability(level).unwrap_or(0.0)))
                    .collect();
                let score: f64 = (0_u32..)
                    .zip(&levels)
                    .map(|(index, (_, p))| f64::from(index) * p)
                    .sum();
                let confidence = levels.iter().map(|(_, p)| *p).fold(0.0, f64::max);
                let legend: Map<String, Value> = option_ids
                    .iter()
                    .cloned()
                    .zip(
                        question["criteria"]
                            .as_array()
                            .into_iter()
                            .flatten()
                            .cloned(),
                    )
                    .collect();
                let rounded: Map<String, Value> = levels
                    .iter()
                    .map(|(level, p)| (level.clone(), json!(round(*p))))
                    .collect();
                json!({
                    "type": "score",
                    "score": round(score),
                    "confidence": round(confidence),
                    "legend": legend,
                    "probabilities": rounded,
                })
            }
        };
        answers.insert(id.clone(), answer);
    }
    json!({
        "model": request["model"],
        "answers": answers,
        "usage": {"input_tokens": encoded.tokens.len(), "output_tokens": 0},
    })
}

/// Return `value` rounded to four decimal places, as the release's answers are.
fn round(value: f64) -> f64 {
    (value * 1e4).round() / 1e4
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use bobcat_gguf::Gguf;

    use super::*;

    /// Check that the reference request encodes to the release's tokens and spans.
    #[test]
    fn request_encodes_as_reference() -> Result<(), Error> {
        let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        let model_path = models.join("clef-flash-Q8_0.gguf");
        let dir = models.join("ref/clef-flash");
        if !model_path.exists() || !dir.join("tokens.i32").exists() {
            eprintln!(
                "skip: needs {} and a dump in {}",
                model_path.display(),
                dir.display()
            );
            return Ok(());
        }
        let gguf = Gguf::parse(std::fs::read(&model_path)?)?;
        let tokenizer = crate::tokenizer::from_gguf(&gguf)?;
        let request: Value = serde_json::from_slice(&std::fs::read(dir.join("request.json"))?)?;
        validate(&request)?;
        let encoded = encode(&tokenizer, &request, MAX_TOKENS)?;

        let want: Vec<u32> = std::fs::read(dir.join("tokens.i32"))?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&word| u32::from_le_bytes(word))
            .collect();
        assert_eq!(encoded.tokens, want);
        let reference: Value = serde_json::from_slice(&std::fs::read(dir.join("encoded.json"))?)?;
        let span = |value: &Value| -> Result<Range<usize>, Error> {
            let bound = |index: usize| -> Result<usize, Error> {
                let bound = value[index].as_u64().ok_or("a span needs two bounds")?;
                Ok(usize::try_from(bound)?)
            };
            Ok(bound(0)?..bound(1)?)
        };
        let reference = reference.as_array().ok_or("encoded.json holds a list")?;
        assert_eq!(encoded.questions.len(), reference.len());
        for ((question, (id, option_ids)), want) in
            encoded.questions.iter().zip(&encoded.ids).zip(reference)
        {
            assert_eq!(Some(id.as_str()), want["id"].as_str());
            assert_eq!(question.span, span(&want["span"])?);
            let want_spans = want["option_spans"]
                .as_array()
                .ok_or("a question needs option spans")?
                .iter()
                .map(span)
                .collect::<Result<Vec<_>, _>>()?;
            assert_eq!(question.options, want_spans);
            let want_ids: Vec<Option<&str>> = want["option_ids"]
                .as_array()
                .ok_or("a question needs option ids")?
                .iter()
                .map(Value::as_str)
                .collect();
            let got_ids: Vec<Option<&str>> =
                option_ids.iter().map(|id| Some(id.as_str())).collect();
            assert_eq!(got_ids, want_ids);
        }
        Ok(())
    }
}
