//! Compares Clef decisions against Cloudflare's reference code.
//!
//! `tools/clef_dump.py --release ../models/hf/clef-flash --out ../models/ref/clef-flash` writes the
//! dump, and `tools/clef_gguf.py` writes `models/clef-flash-Q8_0.gguf`. The dump holds the hidden
//! states of the release's bfloat16 backbone and the logits of its head in float32.

#![expect(clippy::print_stderr, reason = "tests report skips on stderr")]

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use bobcat::clef::{Head, Question, QuestionKind};
use bobcat::qwen35::Model;
use serde_json::Value;

type TestResult = Result<(), Box<dyn Error>>;

/// The largest logit error of the head given the reference hidden states, relative to the
/// largest logit. The reference reads each option's output embeddings from the release's
/// bfloat16 weights and bobcat reads them from the Q8_0 file, which moved the logits by up to
/// 1.5e-4 of the largest.
const HEAD_TOLERANCE: f64 = 1e-3;

/// The largest difference allowed between a probability from bobcat's Q8_0 backbone on the GPU and
/// one from the release's bfloat16 backbone.
#[cfg(target_os = "macos")]
const PROBABILITY_TOLERANCE: f64 = 0.05;

#[test]
fn head_matches_reference() -> TestResult {
    let Some((model, reference)) = load()? else {
        return Ok(());
    };
    let head = Head::load(&model)?.ok_or("the model file holds no Clef head")?;
    let hidden = read_f32s(&reference.dir, "hidden.f32")?;
    let logits = head.logits(&model, &hidden, &reference.tokens, &reference.questions)?;
    for (got, want) in logits.iter().zip(&reference.logits) {
        let scale = want.iter().fold(1.0_f64, |m, &v| m.max(f64::from(v).abs()));
        for (&got, &want) in got.iter().zip(want) {
            let error = (f64::from(got) - f64::from(want)).abs() / scale;
            if error > HEAD_TOLERANCE {
                return Err(format!(
                    "logits {logits:?} differ from the reference {:?}",
                    reference.logits
                )
                .into());
            }
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn decision_matches_reference() -> TestResult {
    let Some((model, reference)) = load()? else {
        return Ok(());
    };
    let mut metal = match bobcat::metal::Metal::open() {
        Ok(metal) => metal,
        Err(error) => {
            eprintln!("skip: {error}");
            return Ok(());
        }
    };
    let head = Head::load(&model)?.ok_or("the model file holds no Clef head")?;
    let n_tokens = reference.tokens.len();
    let n_embd = model.hyperparameters().n_embd as usize;
    let mut gpu =
        bobcat::qwen35::Qwen35Metal::new(&model, &mut metal, u32::try_from(n_tokens)?, true)?;
    let mut hidden = vec![0.0; n_tokens * n_embd];
    gpu.prefill(&reference.tokens, None, Some(&mut hidden), None)?;
    let logits = head.logits(&model, &hidden, &reference.tokens, &reference.questions)?;
    for (got, want) in logits.iter().zip(&reference.logits) {
        let got = softmax(got);
        let want = softmax(want);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0, f64::max);
        if worst > PROBABILITY_TOLERANCE || argmax(&got) != argmax(&want) {
            return Err(format!("probabilities {got:?} differ from the reference {want:?}").into());
        }
    }
    Ok(())
}

/// The encoded request and the reference logits of one dump.
struct Reference {
    dir: PathBuf,
    tokens: Vec<u32>,
    questions: Vec<Question>,
    logits: Vec<Vec<f32>>,
}

/// Load `models/clef-flash-Q8_0.gguf` and the dump in `models/ref/clef-flash`, or return `None`
/// when either is missing.
fn load() -> Result<Option<(Model, Reference)>, Box<dyn Error>> {
    let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let model_path = models.join("clef-flash-Q8_0.gguf");
    let dir = models.join("ref/clef-flash");
    if !model_path.exists() || !dir.join("tokens.i32").exists() {
        eprintln!(
            "skip: needs {} from tools/clef_gguf.py and a dump in {} from tools/clef_dump.py",
            model_path.display(),
            dir.display()
        );
        return Ok(None);
    }
    let tokens = fs::read(dir.join("tokens.i32"))?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&word| u32::from_le_bytes(word))
        .collect();
    let encoded: Value = serde_json::from_slice(&fs::read(dir.join("encoded.json"))?)?;
    let span = |value: &Value| -> Result<std::ops::Range<usize>, Box<dyn Error>> {
        let start = value[0].as_u64().ok_or("a span needs a start")?;
        let end = value[1].as_u64().ok_or("a span needs an end")?;
        Ok(usize::try_from(start)?..usize::try_from(end)?)
    };
    let mut questions = Vec::new();
    for question in encoded.as_array().ok_or("encoded.json holds a list")? {
        let kind = match question["type"].as_u64() {
            Some(0) => QuestionKind::Noul,
            Some(1) => QuestionKind::Choice,
            Some(2) => QuestionKind::Score,
            _ => return Err("a question has an unknown type".into()),
        };
        let options = question["option_spans"]
            .as_array()
            .ok_or("a question needs option spans")?
            .iter()
            .map(span)
            .collect::<Result<_, _>>()?;
        questions.push(Question {
            kind,
            span: span(&question["span"])?,
            options,
        });
    }
    let logits: Vec<Vec<f32>> = serde_json::from_slice(&fs::read(dir.join("logits.json"))?)?;
    let reference = Reference {
        dir,
        tokens,
        questions,
        logits,
    };
    Ok(Some((Model::load(model_path)?, reference)))
}

/// Return the floats in the file `name` of `dir`.
fn read_f32s(dir: &Path, name: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    Ok(fs::read(dir.join(name))?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&word| f32::from_le_bytes(word))
        .collect())
}

/// Return the softmax of `logits` in double precision.
#[cfg(target_os = "macos")]
fn softmax(logits: &[f32]) -> Vec<f64> {
    let best = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = logits
        .iter()
        .map(|&logit| f64::from(logit - best).exp())
        .collect();
    let total: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / total).collect()
}

/// Return the index of the largest of `x`.
#[cfg(target_os = "macos")]
fn argmax(x: &[f64]) -> usize {
    let mut best = 0;
    for (i, &value) in x.iter().enumerate() {
        if value > x[best] {
            best = i;
        }
    }
    best
}
