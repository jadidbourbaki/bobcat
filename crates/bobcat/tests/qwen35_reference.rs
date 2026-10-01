//! Compares the Qwen3.5 forward pass against activations dumped from transformers.
//!
//! `tools/ref_dump.py --model ../models/hf/Qwen3.5-0.8B --gguf ../models/Qwen3.5-0.8B-Q8_0.gguf
//! --out ../models/ref/Qwen3.5-0.8B-Q8_0` writes the dump. transformers runs in float32 on the
//! GGUF file's dequantized weights, so the scalar pass matches it to rounding.

#![expect(clippy::print_stderr, reason = "tests report skips on stderr")]

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use bobcat::qwen35::{Model, State};

type TestResult = Result<(), Box<dyn Error>>;

/// The largest error the float32 scalar pass may show against float32 transformers.
const TOLERANCE: f64 = 1e-4;

#[test]
fn q8_0_scalar() -> TestResult {
    let Some((model, reference)) = load("Qwen3.5-0.8B-Q8_0")? else {
        return Ok(());
    };
    let hp = model.hyperparameters();
    let n_embd = hp.n_embd as usize;
    let n_vocab = hp.n_vocab as usize;
    let n_layers = hp.n_layers as usize;
    let n_ctx = u32::try_from(reference.tokens.len() + reference.generated.len())?;
    let mut state = State::new(&model, n_ctx)?;
    let mut logits = vec![0.0; n_vocab];

    // The worst error of the embedding, each layer, the final norm, and the logits.
    let mut worst = vec![0.0_f64; n_layers + 3];
    for (t, &token) in reference.tokens.iter().enumerate() {
        let mut trace = model.trace(1);
        model.step(&mut state, token, Some(&mut logits), Some(&mut trace))?;
        let row = |name: &str| reference.row(name, t, n_embd);
        let mut errors = vec![relative_error(&trace.embedding, &row("embedding")?)];
        for il in 0..n_layers {
            let got = &trace.layers[il * n_embd..(il + 1) * n_embd];
            errors.push(relative_error(got, &row(&format!("layer_{il:02}"))?));
        }
        errors.push(relative_error(&trace.final_norm, &row("final_norm")?));
        errors.push(relative_error(
            &logits,
            &reference.row("logits", t, n_vocab)?,
        ));
        for (worst, error) in worst.iter_mut().zip(errors) {
            *worst = worst.max(error);
        }
    }
    if worst.iter().any(|&error| error > TOLERANCE) {
        let kinds: Vec<&str> = model
            .attention_layers()
            .map(|attention| if attention { "attn" } else { "delta" })
            .collect();
        let labels = std::iter::once("embedding".to_owned())
            .chain(
                kinds
                    .iter()
                    .enumerate()
                    .map(|(il, kind)| format!("layer {il:2} {kind}")),
            )
            .chain(["final norm".to_owned(), "logits".to_owned()]);
        let table: Vec<String> = labels
            .zip(&worst)
            .map(|(label, error)| format!("{label:<14} {error:.3e}"))
            .collect();
        return Err(format!(
            "an activation exceeds the tolerance {TOLERANCE:e}:\n{}",
            table.join("\n")
        )
        .into());
    }

    // The prompt's last logits predict the first generated token, and each prediction feeds the
    // next step.
    for (g, &want) in reference.generated.iter().enumerate() {
        let got = argmax(&logits);
        if got != want {
            return Err(format!("generated token {g} is {got}, transformers chose {want}").into());
        }
        model.step(&mut state, got, Some(&mut logits), None)?;
    }
    Ok(())
}

/// The prompt and the activations of one dump.
struct Reference {
    dir: PathBuf,
    tokens: Vec<u32>,
    generated: Vec<u32>,
}

impl Reference {
    /// Return row `t` of `width` floats of the file `name`.
    fn row(&self, name: &str, t: usize, width: usize) -> Result<Vec<f32>, Box<dyn Error>> {
        let bytes = fs::read(self.dir.join(format!("{name}.f32")))?;
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&word| f32::from_le_bytes(word))
            .collect();
        values
            .get(t * width..(t + 1) * width)
            .map(<[f32]>::to_vec)
            .ok_or_else(|| format!("{name} holds no row {t}").into())
    }
}

/// Load `models/NAME.gguf` and the dump in `models/ref/NAME`, or return `None` when either is
/// missing.
fn load(name: &str) -> Result<Option<(Model, Reference)>, Box<dyn Error>> {
    let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let model_path = models.join(format!("{name}.gguf"));
    let dir = models.join("ref").join(name);
    if !model_path.exists() || !dir.join("tokens.i32").exists() {
        eprintln!(
            "skip: needs {} and a dump in {} from tools/ref_dump.py",
            model_path.display(),
            dir.display()
        );
        return Ok(None);
    }
    let words = |file: &str| -> Result<Vec<u32>, Box<dyn Error>> {
        let bytes = fs::read(dir.join(file))?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&word| u32::from_le_bytes(word))
            .collect())
    };
    let reference = Reference {
        tokens: words("tokens.i32")?,
        generated: words("generated.i32")?,
        dir,
    };
    Ok(Some((Model::load(model_path)?, reference)))
}

/// Return the largest absolute difference of `got` and `want` relative to the largest magnitude
/// of `want`. A NaN counts as infinite.
fn relative_error(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "compared rows differ in length");
    let mut max_diff = 0.0_f64;
    let mut max_want = 0.0_f64;
    for (&got, &want) in got.iter().zip(want) {
        let diff = (f64::from(got) - f64::from(want)).abs();
        max_diff = if diff.is_nan() {
            f64::INFINITY
        } else {
            max_diff.max(diff)
        };
        max_want = max_want.max(f64::from(want).abs());
    }
    if max_want > 0.0 {
        max_diff / max_want
    } else {
        max_diff
    }
}

/// Return the index of the largest of `x`. Ties go to the lowest index.
fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &value) in x.iter().enumerate() {
        if value > x[best] {
            best = i;
        }
    }
    u32::try_from(best).expect("the vocabulary fits in a u32")
}
