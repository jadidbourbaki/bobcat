//! Compares the Qwen3.5 forward passes against activations dumped from transformers.
//!
//! `tools/ref_dump.py --model ../models/hf/Qwen3.5-0.8B --gguf ../models/Qwen3.5-0.8B-Q8_0.gguf
//! --out ../models/ref/Qwen3.5-0.8B-Q8_0` writes the dump. transformers runs in float32 on the
//! GGUF file's dequantized weights, so the scalar pass matches it to rounding.

#![expect(clippy::print_stderr, reason = "tests report skips on stderr")]

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use bobcat::Trace;
use bobcat::qwen35::{Model, State};

type TestResult = Result<(), Box<dyn Error>>;

/// The largest error a float32 pass may show against float32 transformers.
const TOLERANCE: f64 = 1e-4;

/// The largest error allowed when batched matmuls multiply half-precision weight tiles.
#[cfg(target_os = "macos")]
const HALF_TOLERANCE: f64 = 5e-3;

#[test]
fn q8_0_scalar() -> TestResult {
    let Some((model, reference)) = load("Qwen3.5-0.8B-Q8_0")? else {
        return Ok(());
    };
    let mut state = State::new(&model, reference.n_ctx()?)?;
    let mut pass = |token: u32, logits: &mut [f32], trace: Option<&mut Trace>| {
        model.step(&mut state, token, Some(logits), trace)
    };
    compare_steps(&model, &reference, &mut pass, TOLERANCE)
}

#[cfg(target_os = "macos")]
#[test]
fn q8_0_metal() -> TestResult {
    let Some((model, reference)) = load("Qwen3.5-0.8B-Q8_0")? else {
        return Ok(());
    };
    let Some(mut metal) = open_metal() else {
        return Ok(());
    };
    let mut gpu = bobcat::qwen35::Qwen35Metal::new(&model, &mut metal, reference.n_ctx()?, false)?;
    let mut pass = |token: u32, logits: &mut [f32], trace: Option<&mut Trace>| {
        gpu.step(token, Some(logits), trace)
    };
    compare_steps(&model, &reference, &mut pass, TOLERANCE)
}

#[cfg(target_os = "macos")]
#[test]
fn q8_0_metal_batch() -> TestResult {
    check_batch("Qwen3.5-0.8B-Q8_0", "Qwen3.5-0.8B-Q8_0")
}

/// The long prompt of 946 tokens spans two prefill batches.
#[cfg(target_os = "macos")]
#[test]
fn q8_0_metal_long_batch() -> TestResult {
    check_batch("Qwen3.5-0.8B-Q8_0", "Qwen3.5-0.8B-Q8_0-long")
}

/// Run the prompt of the dump `ref_name` through one Metal prefill of `models/NAME.gguf` with a
/// half-precision cache, compare every activation, then check greedy decoding.
#[cfg(target_os = "macos")]
fn check_batch(name: &str, ref_name: &str) -> TestResult {
    let Some((model, reference)) = load_reference(name, ref_name)? else {
        return Ok(());
    };
    let Some(mut metal) = open_metal() else {
        return Ok(());
    };
    let hp = model.hyperparameters();
    let n_embd = hp.n_embd as usize;
    let n_vocab = hp.n_vocab as usize;
    let n_tokens = reference.tokens.len();
    let mut gpu = bobcat::qwen35::Qwen35Metal::new(&model, &mut metal, reference.n_ctx()?, true)?;
    let mut trace = model.trace(u32::try_from(n_tokens)?);
    let mut logits = vec![0.0; n_vocab];
    let mut final_norm = vec![0.0; n_tokens * n_embd];
    gpu.prefill(
        &reference.tokens,
        Some(&mut logits),
        Some(&mut final_norm),
        Some(&mut trace),
    )?;

    let mut worst = Worst::new(&model);
    for t in 0..n_tokens {
        let rows = |values: &[f32], layer: usize| {
            values[(layer * n_tokens + t) * n_embd..][..n_embd].to_vec()
        };
        let mut errors = vec![relative_error(
            &rows(&trace.embedding, 0),
            &reference.row("embedding", t, n_embd)?,
        )];
        for il in 0..hp.n_layers as usize {
            errors.push(relative_error(
                &rows(&trace.layers, il),
                &reference.row(&format!("layer_{il:02}"), t, n_embd)?,
            ));
        }
        let want_norm = reference.row("final_norm", t, n_embd)?;
        errors.push(relative_error(&rows(&trace.final_norm, 0), &want_norm));
        // The final norm the caller asks for matches the traced one.
        errors.push(relative_error(&rows(&final_norm, 0), &want_norm));
        worst.add(&errors);
    }
    worst.add_logits(relative_error(
        &logits,
        &reference.row("logits", n_tokens - 1, n_vocab)?,
    ));
    worst.check(HALF_TOLERANCE)?;

    // Greedy decoding continues from the prompt's logits on the GPU.
    for (g, &want) in reference.generated.iter().enumerate() {
        let got = argmax(&logits);
        if got != want {
            return Err(format!("generated token {g} is {got}, transformers chose {want}").into());
        }
        gpu.step(got, Some(&mut logits), None)?;
    }
    Ok(())
}

/// Run the prompt of `reference` through `pass` one token at a time, compare every activation
/// within `tolerance`, then check greedy decoding.
fn compare_steps(
    model: &Model,
    reference: &Reference,
    pass: &mut impl FnMut(u32, &mut [f32], Option<&mut Trace>) -> Result<(), bobcat::Error>,
    tolerance: f64,
) -> TestResult {
    let hp = model.hyperparameters();
    let n_embd = hp.n_embd as usize;
    let n_vocab = hp.n_vocab as usize;
    let mut logits = vec![0.0; n_vocab];
    let mut worst = Worst::new(model);
    for (t, &token) in reference.tokens.iter().enumerate() {
        let mut trace = model.trace(1);
        pass(token, &mut logits, Some(&mut trace))?;
        let row = |name: &str| reference.row(name, t, n_embd);
        let mut errors = vec![relative_error(&trace.embedding, &row("embedding")?)];
        for il in 0..hp.n_layers as usize {
            let got = &trace.layers[il * n_embd..(il + 1) * n_embd];
            errors.push(relative_error(got, &row(&format!("layer_{il:02}"))?));
        }
        errors.push(relative_error(&trace.final_norm, &row("final_norm")?));
        worst.add(&errors);
        worst.add_logits(relative_error(
            &logits,
            &reference.row("logits", t, n_vocab)?,
        ));
    }
    worst.check(tolerance)?;

    // The prompt's last logits predict the first generated token, and each prediction feeds the
    // next step.
    for (g, &want) in reference.generated.iter().enumerate() {
        let got = argmax(&logits);
        if got != want {
            return Err(format!("generated token {g} is {got}, transformers chose {want}").into());
        }
        pass(got, &mut logits, None)?;
    }
    Ok(())
}

/// The worst error of each activation across the prompt's tokens.
struct Worst {
    labels: Vec<String>,
    errors: Vec<f64>,
    logits: f64,
}

impl Worst {
    fn new(model: &Model) -> Self {
        let layers = model.attention_layers().enumerate().map(|(il, attention)| {
            format!("layer {il:2} {}", if attention { "attn" } else { "delta" })
        });
        let labels: Vec<String> = std::iter::once("embedding".to_owned())
            .chain(layers)
            .chain(["final norm".to_owned(), "final out".to_owned()])
            .collect();
        Self {
            errors: vec![0.0; labels.len()],
            labels,
            logits: 0.0,
        }
    }

    /// Fold in one token's errors, in the order of the labels.
    fn add(&mut self, errors: &[f64]) {
        for (worst, &error) in self.errors.iter_mut().zip(errors) {
            *worst = worst.max(error);
        }
    }

    fn add_logits(&mut self, error: f64) {
        self.logits = self.logits.max(error);
    }

    /// Fail with a table of the errors when any exceeds `tolerance`.
    fn check(&self, tolerance: f64) -> TestResult {
        if self.logits <= tolerance && self.errors.iter().all(|&error| error <= tolerance) {
            return Ok(());
        }
        let table: Vec<String> = self
            .labels
            .iter()
            .zip(&self.errors)
            .map(|(label, error)| format!("{label:<14} {error:.3e}"))
            .chain([format!("{:<14} {:.3e}", "logits", self.logits)])
            .collect();
        Err(format!(
            "an activation exceeds the tolerance {tolerance:e}:\n{}",
            table.join("\n")
        )
        .into())
    }
}

/// The prompt and the activations of one dump.
struct Reference {
    dir: PathBuf,
    tokens: Vec<u32>,
    generated: Vec<u32>,
}

impl Reference {
    /// Return the context the prompt and the greedy continuation need.
    fn n_ctx(&self) -> Result<u32, Box<dyn Error>> {
        Ok(u32::try_from(self.tokens.len() + self.generated.len())?)
    }

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
    load_reference(name, name)
}

/// Load `models/NAME.gguf` and the dump in `models/ref/REF_NAME`, or return `None` when either is
/// missing.
fn load_reference(
    name: &str,
    ref_name: &str,
) -> Result<Option<(Model, Reference)>, Box<dyn Error>> {
    let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let model_path = models.join(format!("{name}.gguf"));
    let dir = models.join("ref").join(ref_name);
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

/// Open the Metal backend, or return `None` when the machine has none.
#[cfg(target_os = "macos")]
fn open_metal() -> Option<bobcat::metal::Metal> {
    match bobcat::metal::Metal::open() {
        Ok(metal) => Some(metal),
        Err(error) => {
            eprintln!("skip: {error}");
            None
        }
    }
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
