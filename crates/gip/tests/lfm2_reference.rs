//! Checks gip's LFM2 forward pass against activations that `tools/ref_dump.py` saved from
//! transformers.
//!
//! Each test compares the embedding, every layer's output, the final norm, and the logits for
//! each prompt token, then checks that greedy decoding reproduces transformers' continuation token
//! for token. The BF16 file holds the checkpoint's own weights. The Q8_0 file's reference comes
//! from transformers loading the same GGUF, so both sides see identical dequantized weights.

#![expect(clippy::print_stderr, reason = "tests report skips on stderr")]

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use gip::{Hyperparameters, Model, State, Trace};

/// The largest error allowed, relative to the largest magnitude in the reference row. Both sides
/// compute in float32 from the same weights, so only summation order separates them.
const TOLERANCE: f64 = 1e-4;

/// The largest error allowed when the KV cache holds half precision, which rounds every cached
/// key and value to 11 significant bits. On LFM2.5-350M the largest error measured was 1.2e-3.
#[cfg(target_os = "macos")]
const HALF_KV_TOLERANCE: f64 = 5e-3;

type TestResult = Result<(), Box<dyn Error>>;

#[test]
fn bf16_scalar() -> TestResult {
    check_scalar("LFM2.5-350M-BF16.gguf", "LFM2.5-350M")
}

#[test]
fn q8_0_scalar() -> TestResult {
    check_scalar("LFM2.5-350M-Q8_0.gguf", "LFM2.5-350M-Q8_0")
}

#[cfg(target_os = "macos")]
#[test]
fn q8_0_metal() -> TestResult {
    check_metal(true, false)
}

#[cfg(target_os = "macos")]
#[test]
fn q8_0_metal_f32_kv() -> TestResult {
    check_metal(false, false)
}

#[cfg(target_os = "macos")]
#[test]
fn q8_0_metal_batch() -> TestResult {
    check_metal(true, true)
}

/// Compare the scalar pass on the model in `model_file` against the dump in `ref_name`.
fn check_scalar(model_file: &str, ref_name: &str) -> TestResult {
    let Some((model, reference)) = load(model_file, ref_name)? else {
        return Ok(());
    };
    let mut pass = ScalarPass {
        model: &model,
        state: State::new(&model, reference.n_ctx()?)?,
    };
    compare(&model, &reference, &mut pass, false, TOLERANCE)
}

/// Compare the Metal pass on LFM2.5-350M at Q8_0 against its dump. The KV cache holds half
/// precision when `kv_half` is true. The prompt runs in one prefill call when `batched` is true.
#[cfg(target_os = "macos")]
fn check_metal(kv_half: bool, batched: bool) -> TestResult {
    let Some((model, reference)) = load("LFM2.5-350M-Q8_0.gguf", "LFM2.5-350M-Q8_0")? else {
        return Ok(());
    };
    let mut metal = match gip::metal::Metal::open() {
        Ok(metal) => metal,
        Err(error) => {
            eprintln!("skip: {error}");
            return Ok(());
        }
    };
    let mut gpu = gip::Lfm2Metal::new(&model, &mut metal, reference.n_ctx()?, kv_half)?;
    let tolerance = if kv_half {
        HALF_KV_TOLERANCE
    } else {
        TOLERANCE
    };
    compare(&model, &reference, &mut gpu, batched, tolerance)
}

/// A forward pass under test.
trait Pass {
    fn step(&mut self, token: u32, logits: &mut [f32], trace: &mut Trace) -> TestResult;
    fn prefill(&mut self, tokens: &[u32], logits: &mut [f32], trace: &mut Trace) -> TestResult;
    /// Decode `out.len()` tokens greedily, starting from `logits`.
    fn generate(&mut self, logits: &mut [f32], out: &mut [u32]) -> TestResult;
}

/// The scalar pass on the CPU.
struct ScalarPass<'m> {
    model: &'m Model,
    state: State,
}

impl Pass for ScalarPass<'_> {
    fn step(&mut self, token: u32, logits: &mut [f32], trace: &mut Trace) -> TestResult {
        Ok(self
            .model
            .step(&mut self.state, token, Some(logits), Some(trace))?)
    }

    fn prefill(&mut self, _: &[u32], _: &mut [f32], _: &mut Trace) -> TestResult {
        Err("the scalar pass has no batched prefill".into())
    }

    fn generate(&mut self, logits: &mut [f32], out: &mut [u32]) -> TestResult {
        let n = out.len();
        for (g, out) in out.iter_mut().enumerate() {
            *out = argmax(logits);
            if g + 1 < n {
                self.model.step(&mut self.state, *out, Some(logits), None)?;
            }
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl Pass for gip::Lfm2Metal<'_> {
    fn step(&mut self, token: u32, logits: &mut [f32], trace: &mut Trace) -> TestResult {
        Ok(Self::step(self, token, Some(logits), Some(trace))?)
    }

    fn prefill(&mut self, tokens: &[u32], logits: &mut [f32], trace: &mut Trace) -> TestResult {
        Ok(Self::prefill(self, tokens, Some(logits), Some(trace))?)
    }

    // The GPU picks every token itself in one pipelined call.
    fn generate(&mut self, _: &mut [f32], out: &mut [u32]) -> TestResult {
        Ok(Self::generate(self, out)?)
    }
}

/// Return the repository's `models` directory.
fn models_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

/// Load the model in `model_file` and the dump in `ref_name`, or return `None` when either is
/// missing.
fn load(model_file: &str, ref_name: &str) -> Result<Option<(Model, Reference)>, Box<dyn Error>> {
    let model_path = models_dir().join(model_file);
    let ref_dir = models_dir().join("ref").join(ref_name);
    if !model_path.exists() || !ref_dir.join("tokens.i32").exists() {
        eprintln!(
            "skip: needs {} and a dump in {} from tools/ref_dump.py",
            model_path.display(),
            ref_dir.display()
        );
        return Ok(None);
    }
    let model = Model::load(&model_path)?;
    let reference = Reference::load(&ref_dir, model.hyperparameters())?;
    Ok(Some((model, reference)))
}

/// Return the little-endian 32-bit values in the file `name` of `dir`.
fn read_words(dir: &Path, name: &str) -> Result<Vec<[u8; 4]>, Box<dyn Error>> {
    let bytes = fs::read(dir.join(name))?;
    Ok(bytes.as_chunks::<4>().0.to_vec())
}

/// Return the floats in the file `name` of `dir`, which must hold `count` of them.
fn read_f32s(dir: &Path, name: &str, count: usize) -> Result<Vec<f32>, Box<dyn Error>> {
    let values: Vec<f32> = read_words(dir, name)?
        .into_iter()
        .map(f32::from_le_bytes)
        .collect();
    if values.len() != count {
        return Err(format!("{name} holds {} floats, expected {count}", values.len()).into());
    }
    Ok(values)
}

/// The reference activations of a prompt and its greedy continuation.
struct Reference {
    tokens: Vec<u32>,
    generated: Vec<u32>,
    embedding: Vec<f32>,
    layers: Vec<Vec<f32>>,
    final_norm: Vec<f32>,
    logits: Vec<f32>,
}

impl Reference {
    fn load(dir: &Path, hp: &Hyperparameters) -> Result<Self, Box<dyn Error>> {
        let tokens: Vec<u32> = read_words(dir, "tokens.i32")?
            .into_iter()
            .map(u32::from_le_bytes)
            .collect();
        let generated = read_words(dir, "generated.i32")?
            .into_iter()
            .map(u32::from_le_bytes)
            .collect();
        let rows = tokens.len() * hp.n_embd as usize;
        let layers = (0..hp.n_layers)
            .map(|il| read_f32s(dir, &format!("layer_{il:02}.f32"), rows))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            embedding: read_f32s(dir, "embedding.f32", rows)?,
            layers,
            final_norm: read_f32s(dir, "final_norm.f32", rows)?,
            logits: read_f32s(dir, "logits.f32", tokens.len() * hp.n_vocab as usize)?,
            tokens,
            generated,
        })
    }

    /// Return the context the prompt and its continuation need.
    fn n_ctx(&self) -> Result<u32, Box<dyn Error>> {
        Ok(u32::try_from(self.tokens.len() + self.generated.len())?)
    }
}

/// Return the largest absolute difference between `got` and `want`, divided by the largest
/// magnitude in `want`. A NaN in `got` counts as an infinite error.
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
    u32::try_from(best).expect("the vocabulary fits in u32")
}

/// Run `pass` on the prompt of `reference` and compare every activation within `tolerance`,
/// then check greedy decoding. The prompt runs in one prefill call when `batched` is true.
fn compare(
    model: &Model,
    reference: &Reference,
    pass: &mut impl Pass,
    batched: bool,
    tolerance: f64,
) -> TestResult {
    let hp = model.hyperparameters();
    let n_embd = hp.n_embd as usize;
    let n_vocab = hp.n_vocab as usize;
    let n_layers = hp.n_layers as usize;
    let n_tokens = reference.tokens.len();

    // WORST holds the largest error of each compared activation across the prompt: the
    // embedding, each layer, the final norm, and the logits.
    let mut worst = vec![0.0_f64; n_layers + 3];
    let mut logits = vec![0.0_f32; n_vocab];

    // The batched mode traces every token of one prefill call, which returns only the last
    // token's logits.
    let mut trace = Trace::new(hp, if batched { u32::try_from(n_tokens)? } else { 1 });
    if batched {
        pass.prefill(&reference.tokens, &mut logits, &mut trace)?;
    }

    for t in 0..n_tokens {
        let row = t * n_embd..(t + 1) * n_embd;
        // A traced step holds one token at row 0 of each activation. A traced prefill holds
        // every token, with each layer's rows after the previous layer's.
        let (got_row, layer_stride) = if batched {
            (row.clone(), n_tokens * n_embd)
        } else {
            pass.step(reference.tokens[t], &mut logits, &mut trace)?;
            (0..n_embd, n_embd)
        };
        let layer_row = |il: usize| il * layer_stride + got_row.start;

        let mut errors = vec![0.0_f64; n_layers + 3];
        errors[0] = relative_error(
            &trace.embedding[got_row.clone()],
            &reference.embedding[row.clone()],
        );
        for il in 0..n_layers {
            let start = layer_row(il);
            errors[1 + il] = relative_error(
                &trace.layers[start..start + n_embd],
                &reference.layers[il][row.clone()],
            );
        }
        errors[n_layers + 1] =
            relative_error(&trace.final_norm[got_row], &reference.final_norm[row]);
        if !batched || t + 1 == n_tokens {
            errors[n_layers + 2] =
                relative_error(&logits, &reference.logits[t * n_vocab..(t + 1) * n_vocab]);
        }
        for (worst, error) in worst.iter_mut().zip(errors) {
            *worst = worst.max(error);
        }
    }

    // relative_error reports NaN as infinity, so a plain comparison catches it.
    if worst.iter().any(|&error| error > tolerance) {
        let layer_labels = model.attention_layers().enumerate().map(|(il, attention)| {
            format!("layer {il:2} {}", if attention { "attn" } else { "conv" })
        });
        let table: Vec<String> = std::iter::once("embedding".to_owned())
            .chain(layer_labels)
            .chain(["final norm".to_owned(), "logits".to_owned()])
            .zip(&worst)
            .map(|(label, error)| format!("{label:<14} {error:.3e}"))
            .collect();
        let table = table.join("\n");
        return Err(format!("an activation exceeds the tolerance {tolerance:e}:\n{table}").into());
    }

    // The prompt's last logits predict the first generated token. Each prediction then feeds the
    // next step.
    let mut predicted = vec![0; reference.generated.len()];
    pass.generate(&mut logits, &mut predicted)?;
    for (g, (&got, &want)) in predicted.iter().zip(&reference.generated).enumerate() {
        if got != want {
            return Err(format!("generated token {g} is {got}, transformers chose {want}").into());
        }
    }
    Ok(())
}
