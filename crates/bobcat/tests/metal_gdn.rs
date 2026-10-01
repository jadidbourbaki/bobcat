//! Checks the Gated DeltaNet kernels and the gates of Qwen3.5's attention against
//! double-precision references on random inputs.

#![cfg(target_os = "macos")]
#![expect(clippy::print_stderr, reason = "tests report skips on stderr")]

use std::error::Error;

use bobcat::metal::{DeltaShape, Metal};

type TestResult = Result<(), Box<dyn Error>>;

/// The largest error allowed, relative to the largest magnitude of the reference. The kernels
/// sum in float and the references in double.
const TOLERANCE: f64 = 1e-5;

/// A xorshift64 generator with a fixed seed, so every run sees the same inputs.
struct Random(u64);

impl Random {
    /// Return a float spread evenly over -1 to 1.
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let fraction = (self.0 >> 40) as f32 / (1_u64 << 24) as f32;
        2.0 * fraction - 1.0
    }

    fn floats(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.unit()).collect()
    }
}

/// Return the Metal backend, or `None` after printing why the machine has none.
fn open() -> Option<Metal> {
    match Metal::open() {
        Ok(metal) => Some(metal),
        Err(error) => {
            eprintln!("skip: {error}");
            None
        }
    }
}

/// Fail when `got` differs from `want` by more than `TOLERANCE` of the largest magnitude of
/// `want`.
fn check(name: &str, got: &[f32], want: &[f64]) -> TestResult {
    assert_eq!(got.len(), want.len(), "{name} length");
    let scale = want.iter().fold(1e-12_f64, |m, &v| m.max(v.abs()));
    let worst = got
        .iter()
        .zip(want)
        .map(|(&g, &w)| (f64::from(g) - w).abs())
        .fold(0.0, f64::max);
    if worst / scale > TOLERANCE {
        return Err(format!("{name} differs by {worst:e} against a scale of {scale:e}").into());
    }
    Ok(())
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// Run consecutive batches through the convolution, so the history carries across calls. The
/// batches of one and two tokens take `gdn_conv`, and the longer ones take `gdn_conv_batch` with
/// `gdn_conv_history`.
#[test]
fn convolution_matches_reference() -> TestResult {
    let Some(mut metal) = open() else {
        return Ok(());
    };
    let (channels, kernel_size) = (40_usize, 4_usize);
    let past = kernel_size - 1;
    let mut random = Random(0x2545_f491_4f6c_dd1d);
    let taps = random.floats(channels * kernel_size);
    let start = random.floats(channels * past);
    let batches = [1_usize, 2, 7, 1, 64, 3];
    let n_total: usize = batches.iter().sum();
    let inputs = random.floats(n_total * channels);

    let taps_buffer = metal.new_buffer(taps.len() * 4)?;
    let history = metal.new_buffer(start.len() * 4)?;
    let x = metal.new_buffer(64 * channels * 4)?;
    let y = metal.new_buffer(64 * channels * 4)?;
    metal.write(taps_buffer.at(0), &taps)?;
    metal.write(history.at(0), &start)?;

    // The reference keeps every input with the starting history in front, channel by channel.
    let mut seen: Vec<Vec<f64>> = (0..channels)
        .map(|c| {
            start[c * past..(c + 1) * past]
                .iter()
                .map(|&v| f64::from(v))
                .collect()
        })
        .collect();
    let mut offset = 0;
    for n in batches {
        let batch = &inputs[offset * channels..(offset + n) * channels];
        offset += n;
        metal.write(x.at(0), batch)?;
        metal.begin()?;
        metal.gdn_conv(
            x.at(0),
            taps_buffer.at(0),
            history.at(0),
            y.at(0),
            u32::try_from(channels)?,
            u32::try_from(kernel_size)?,
            u32::try_from(n)?,
        )?;
        metal.end()?;
        let mut got = vec![0.0_f32; n * channels];
        metal.read(y.at(0), &mut got)?;
        let mut want = vec![0.0_f64; n * channels];
        for t in 0..n {
            for c in 0..channels {
                seen[c].push(f64::from(batch[t * channels + c]));
                let window = &seen[c][seen[c].len() - kernel_size..];
                let sum: f64 = window
                    .iter()
                    .zip(&taps[c * kernel_size..(c + 1) * kernel_size])
                    .map(|(&input, &tap)| input * f64::from(tap))
                    .sum();
                want[t * channels + c] = silu(sum);
            }
        }
        check(&format!("convolution of {n} tokens"), &got, &want)?;
    }
    let mut got = vec![0.0_f32; channels * past];
    metal.read(history.at(0), &mut got)?;
    let want: Vec<f64> = seen
        .iter()
        .flat_map(|inputs| inputs[inputs.len() - past..].to_vec())
        .collect();
    check("history", &got, &want)
}

/// Run a decode step, a batch, and another decode step through the recurrence. A decode step
/// normalizes the queries and keys inside `gdn_recurrence`, and a batch normalizes them with
/// `gdn_qk_norm` first.
#[test]
fn recurrence_matches_reference() -> TestResult {
    let Some(mut metal) = open() else {
        return Ok(());
    };
    let shape = DeltaShape {
        n_k_heads: 2,
        n_v_heads: 4,
        k_dim: 128,
        v_dim: 64,
    };
    let (n_k, n_v) = (shape.n_k_heads as usize, shape.n_v_heads as usize);
    let (k_dim, v_dim) = (shape.k_dim as usize, shape.v_dim as usize);
    let key_dim = n_k * k_dim;
    let conv_dim = shape.conv_dim() as usize;
    let q_scale = 1.0 / (k_dim as f32).sqrt();
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    let start: Vec<f32> = random
        .floats(n_v * k_dim * v_dim)
        .iter()
        .map(|v| 0.1 * v)
        .collect();
    let mut state: Vec<f64> = start.iter().map(|&v| f64::from(v)).collect();

    let batch_max = 5;
    let y = metal.new_buffer(batch_max * conv_dim * 4)?;
    let beta = metal.new_buffer(batch_max * n_v * 4)?;
    let decay = metal.new_buffer(batch_max * n_v * 4)?;
    let state_buffer = metal.new_buffer(start.len() * 4)?;
    let out = metal.new_buffer(batch_max * n_v * v_dim * 4)?;
    metal.write(state_buffer.at(0), &start)?;

    for n in [1_usize, batch_max, 1] {
        let tokens = random.floats(n * conv_dim);
        let betas: Vec<f32> = random
            .floats(n * n_v)
            .iter()
            .map(|v| 0.5 + 0.5 * v)
            .collect();
        let decays: Vec<f32> = random
            .floats(n * n_v)
            .iter()
            .map(|v| 0.9 + 0.1 * v)
            .collect();
        metal.write(y.at(0), &tokens)?;
        metal.write(beta.at(0), &betas)?;
        metal.write(decay.at(0), &decays)?;
        metal.begin()?;
        metal.gdn_recurrence(
            y.at(0),
            beta.at(0),
            decay.at(0),
            state_buffer.at(0),
            out.at(0),
            shape,
            u32::try_from(n)?,
            q_scale,
        )?;
        metal.end()?;
        let mut got = vec![0.0_f32; n * n_v * v_dim];
        metal.read(out.at(0), &mut got)?;

        let mut want = vec![0.0_f64; n * n_v * v_dim];
        for t in 0..n {
            let token = &tokens[t * conv_dim..(t + 1) * conv_dim];
            for h in 0..n_v {
                let key_head = h % n_k;
                let unit = |values: &[f32], scale: f64| -> Vec<f64> {
                    let norm = values.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
                    let factor = scale / (norm + 1e-6).sqrt();
                    values.iter().map(|&v| f64::from(v) * factor).collect()
                };
                let q = unit(
                    &token[key_head * k_dim..(key_head + 1) * k_dim],
                    f64::from(q_scale),
                );
                let k = unit(
                    &token[key_dim + key_head * k_dim..key_dim + (key_head + 1) * k_dim],
                    1.0,
                );
                let values = &token[2 * key_dim + h * v_dim..2 * key_dim + (h + 1) * v_dim];
                let s = &mut state[h * k_dim * v_dim..(h + 1) * k_dim * v_dim];
                let gate = f64::from(decays[t * n_v + h]);
                let strength = f64::from(betas[t * n_v + h]);
                for j in 0..v_dim {
                    let mut remembered = 0.0;
                    for i in 0..k_dim {
                        s[i * v_dim + j] *= gate;
                        remembered += s[i * v_dim + j] * k[i];
                    }
                    let delta = (f64::from(values[j]) - remembered) * strength;
                    let mut sum = 0.0;
                    for i in 0..k_dim {
                        s[i * v_dim + j] += k[i] * delta;
                        sum += s[i * v_dim + j] * q[i];
                    }
                    want[(t * n_v + h) * v_dim + j] = sum;
                }
            }
        }
        check(&format!("recurrence of {n} tokens"), &got, &want)?;
    }
    let mut got = vec![0.0_f32; state.len()];
    metal.read(state_buffer.at(0), &mut got)?;
    check("state", &got, &state)
}

/// Check the elementwise launches that finish a DeltaNet layer and an attention layer: the
/// gates, the gated output norm, SiLU times a gate, and the attention gate.
#[test]
fn gates_and_norms_match_reference() -> TestResult {
    let Some(mut metal) = open() else {
        return Ok(());
    };
    let mut random = Random(0x1234_5678_9abc_def1);
    let (n_tokens, n_v_heads) = (3_usize, 4_usize);
    let n = n_tokens * n_v_heads;

    let betas = random.floats(n);
    let alphas: Vec<f32> = random.floats(n).iter().map(|v| 25.0 * v).collect();
    let a: Vec<f32> = random
        .floats(n_v_heads)
        .iter()
        .map(|v| -0.5 - 0.5 * v.abs())
        .collect();
    let dt_bias = random.floats(n_v_heads);
    let beta = metal.new_buffer(n * 4)?;
    let alpha = metal.new_buffer(n * 4)?;
    let a_buffer = metal.new_buffer(n_v_heads * 4)?;
    let dt_buffer = metal.new_buffer(n_v_heads * 4)?;
    metal.write(beta.at(0), &betas)?;
    metal.write(alpha.at(0), &alphas)?;
    metal.write(a_buffer.at(0), &a)?;
    metal.write(dt_buffer.at(0), &dt_bias)?;
    metal.begin()?;
    metal.gdn_gates(
        beta.at(0),
        alpha.at(0),
        a_buffer.at(0),
        dt_buffer.at(0),
        u32::try_from(n_v_heads)?,
        u32::try_from(n_tokens)?,
    )?;
    metal.end()?;
    let mut got = vec![0.0_f32; n];
    metal.read(beta.at(0), &mut got)?;
    let want: Vec<f64> = betas
        .iter()
        .map(|&b| 1.0 / (1.0 + (-f64::from(b)).exp()))
        .collect();
    check("update strengths", &got, &want)?;
    metal.read(alpha.at(0), &mut got)?;
    let want: Vec<f64> = alphas
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let head = i % n_v_heads;
            let rate = f64::from(x) + f64::from(dt_bias[head]);
            // transformers' softplus returns its input above 20.
            let softplus = if rate > 20.0 {
                rate
            } else {
                rate.exp().ln_1p()
            };
            (f64::from(a[head]) * softplus).exp()
        })
        .collect();
    check("decays", &got, &want)?;

    let v_dim = 64_usize;
    let rows = n_tokens * n_v_heads;
    let eps = 1e-6_f32;
    let x = random.floats(rows * v_dim);
    let gate = random.floats(rows * v_dim);
    let weight = random.floats(v_dim);
    let x_buffer = metal.new_buffer(x.len() * 4)?;
    let gate_buffer = metal.new_buffer(gate.len() * 4)?;
    let weight_buffer = metal.new_buffer(weight.len() * 4)?;
    metal.write(x_buffer.at(0), &x)?;
    metal.write(gate_buffer.at(0), &gate)?;
    metal.write(weight_buffer.at(0), &weight)?;
    metal.begin()?;
    metal.gdn_gated_norm(
        x_buffer.at(0),
        weight_buffer.at(0),
        gate_buffer.at(0),
        u32::try_from(v_dim)?,
        u32::try_from(rows)?,
        eps,
    )?;
    metal.end()?;
    let mut got = vec![0.0_f32; x.len()];
    metal.read(x_buffer.at(0), &mut got)?;
    let mut want = vec![0.0_f64; x.len()];
    for r in 0..rows {
        let row = &x[r * v_dim..(r + 1) * v_dim];
        let mean = row.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / v_dim as f64;
        let scale = 1.0 / (mean + f64::from(eps)).sqrt();
        for d in 0..v_dim {
            want[r * v_dim + d] = f64::from(row[d])
                * scale
                * f64::from(weight[d])
                * silu(f64::from(gate[r * v_dim + d]));
        }
    }
    check("gated norm", &got, &want)?;

    metal.write(x_buffer.at(0), &x)?;
    metal.begin()?;
    metal.silu_mul(x_buffer.at(0), gate_buffer.at(0), u32::try_from(x.len())?)?;
    metal.end()?;
    metal.read(x_buffer.at(0), &mut got)?;
    let want: Vec<f64> = x
        .iter()
        .zip(&gate)
        .map(|(&v, &g)| f64::from(v) * silu(f64::from(g)))
        .collect();
    check("SiLU times the gate", &got, &want)?;

    // Each head's queries come before its gates in the query projection.
    let (head_dim, n_heads) = (32_usize, 3_usize);
    let q_dim = head_dim * n_heads;
    let attended = random.floats(n_tokens * q_dim);
    let projection = random.floats(n_tokens * 2 * q_dim);
    let out = metal.new_buffer(attended.len() * 4)?;
    let qg = metal.new_buffer(projection.len() * 4)?;
    metal.write(out.at(0), &attended)?;
    metal.write(qg.at(0), &projection)?;
    metal.begin()?;
    metal.attention_gate(
        out.at(0),
        qg.at(0),
        u32::try_from(head_dim)?,
        u32::try_from(q_dim)?,
        u32::try_from(n_tokens)?,
    )?;
    metal.end()?;
    let mut got = vec![0.0_f32; attended.len()];
    metal.read(out.at(0), &mut got)?;
    let want: Vec<f64> = (0..attended.len())
        .map(|i| {
            let (token, within) = (i / q_dim, i % q_dim);
            let (head, d) = (within / head_dim, within % head_dim);
            let g = f64::from(projection[token * 2 * q_dim + head * 2 * head_dim + head_dim + d]);
            f64::from(attended[i]) / (1.0 + (-g).exp())
        })
        .collect();
    check("attention gate", &got, &want)
}
