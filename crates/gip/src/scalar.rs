//! The scalar reference implementation of every op.
//!
//! The code favors plain loops and double-precision sums, so it defines the correct output for
//! faster kernels to match. Every faster kernel is tested against these functions.

use gip_gguf::TensorType;
use half::{bf16, f16};

/// Must match [`gip_gguf::Q8_0_BLOCK_ELEMENTS`] and [`gip_gguf::Q8_0_BLOCK_BYTES`].
const Q8_0_ELEMENTS: usize = 32;
const Q8_0_BYTES: usize = 34;

/// Return the float value of the IEEE half-precision bits `bits`.
pub fn f16_to_f32(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

/// Return the float value of the bfloat16 bits `bits`.
pub fn bf16_to_f32(bits: u16) -> f32 {
    bf16::from_bits(bits).to_f32()
}

/// Return the bytes of one row of `n_cols` elements of `data_type`.
pub fn row_bytes(data_type: TensorType, n_cols: usize) -> usize {
    match data_type {
        TensorType::F32 => n_cols * 4,
        TensorType::F16 | TensorType::Bf16 => n_cols * 2,
        TensorType::Q8_0 => n_cols / Q8_0_ELEMENTS * Q8_0_BYTES,
    }
}

/// Return the little-endian 16-bit numbers in `bytes`.
fn u16s(bytes: &[u8]) -> impl Iterator<Item = u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&pair| u16::from_le_bytes(pair))
}

/// Return the little-endian floats in `bytes`.
pub fn f32s(bytes: &[u8]) -> impl Iterator<Item = f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&quad| f32::from_le_bytes(quad))
}

/// Return the scale of a Q8_0 block and its quants.
fn q8_0_block(block: &[u8; Q8_0_BYTES]) -> (f32, &[u8]) {
    let (scale, quants) = block.split_at(2);
    (f16_to_f32(u16::from_le_bytes([scale[0], scale[1]])), quants)
}

/// Return the dot product of the elements of `data_type` in `row` with the floats in `x`.
fn dot_row(data_type: TensorType, row: &[u8], x: &[f32]) -> f64 {
    match data_type {
        TensorType::F32 => f32s(row)
            .zip(x)
            .map(|(w, &x)| f64::from(w) * f64::from(x))
            .sum(),
        TensorType::F16 => u16s(row)
            .zip(x)
            .map(|(w, &x)| f64::from(f16_to_f32(w)) * f64::from(x))
            .sum(),
        TensorType::Bf16 => u16s(row)
            .zip(x)
            .map(|(w, &x)| f64::from(bf16_to_f32(w)) * f64::from(x))
            .sum(),
        TensorType::Q8_0 => row
            .as_chunks::<Q8_0_BYTES>()
            .0
            .iter()
            .zip(x.as_chunks::<Q8_0_ELEMENTS>().0)
            .map(|(block, xs)| {
                let (scale, quants) = q8_0_block(block);
                let sum: f64 = quants
                    .iter()
                    .zip(xs)
                    .map(|(&q, &x)| f64::from(q.cast_signed()) * f64::from(x))
                    .sum();
                f64::from(scale) * sum
            })
            .sum(),
    }
}

/// Dequantize row `row` of the matrix in `weights`, whose rows hold `out.len()` elements of
/// `data_type`, into `out`.
pub fn get_row(data_type: TensorType, weights: &[u8], row: usize, out: &mut [f32]) {
    let n_cols = out.len();
    let bytes = row_bytes(data_type, n_cols);
    let source = &weights[row * bytes..(row + 1) * bytes];
    match data_type {
        TensorType::F32 => {
            for (out, value) in out.iter_mut().zip(f32s(source)) {
                *out = value;
            }
        }
        TensorType::F16 => {
            for (out, bits) in out.iter_mut().zip(u16s(source)) {
                *out = f16_to_f32(bits);
            }
        }
        TensorType::Bf16 => {
            for (out, bits) in out.iter_mut().zip(u16s(source)) {
                *out = bf16_to_f32(bits);
            }
        }
        TensorType::Q8_0 => {
            for (outs, block) in out
                .as_chunks_mut::<Q8_0_ELEMENTS>()
                .0
                .iter_mut()
                .zip(source.as_chunks::<Q8_0_BYTES>().0)
            {
                let (scale, quants) = q8_0_block(block);
                for (out, &q) in outs.iter_mut().zip(quants) {
                    *out = scale * f32::from(q.cast_signed());
                }
            }
        }
    }
}

/// Multiply the matrix in `weights`, which has `y.len()` rows of `x.len()` elements of
/// `data_type`, by `x`, and store the results in `y`.
pub fn matvec(data_type: TensorType, weights: &[u8], x: &[f32], y: &mut [f32]) {
    let bytes = row_bytes(data_type, x.len());
    for (y, row) in y.iter_mut().zip(weights.chunks_exact(bytes)) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the double sum rounds to the float result"
        )]
        let value = dot_row(data_type, row, x) as f32;
        *y = value;
    }
}

/// Normalize `x` by its root mean square, scale it by `weight`, and store the result in `out`.
/// `eps` guards the division.
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let scale = rms_scale(x, eps);
    for ((out, &x), &w) in out.iter_mut().zip(x).zip(weight) {
        *out = w * (x * scale);
    }
}

/// Normalize `x` in place by its root mean square and scale it by `weight`.
pub fn rms_norm_in_place(x: &mut [f32], weight: &[f32], eps: f32) {
    let scale = rms_scale(x, eps);
    for (x, &w) in x.iter_mut().zip(weight) {
        *x = w * (*x * scale);
    }
}

/// Return the reciprocal root mean square of `x`, guarded by `eps`.
fn rms_scale(x: &[f32], eps: f32) -> f32 {
    let sum_squares: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let mean = sum_squares / x.len() as f64;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the double scale rounds to the float factor"
    )]
    let scale = (1.0 / (mean + f64::from(eps)).sqrt()) as f32;
    scale
}

/// Rotate the head `vec` for position `pos` with base `theta`. Element `i` pairs with element
/// `i + vec.len() / 2`, as in GPT-NeoX.
pub fn rope_neox(vec: &mut [f32], pos: u32, theta: f32) {
    let head_dim = vec.len();
    let (first, second) = vec.split_at_mut(head_dim / 2);
    for (i, (x0, x1)) in first.iter_mut().zip(second).enumerate() {
        // transformers computes the frequencies and angles in float32. Matching its precision
        // keeps the comparison tight.
        let inv_freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
        let angle = pos as f32 * inv_freq;
        let (sin, cos) = angle.sin_cos();
        let (a, b) = (*x0, *x1);
        *x0 = a * cos - b * sin;
        *x1 = b * cos + a * sin;
    }
}

/// Return `x` times the logistic sigmoid of `x`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Replace `x` with its softmax.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f64;
    for value in x.iter_mut() {
        *value = (*value - max).exp();
        sum += f64::from(*value);
    }
    for value in x.iter_mut() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the double quotient rounds to the float probability"
        )]
        let probability = (f64::from(*value) / sum) as f32;
        *value = probability;
    }
}

/// Return the dot product of `a` and `b`.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let sum: f64 = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the double sum rounds to the float result"
    )]
    let result = sum as f32;
    result
}
