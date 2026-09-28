//! The scalar reference implementation of every op.
//!
//! The code favors plain loops and double-precision sums, so it defines the correct output for
//! faster kernels to match. Every faster kernel is tested against these functions.

use gip_gguf::TensorType;
use half::{bf16, f16};

/// Block sizes in bytes and elements, which must match [`TensorType::block`].
const Q4_0_BYTES: usize = 18;
const Q8_0_BYTES: usize = 34;
const Q4K_BYTES: usize = 144;
const Q6K_BYTES: usize = 210;
const BLOCK_ELEMENTS: usize = 32;
const SUPER_BLOCK_ELEMENTS: usize = 256;

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
    let (elements, bytes) = data_type.block();
    n_cols / elements * bytes
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

/// Return the half-precision number at byte `at` of `bytes` as a float.
fn f16_at(bytes: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

/// Dequantize the elements of `data_type` in `bytes` into `out`, which holds as many elements as
/// `bytes` encodes. The arithmetic follows ggml's reference dequantization in
/// `ggml/src/ggml-quants.c`.
pub fn dequantize(data_type: TensorType, bytes: &[u8], out: &mut [f32]) {
    match data_type {
        TensorType::F32 => {
            for (out, value) in out.iter_mut().zip(f32s(bytes)) {
                *out = value;
            }
        }
        TensorType::F16 => {
            for (out, bits) in out.iter_mut().zip(u16s(bytes)) {
                *out = f16_to_f32(bits);
            }
        }
        TensorType::Bf16 => {
            for (out, bits) in out.iter_mut().zip(u16s(bytes)) {
                *out = bf16_to_f32(bits);
            }
        }
        TensorType::Q4_0 => blocks(bytes, out, dequantize_q4_0),
        TensorType::Q8_0 => blocks(bytes, out, dequantize_q8_0),
        TensorType::Q4K => blocks(bytes, out, dequantize_q4k),
        TensorType::Q6K => blocks(bytes, out, dequantize_q6k),
    }
}

/// Dequantize each block of `B` bytes in `bytes` with `block` into the next `E` floats of `out`.
fn blocks<const B: usize, const E: usize>(
    bytes: &[u8],
    out: &mut [f32],
    block: fn(&[u8; B], &mut [f32; E]),
) {
    let blocks = bytes.as_chunks::<B>().0.iter();
    for (bytes, out) in blocks.zip(out.as_chunks_mut::<E>().0) {
        block(bytes, out);
    }
}

/// Dequantize a Q4_0 block: an fp16 scale, then 16 bytes whose low nibbles hold elements 0 to 15
/// and whose high nibbles hold elements 16 to 31, each offset by 8.
fn dequantize_q4_0(block: &[u8; Q4_0_BYTES], out: &mut [f32; BLOCK_ELEMENTS]) {
    let d = f16_at(block, 0);
    let (low, high) = out.split_at_mut(BLOCK_ELEMENTS / 2);
    for ((&q, low), high) in block[2..].iter().zip(low).zip(high) {
        *low = f32::from(i16::from(q & 0xf) - 8) * d;
        *high = f32::from(i16::from(q >> 4) - 8) * d;
    }
}

/// Dequantize a Q8_0 block: an fp16 scale, then 32 int8 quants.
fn dequantize_q8_0(block: &[u8; Q8_0_BYTES], out: &mut [f32; BLOCK_ELEMENTS]) {
    let d = f16_at(block, 0);
    for (&q, out) in block[2..].iter().zip(out) {
        *out = f32::from(q.cast_signed()) * d;
    }
}

/// Return the 6-bit scale and minimum of block `j` of a Q4_K super-block's packed `scales`.
fn q4k_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0xf) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// Dequantize a Q4_K super-block: fp16 `d` and `dmin`, 12 bytes of packed 6-bit scales and
/// minimums, then 128 bytes of 4-bit quants. Each 32 bytes of quants hold two 32-element blocks,
/// the first in the low nibbles.
fn dequantize_q4k(block: &[u8; Q4K_BYTES], out: &mut [f32; SUPER_BLOCK_ELEMENTS]) {
    let d = f16_at(block, 0);
    let dmin = f16_at(block, 2);
    let scales = &block[4..16];
    let quants = &block[16..];
    for (pair, out) in out
        .as_chunks_mut::<{ 2 * BLOCK_ELEMENTS }>()
        .0
        .iter_mut()
        .enumerate()
    {
        let (scale_low, min_low) = q4k_scale_min(2 * pair, scales);
        let (scale_high, min_high) = q4k_scale_min(2 * pair + 1, scales);
        let (d_low, m_low) = (d * f32::from(scale_low), dmin * f32::from(min_low));
        let (d_high, m_high) = (d * f32::from(scale_high), dmin * f32::from(min_high));
        let (low, high) = out.split_at_mut(BLOCK_ELEMENTS);
        let quants = &quants[pair * BLOCK_ELEMENTS..(pair + 1) * BLOCK_ELEMENTS];
        for ((&q, low), high) in quants.iter().zip(low).zip(high) {
            *low = d_low * f32::from(q & 0xf) - m_low;
            *high = d_high * f32::from(q >> 4) - m_high;
        }
    }
}

/// Dequantize a Q6_K super-block: 128 bytes of low 4-bit halves, 64 bytes of high 2-bit pairs,
/// 16 int8 scales of 16-element blocks, then an fp16 `d`. Each half of the super-block spreads
/// 128 elements over 64 low bytes and 32 high bytes.
fn dequantize_q6k(block: &[u8; Q6K_BYTES], out: &mut [f32; SUPER_BLOCK_ELEMENTS]) {
    let d = f16_at(block, 208);
    for (half, out) in out.as_chunks_mut::<128>().0.iter_mut().enumerate() {
        let low = &block[half * 64..half * 64 + 64];
        let high = &block[128 + half * 32..128 + half * 32 + 32];
        let scales = &block[192 + half * 8..192 + half * 8 + 8];
        for l in 0..32 {
            let is = l / 16;
            let quant = |low: u8, shift: u8| {
                let high = (high[l] >> shift) & 3;
                f32::from(i16::from(low | (high << 4)) - 32)
            };
            let scale = |index: usize| d * f32::from(scales[index].cast_signed());
            out[l] = scale(is) * quant(low[l] & 0xf, 0);
            out[l + 32] = scale(is + 2) * quant(low[l + 32] & 0xf, 2);
            out[l + 64] = scale(is + 4) * quant(low[l] >> 4, 4);
            out[l + 96] = scale(is + 6) * quant(low[l + 32] >> 4, 6);
        }
    }
}

/// Dequantize row `row` of the matrix in `weights`, whose rows hold `out.len()` elements of
/// `data_type`, into `out`.
pub fn get_row(data_type: TensorType, weights: &[u8], row: usize, out: &mut [f32]) {
    let bytes = row_bytes(data_type, out.len());
    dequantize(data_type, &weights[row * bytes..(row + 1) * bytes], out);
}

/// Multiply the matrix in `weights`, which has `y.len()` rows of `x.len()` elements of
/// `data_type`, by `x`, and store the results in `y`.
pub fn matvec(data_type: TensorType, weights: &[u8], x: &[f32], y: &mut [f32]) {
    let bytes = row_bytes(data_type, x.len());
    let mut row = vec![0.0; x.len()];
    for (y, row_bytes) in y.iter_mut().zip(weights.chunks_exact(bytes)) {
        dequantize(data_type, row_bytes, &mut row);
        *y = dot(&row, x);
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
