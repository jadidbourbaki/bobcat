//! Checks the Metal matrix-vector kernels of every weight format against the scalar reference on
//! random matrices, including row counts that leave a partial threadgroup.
//!
//! Each shape runs as a plain multiply, with the RMS norm fused in, accumulating into its output,
//! and as the fused SwiGLU pair.

#![cfg(target_os = "macos")]

use std::error::Error;

use bobcat::TensorType;
use bobcat::metal::{Format, MatvecOptions, Metal, Norm};
use bobcat::scalar;
use half::f16;

/// The largest error allowed, relative to the largest magnitude in the scalar result. The GPU
/// sums in float32 and the scalar code sums in double, and rows of a few thousand elements stay
/// far below this bound.
const TOLERANCE: f64 = 1e-5;

/// The epsilon of the fused RMS norm.
const NORM_EPS: f32 = 1e-5;

/// The shapes cover the LFM2.5 projections, row counts that fill no threadgroup, and one that
/// leaves a partial threadgroup. A format skips the shapes whose rows fill no whole block.
const SHAPES: [(u32, u32); 9] = [
    (1, 32),
    (7, 64),
    (7, 256),
    (1000, 1024),
    (1024, 1024),
    (3072, 1024),
    (4608, 1024),
    (1024, 4608),
    (65536, 1024),
];

/// Each format with its tensor type and the byte offsets of the fp16 scales in each block.
const FORMATS: [(Format, TensorType, &[usize]); 5] = [
    (Format::Q8_0, TensorType::Q8_0, &[0]),
    (Format::Q4_0, TensorType::Q4_0, &[0]),
    (Format::Q4K, TensorType::Q4K, &[0, 2]),
    (Format::Q5K, TensorType::Q5K, &[0, 2]),
    (Format::Q6K, TensorType::Q6K, &[208]),
];

/// The kinds of launch the test checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Plain,
    Norm,
    Accumulate,
    Swiglu,
}

/// A xorshift64 generator with a fixed seed, so every run sees the same matrices.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Return a float in [-1, 1).
    fn unit(&mut self) -> f32 {
        // 24 random bits convert to f32 exactly.
        (self.next() >> 40) as f32 / (1 << 23) as f32 - 1.0
    }

    /// Return the half-precision bits of a scale in [2^-10, 2^-6), a range typical of quantized
    /// weights.
    fn scale_bits(&mut self) -> u16 {
        let exponent = 5 + (self.next() % 4) as u16;
        let mantissa = (self.next() & 0x3ff) as u16;
        (exponent << 10) | mantissa
    }

    /// Return `n_bytes` bytes of random blocks of `block_bytes` bytes each, with a random fp16
    /// scale at each offset in `scales`.
    fn blocks(&mut self, n_bytes: usize, block_bytes: usize, scales: &[usize]) -> Vec<u8> {
        let mut bytes: Vec<u8> = (0..n_bytes).map(|_| (self.next() & 0xff) as u8).collect();
        for block in bytes.chunks_exact_mut(block_bytes) {
            for &at in scales {
                block[at..at + 2].copy_from_slice(&self.scale_bits().to_le_bytes());
            }
        }
        bytes
    }
}

/// Return the largest difference between `got` and `want`. A NaN counts as an infinite difference.
fn max_difference(got: &[f32], want: &[f32]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(&got, &want)| (f64::from(got) - f64::from(want)).abs())
        .fold(0.0, |max, diff| {
            if diff.is_nan() {
                f64::INFINITY
            } else {
                max.max(diff)
            }
        })
}

/// Return the largest magnitude in `values`.
fn max_magnitude(values: &[f32]) -> f64 {
    values
        .iter()
        .fold(0.0, |max, &value| max.max(f64::from(value).abs()))
}

/// Return the largest sum over one row of the magnitudes of the terms `weight * input` that the
/// matrix of `data_type` in `weights` multiplies `input` by.
fn term_scale(data_type: TensorType, weights: &[u8], input: &[f32]) -> f64 {
    let mut row = vec![0.0; input.len()];
    weights
        .chunks_exact(scalar::row_bytes(data_type, input.len()))
        .map(|bytes| {
            scalar::dequantize(data_type, bytes, &mut row);
            row.iter()
                .zip(input)
                .map(|(&w, &x)| f64::from(w * x).abs())
                .sum::<f64>()
        })
        .fold(0.0, f64::max)
}

/// Run a launch of `mode` on random `n_rows` by `n_cols` matrices of `format` and return its error
/// against the scalar reference.
fn launch_error(
    metal: &mut Metal,
    random: &mut Random,
    (format, data_type, scales): (Format, TensorType, &[usize]),
    mode: Mode,
    n_rows: u32,
    n_cols: u32,
) -> Result<f64, Box<dyn Error>> {
    let rows = n_rows as usize;
    let cols = n_cols as usize;
    let block_bytes = data_type.block().1;
    let weight_bytes = scalar::row_bytes(data_type, cols) * rows;
    let weights = |random: &mut Random| {
        if format == Format::F16 {
            let mut bytes = Vec::with_capacity(weight_bytes);
            for _ in 0..weight_bytes / 2 {
                bytes.extend_from_slice(&f16::from_f32(random.unit()).to_bits().to_le_bytes());
            }
            bytes
        } else {
            random.blocks(weight_bytes, block_bytes, scales)
        }
    };
    let gate_weights = weights(random);
    let up_weights = weights(random);
    let x: Vec<f32> = (0..cols).map(|_| random.unit()).collect();
    let norm_weight: Vec<f32> = (0..cols).map(|_| 1.0 + 0.5 * random.unit()).collect();
    let prior: Vec<f32> = (0..rows).map(|_| random.unit()).collect();

    // The scalar reference: normalize when the mode fuses the norm, then multiply, then combine
    // with the prior output or the up matrix.
    let mut input = x.clone();
    if matches!(mode, Mode::Norm | Mode::Swiglu) {
        scalar::rms_norm(&x, &norm_weight, NORM_EPS, &mut input);
    }
    let mut want = vec![0.0; rows];
    scalar::matvec(data_type, &gate_weights, &input, &mut want);
    match mode {
        Mode::Plain | Mode::Norm => {}
        Mode::Accumulate => {
            for (want, &prior) in want.iter_mut().zip(&prior) {
                *want += prior;
            }
        }
        Mode::Swiglu => {
            let mut up = vec![0.0; rows];
            scalar::matvec(data_type, &up_weights, &input, &mut up);
            for (want, &up) in want.iter_mut().zip(&up) {
                *want = scalar::silu(*want) * up;
            }
        }
    }

    let gate = metal.new_buffer(weight_bytes)?;
    let up = metal.new_buffer(weight_bytes)?;
    let x_buffer = metal.new_buffer(cols * 4)?;
    let norm_buffer = metal.new_buffer(cols * 4)?;
    let y = metal.new_buffer(rows * 4)?;
    metal.write(gate.at(0), &gate_weights)?;
    metal.write(up.at(0), &up_weights)?;
    metal.write(x_buffer.at(0), &x)?;
    metal.write(norm_buffer.at(0), &norm_weight)?;
    metal.write(y.at(0), &prior)?;

    let norm = Norm {
        weight: norm_buffer.at(0),
        eps: NORM_EPS,
    };
    metal.begin()?;
    if mode == Mode::Swiglu {
        metal.matvec_swiglu(
            format,
            gate.at(0),
            up.at(0),
            n_rows,
            n_cols,
            x_buffer.at(0),
            norm,
            y.at(0),
        )?;
    } else {
        let options = MatvecOptions {
            norm: (mode == Mode::Norm).then_some(norm),
            accumulate: mode == Mode::Accumulate,
        };
        metal.matvec(
            format,
            gate.at(0),
            n_rows,
            n_cols,
            x_buffer.at(0),
            y.at(0),
            options,
        )?;
    }
    metal.end()?;

    let mut got = vec![0.0; rows];
    metal.read(y.at(0), &mut got)?;
    // A dot product's rounding error scales with the terms it sums, which can far exceed a result
    // that cancels to near zero. SwiGLU outputs scale with their own magnitude.
    let scale = match mode {
        Mode::Plain | Mode::Norm | Mode::Accumulate => {
            term_scale(data_type, &gate_weights, &input).max(max_magnitude(&want))
        }
        Mode::Swiglu => max_magnitude(&want),
    };
    Ok(max_difference(&got, &want) / scale)
}

#[test]
fn f16_matvec_matches_scalar() -> Result<(), Box<dyn Error>> {
    let mut metal = match Metal::open() {
        Ok(metal) => metal,
        Err(error) => {
            eprintln!("skip: {error}");
            return Ok(());
        }
    };
    let mut random = Random(0x60e7_927b_081d_a5ef);
    let format = (Format::F16, TensorType::F16, &[][..]);
    for mode in [Mode::Plain, Mode::Norm, Mode::Accumulate, Mode::Swiglu] {
        for (rows, cols) in [(7, 256), (1000, 1024)] {
            let error = launch_error(&mut metal, &mut random, format, mode, rows, cols)?;
            assert!(error < TOLERANCE, "F16 {mode:?} {rows}x{cols}: {error:e}");
        }
    }
    Ok(())
}

#[test]
fn matvec_kernels_match_scalar() -> Result<(), Box<dyn Error>> {
    let mut metal = match Metal::open() {
        Ok(metal) => metal,
        Err(error) => {
            eprintln!("skip: {error}");
            return Ok(());
        }
    };
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    let mut failures = Vec::new();
    for format in FORMATS {
        let block = format.1.block().0;
        for mode in [Mode::Plain, Mode::Norm, Mode::Accumulate, Mode::Swiglu] {
            for (n_rows, n_cols) in SHAPES {
                if !(n_cols as usize).is_multiple_of(block) {
                    continue;
                }
                // max_difference reports NaN as infinity, so a plain comparison catches it.
                let error = launch_error(&mut metal, &mut random, format, mode, n_rows, n_cols)?;
                if error > TOLERANCE {
                    failures.push(format!(
                        "{:?} {mode:?} {n_rows}x{n_cols}: {error:.3e}",
                        format.0
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "errors above {TOLERANCE:e}: {failures:?}"
    );
    Ok(())
}
