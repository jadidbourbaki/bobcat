//! Checks the Metal Q8_0 matrix-vector kernels against the scalar reference on random matrices,
//! including row counts that leave a partial threadgroup.
//!
//! Each shape runs as a plain multiply, with the RMS norm fused in, accumulating into its output,
//! and as the fused SwiGLU pair.

#![cfg(target_os = "macos")]

use std::error::Error;

use gip::TensorType;
use gip::metal::{MatvecOptions, Metal, Norm};
use gip::scalar;

/// The largest error allowed, relative to the largest magnitude in the scalar result. The GPU
/// sums in float32 and the scalar code sums in double, and rows of a few thousand elements stay
/// far below this bound.
const TOLERANCE: f64 = 1e-5;

/// The epsilon of the fused RMS norm.
const NORM_EPS: f32 = 1e-5;

/// The shapes cover the LFM2.5 projections, a row count that fills no threadgroup, and one that
/// leaves a partial threadgroup.
const SHAPES: [(u32, u32); 8] = [
    (1, 32),
    (7, 64),
    (1000, 1024),
    (1024, 1024),
    (3072, 1024),
    (4608, 1024),
    (1024, 4608),
    (65536, 1024),
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

    /// Return the half-precision bits of a scale in [2^-10, 2^-6), a range typical of Q8_0
    /// weights.
    fn scale_bits(&mut self) -> u16 {
        let exponent = 5 + (self.next() % 4) as u16;
        let mantissa = (self.next() & 0x3ff) as u16;
        (exponent << 10) | mantissa
    }

    /// Return `n_bytes` bytes of random Q8_0 blocks.
    fn q8_0(&mut self, n_bytes: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(n_bytes);
        for _ in 0..n_bytes / 34 {
            bytes.extend_from_slice(&self.scale_bits().to_le_bytes());
            bytes.extend((0..32).map(|_| (self.next() & 0xff) as u8));
        }
        bytes
    }
}

/// Return the largest difference between `got` and `want`, divided by the largest magnitude in
/// `want`.
fn relative_error(got: &[f32], want: &[f32]) -> f64 {
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

/// Run a launch of `mode` on random `n_rows` by `n_cols` matrices and return its error against
/// the scalar reference.
fn launch_error(
    metal: &mut Metal,
    random: &mut Random,
    mode: Mode,
    n_rows: u32,
    n_cols: u32,
) -> Result<f64, Box<dyn Error>> {
    let rows = n_rows as usize;
    let cols = n_cols as usize;
    let weight_bytes = scalar::row_bytes(TensorType::Q8_0, cols) * rows;
    let gate_weights = random.q8_0(weight_bytes);
    let up_weights = random.q8_0(weight_bytes);
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
    scalar::matvec(TensorType::Q8_0, &gate_weights, &input, &mut want);
    match mode {
        Mode::Plain | Mode::Norm => {}
        Mode::Accumulate => {
            for (want, &prior) in want.iter_mut().zip(&prior) {
                *want += prior;
            }
        }
        Mode::Swiglu => {
            let mut up = vec![0.0; rows];
            scalar::matvec(TensorType::Q8_0, &up_weights, &input, &mut up);
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
        metal.matvec_q8_0_swiglu(
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
        metal.matvec_q8_0(gate.at(0), n_rows, n_cols, x_buffer.at(0), y.at(0), options)?;
    }
    metal.end()?;

    let mut got = vec![0.0; rows];
    metal.read(y.at(0), &mut got)?;
    Ok(relative_error(&got, &want))
}

#[test]
fn q8_0_kernels_match_scalar() -> Result<(), Box<dyn Error>> {
    let mut metal = match Metal::open() {
        Ok(metal) => metal,
        Err(error) => {
            eprintln!("skip: {error}");
            return Ok(());
        }
    };
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    let mut failures = Vec::new();
    for mode in [Mode::Plain, Mode::Norm, Mode::Accumulate, Mode::Swiglu] {
        for (n_rows, n_cols) in SHAPES {
            // relative_error reports NaN as infinity, so a plain comparison catches it.
            let error = launch_error(&mut metal, &mut random, mode, n_rows, n_cols)?;
            if error > TOLERANCE {
                failures.push(format!("{mode:?} {n_rows}x{n_cols}: {error:.3e}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "errors above {TOLERANCE:e}: {failures:?}"
    );
    Ok(())
}
