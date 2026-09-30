//! Measure one Metal matrix multiplication shape without the model graph.

#![expect(
    clippy::print_stdout,
    reason = "bobcat-matmul-bench reports measurements on stdout"
)]

use std::process::ExitCode;

use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Quant {
    F16,
    Q8_0,
    Q4_0,
    Q4k,
    Q6k,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum StoreMode {
    Overwrite,
    Accumulate,
    Swiglu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Matmul {
    Auto,
    Simd,
    Tensor,
}

/// Measure the GPU time of a repeated matrix multiplication.
#[derive(Debug, Parser)]
#[command(version)]
struct Options {
    /// Weight format.
    #[arg(long, value_enum, default_value_t = Quant::Q8_0)]
    format: Quant,
    /// Matrix rows.
    #[arg(long, default_value_t = 10752)]
    rows: u32,
    /// Matrix columns.
    #[arg(long, default_value_t = 2048)]
    cols: u32,
    /// Input tokens.
    #[arg(long, default_value_t = 512)]
    tokens: u32,
    /// Repeated GPU measurements after one warmup.
    #[arg(short, long, default_value_t = 20, value_parser = clap::value_parser!(u32).range(1..=1000))]
    reps: u32,
    /// How the result combines with the output buffer.
    #[arg(long, value_enum, default_value_t = StoreMode::Overwrite)]
    store: StoreMode,
    /// Expand Q4_0, Q4_K, or Q6_K weights to half precision before each matrix multiply.
    #[arg(long)]
    expand: bool,
    /// Matrix kernel selection for A/B comparisons.
    #[arg(long, value_enum, default_value_t = Matmul::Auto)]
    matmul: Matmul,
    /// Measure the decode matrix-vector kernel instead, one token per launch. With `--store
    /// swiglu`, each launch multiplies a gate and an up matrix with the input normalization fused.
    #[arg(long, conflicts_with_all = ["expand", "matmul", "tokens"])]
    matvec: bool,
}

#[cfg(target_os = "macos")]
fn main() -> ExitCode {
    match run(&Options::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            #[expect(clippy::print_stderr, reason = "benchmark errors go to stderr")]
            {
                eprintln!("bobcat-matmul-bench: {error}");
            }
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn main() -> ExitCode {
    let _ = Options::parse();
    #[expect(clippy::print_stderr, reason = "benchmark errors go to stderr")]
    {
        eprintln!("bobcat-matmul-bench: Metal requires macOS");
    }
    ExitCode::FAILURE
}

#[cfg(target_os = "macos")]
fn run(options: &Options) -> Result<(), Box<dyn std::error::Error>> {
    use bobcat::metal::{Format, Metal, Store};

    let (format, block_weights, block_bytes) = match options.format {
        Quant::F16 => (Format::F16, 32, 64),
        Quant::Q8_0 => (Format::Q8_0, 32, 34),
        Quant::Q4_0 => (Format::Q4_0, 32, 18),
        Quant::Q4k => (Format::Q4K, 256, 144),
        Quant::Q6k => (Format::Q6K, 256, 210),
    };
    if options.expand && !matches!(options.format, Quant::Q4_0 | Quant::Q4k | Quant::Q6k) {
        return Err("--expand requires --format q4-0, q4k, or q6k".into());
    }
    if options.matmul == Matmul::Tensor && !options.expand && !matches!(options.format, Quant::F16)
    {
        return Err("--matmul tensor requires --format f16 or --expand".into());
    }
    let store = match options.store {
        StoreMode::Overwrite => Store::Overwrite,
        StoreMode::Accumulate => Store::Accumulate,
        StoreMode::Swiglu => Store::Swiglu,
    };
    let rows = options.rows as usize;
    let cols = options.cols as usize;
    let tokens = options.tokens as usize;
    if rows == 0 || tokens == 0 || cols == 0 || !cols.is_multiple_of(block_weights) {
        return Err(format!(
            "rows and tokens must be positive, cols a multiple of {block_weights}"
        )
        .into());
    }
    let weight_bytes = rows
        .checked_mul(cols / block_weights)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or("weight size overflow")?;
    if options.matvec {
        return run_matvec(options, format, weight_bytes);
    }
    let input_len = tokens.checked_mul(cols).ok_or("input size overflow")?;
    let output_len = tokens.checked_mul(rows).ok_or("output size overflow")?;
    let metal = &mut match options.matmul {
        Matmul::Auto => Metal::open()?,
        Matmul::Simd => Metal::open_simd_matmul()?,
        Matmul::Tensor => Metal::open_tensor_matmul()?,
    };
    let weights = metal.new_buffer(weight_bytes)?;
    let input = metal.new_buffer(input_len.checked_mul(4).ok_or("input size overflow")?)?;
    let output = metal.new_buffer(output_len.checked_mul(4).ok_or("output size overflow")?)?;
    let expanded = if options.expand {
        let bytes = rows
            .checked_mul(cols)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or("weight size overflow")?;
        Some(metal.new_buffer(bytes)?)
    } else {
        None
    };
    metal.write(input.at(0), &vec![1.0_f32; input_len])?;

    let mut samples = Vec::with_capacity(options.reps as usize);
    for rep in 0..=options.reps {
        metal.begin()?;
        if let Some(expanded) = &expanded {
            metal.expand(
                format,
                weights.at(0),
                expanded.at(0),
                options.rows,
                options.cols,
            )?;
            metal.barrier();
        }
        metal.matmul(
            if options.expand { Format::F16 } else { format },
            expanded.as_ref().unwrap_or(&weights).at(0),
            options.rows,
            options.cols,
            input.at(0),
            output.at(0),
            options.tokens,
            store,
        )?;
        let seconds = metal.end()?;
        if rep > 0 {
            samples.push(seconds);
        }
    }
    let n = f64::from(options.reps);
    let mean = samples.iter().sum::<f64>() / n;
    let deviation = if samples.len() > 1 {
        (samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    } else {
        0.0
    };
    println!("command submission: Metal 4");
    println!(
        "{} {:?} {:?} {}x{} by {} tokens, {} runs: {:.3} ± {:.3} ms GPU, {:.1} tokens/s",
        if metal.uses_tensor_matmul() && (options.expand || matches!(options.format, Quant::F16)) {
            "Metal 4 tensor"
        } else {
            "SIMD-group"
        },
        options.format,
        options.store,
        options.rows,
        options.cols,
        options.tokens,
        options.reps,
        mean * 1e3,
        deviation * 1e3,
        n * f64::from(options.tokens) / samples.iter().sum::<f64>()
    );
    Ok(())
}

/// Measure the decode matrix-vector kernel of `format` on `weight_bytes` of weights.
///
/// Each run multiplies enough copies of the matrix to fill 512 MB, one launch per copy with a
/// barrier between launches, as the model's dependent steps have. The copies keep the weights
/// out of the GPU's caches, so the rate is the rate a model reads its weights at.
#[cfg(target_os = "macos")]
fn run_matvec(
    options: &Options,
    format: bobcat::metal::Format,
    weight_bytes: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use bobcat::metal::{MatvecOptions, Metal, Norm};

    const TOTAL_BYTES: usize = 512 << 20;
    let swiglu = matches!(options.store, StoreMode::Swiglu);
    // A SwiGLU launch reads a gate matrix and an up matrix.
    let launch_bytes = if swiglu {
        2 * weight_bytes
    } else {
        weight_bytes
    };
    let copies = TOTAL_BYTES.div_ceil(launch_bytes);
    let rows = options.rows as usize;
    let cols = options.cols as usize;
    let metal = &mut Metal::open()?;
    let weights = metal.new_buffer(copies * launch_bytes)?;
    let input = metal.new_buffer(cols * 4)?;
    let norm_weight = metal.new_buffer(cols * 4)?;
    let output = metal.new_buffer(rows * 4)?;
    // Nonzero bytes make every page of the weights real memory.
    metal.write(weights.at(0), &vec![0x11_u8; copies * launch_bytes])?;
    metal.write(input.at(0), &vec![1.0_f32; cols])?;
    metal.write(norm_weight.at(0), &vec![1.0_f32; cols])?;

    let mut samples = Vec::with_capacity(options.reps as usize);
    for rep in 0..=options.reps {
        metal.begin()?;
        for copy in 0..copies {
            let start = copy * launch_bytes;
            if swiglu {
                metal.matvec_swiglu(
                    format,
                    weights.at(start),
                    weights.at(start + weight_bytes),
                    options.rows,
                    options.cols,
                    input.at(0),
                    Norm {
                        weight: norm_weight.at(0),
                        eps: 1e-5,
                    },
                    output.at(0),
                )?;
            } else {
                metal.matvec(
                    format,
                    weights.at(start),
                    options.rows,
                    options.cols,
                    input.at(0),
                    output.at(0),
                    MatvecOptions::default(),
                )?;
            }
            metal.barrier();
        }
        let seconds = metal.end()?;
        if rep > 0 {
            samples.push(seconds / copies as f64);
        }
    }
    let n = f64::from(options.reps);
    let mean = samples.iter().sum::<f64>() / n;
    println!(
        "matvec {:?} {:?} {}x{}, {} copies, {} runs: {:.1} us per launch, {:.1} GB/s",
        options.format,
        options.store,
        options.rows,
        options.cols,
        copies,
        options.reps,
        mean * 1e6,
        launch_bytes as f64 / mean / 1e9
    );
    Ok(())
}
