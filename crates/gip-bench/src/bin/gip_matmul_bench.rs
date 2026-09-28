//! Measure one Metal matrix multiplication shape without the model graph.

#![expect(
    clippy::print_stdout,
    reason = "gip-matmul-bench reports measurements on stdout"
)]

use std::process::ExitCode;

use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Quant {
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
}

#[cfg(target_os = "macos")]
fn main() -> ExitCode {
    match run(&Options::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            #[expect(clippy::print_stderr, reason = "benchmark errors go to stderr")]
            {
                eprintln!("gip-matmul-bench: {error}");
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
        eprintln!("gip-matmul-bench: Metal requires macOS");
    }
    ExitCode::FAILURE
}

#[cfg(target_os = "macos")]
fn run(options: &Options) -> Result<(), Box<dyn std::error::Error>> {
    use gip::metal::{Format, Metal, Store};

    let (format, block_weights, block_bytes) = match options.format {
        Quant::Q8_0 => (Format::Q8_0, 32, 34),
        Quant::Q4_0 => (Format::Q4_0, 32, 18),
        Quant::Q4k => (Format::Q4K, 256, 144),
        Quant::Q6k => (Format::Q6K, 256, 210),
    };
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
    let input_len = tokens.checked_mul(cols).ok_or("input size overflow")?;
    let output_len = tokens.checked_mul(rows).ok_or("output size overflow")?;
    let metal = &mut Metal::open()?;
    let weights = metal.new_buffer(weight_bytes)?;
    let input = metal.new_buffer(input_len.checked_mul(4).ok_or("input size overflow")?)?;
    let output = metal.new_buffer(output_len.checked_mul(4).ok_or("output size overflow")?)?;
    metal.write(input.at(0), &vec![1.0_f32; input_len])?;

    let mut samples = Vec::with_capacity(options.reps as usize);
    for rep in 0..=options.reps {
        metal.begin()?;
        metal.matmul(
            format,
            weights.at(0),
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
    println!(
        "{:?} {:?} {}x{} by {} tokens, {} runs: {:.3} ± {:.3} ms GPU, {:.1} tokens/s",
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
