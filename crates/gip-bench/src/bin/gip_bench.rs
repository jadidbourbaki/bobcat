//! gip-bench measures gip's prefill and decode speed on the Metal GPU.
//!
//! The protocol matches llama-bench and `mlx_lm.benchmark`: a prompt of fixed length, then a fixed
//! number of generated tokens.

#![expect(
    clippy::print_stdout,
    reason = "gip-bench reports its measurements on stdout"
)]

use std::process::ExitCode;

use clap::{Parser, ValueEnum};

/// The prompt repeats a fixed ordinary token. Speed does not depend on which tokens the prompt
/// holds.
#[cfg(target_os = "macos")]
const PROMPT_TOKEN: u32 = 1000;

/// The element type of the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum KvType {
    F16,
    F32,
}

/// Measure gip's prefill and decode speed on the Metal GPU.
#[derive(Debug, Parser)]
#[command(version)]
struct Options {
    /// The GGUF model file.
    model: std::path::PathBuf,
    /// Prompt tokens.
    #[arg(short, long, default_value_t = 512, value_parser = clap::value_parser!(u32).range(1..=1_000_000))]
    prompt: u32,
    /// Generated tokens.
    #[arg(short = 'n', long, default_value_t = 128, value_parser = clap::value_parser!(u32).range(1..=1_000_000))]
    generate: u32,
    /// Repetitions.
    #[arg(short, long, default_value_t = 5, value_parser = clap::value_parser!(u32).range(1..=1_000))]
    reps: u32,
    /// Break prefill and decode GPU time down by kernel.
    #[arg(short = 'P', long)]
    profile: bool,
    /// KV cache type.
    #[arg(short, long, value_enum, default_value_t = KvType::F16)]
    kv: KvType,
}

fn main() -> ExitCode {
    let options = Options::parse();
    match run(&options) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            #[expect(clippy::print_stderr, reason = "gip-bench reports errors on stderr")]
            {
                eprintln!("gip-bench: {error}");
            }
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn run(_: &Options) -> Result<(), gip::Error> {
    Err(gip::Error::MetalUnsupported("macOS".to_owned()))
}

/// Load the model and time the runs `options` asks for.
#[cfg(target_os = "macos")]
fn run(options: &Options) -> Result<(), gip::Error> {
    use std::time::Instant;

    use gip::metal::Metal;
    use gip::{Lfm2Metal, Model};

    let model = Model::load(&options.model)?;
    let mut metal = Metal::open()?;
    let kv_half = options.kv == KvType::F16;
    let n_ctx = options.prompt + options.generate;
    let prompt = vec![PROMPT_TOKEN; options.prompt as usize];
    let mut generated = vec![0; options.generate as usize];
    let mut prefill_rates = Vec::new();
    let mut decode_rates = Vec::new();
    let (mut decode_seconds, mut encode_seconds, mut gpu_seconds) = (0.0, 0.0, 0.0);

    // The first run warms the shader cache and the page cache and does not count.
    for rep in 0..=options.reps {
        let mut gpu = Lfm2Metal::new(&model, &mut metal, n_ctx, kv_half)?;
        let start = Instant::now();
        gpu.prefill(&prompt, None, None)?;
        let prefill = start.elapsed().as_secs_f64();

        // Decode runs pipelined, the way an application generates text.
        let start = Instant::now();
        gpu.generate(&mut generated)?;
        let decode = start.elapsed().as_secs_f64();

        if rep > 0 {
            prefill_rates.push(f64::from(options.prompt) / prefill);
            decode_rates.push(f64::from(options.generate) / decode);
            decode_seconds += decode;
            encode_seconds += gpu.last_encode_seconds();
            gpu_seconds += gpu.last_gpu_seconds();
        }
    }

    println!(
        "{}, {} prompt and {} generated tokens, {} runs",
        options.model.display(),
        options.prompt,
        options.generate,
        options.reps
    );
    print_rates("prefill", &prefill_rates);
    print_rates("decode", &decode_rates);
    // Encoding overlaps the GPU in pipelined decode. The GPU sits idle for whatever part of each
    // token its own work does not cover.
    let per_token = 1e3 / f64::from(options.generate * options.reps);
    println!(
        "decode per token: {:.3} ms total, {:.3} ms GPU busy, {:.3} ms GPU idle, {:.3} ms CPU \
         encoding",
        decode_seconds * per_token,
        gpu_seconds * per_token,
        (decode_seconds - gpu_seconds) * per_token,
        encode_seconds * per_token
    );

    if options.profile {
        // Prefill runs in one batched call, then decode steps one token at a time, with
        // profiling on for both.
        let mut gpu = Lfm2Metal::new(&model, &mut metal, n_ctx, kv_half)?;
        let mut logits = vec![0.0; model.hyperparameters().n_vocab as usize];
        gpu.metal().set_profiling(true);
        gpu.prefill(&prompt, Some(&mut logits), None)?;
        print_profile(gpu.metal(), "prefill", 1);

        gpu.metal().set_profiling(true);
        for _ in 0..options.generate {
            gpu.step(argmax(&logits), Some(&mut logits), None)?;
        }
        gpu.metal().set_profiling(false);
        print_profile(gpu.metal(), "decode", options.generate);
    }
    Ok(())
}

/// Print the mean and sample standard deviation of `rates` under `label`.
fn print_rates(label: &str, rates: &[f64]) {
    let n = rates.len() as f64;
    let mean = rates.iter().sum::<f64>() / n;
    let variance = if rates.len() > 1 {
        rates.iter().map(|rate| (rate - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    println!("{label:<8} {mean:9.2} ± {:.2} tokens/s", variance.sqrt());
}

/// Return the index of the largest of `x`. Ties go to the lowest index.
#[cfg(target_os = "macos")]
fn argmax(x: &[f32]) -> u32 {
    let best = x.iter().enumerate().fold(
        0,
        |best, (i, &value)| if value > x[best] { i } else { best },
    );
    u32::try_from(best).unwrap_or(0)
}

/// Print the profile of `metal` under `label`, with GPU times divided by `n_units`, the number of
/// passes or tokens the profile covers.
#[cfg(target_os = "macos")]
fn print_profile(metal: &gip::metal::Metal, label: &str, n_units: u32) {
    let mut entries = metal.profile().to_vec();
    entries.sort_by(|a, b| b.seconds.total_cmp(&a.seconds));
    let total: f64 = entries.iter().map(|entry| entry.seconds).sum();
    let units = f64::from(n_units);

    println!("\nprofiled {label}, one command buffer per launch:");
    println!(
        "{:<18} {:>13} {:>10} {:>9} {:>6} {:>8}",
        "kernel", "shape", "launches", "ms/unit", "share", "GB/s"
    );
    for entry in &entries {
        let shape = if entry.n_rows == 0 {
            String::new()
        } else {
            format!("{}x{}", entry.n_rows, entry.n_cols)
        };
        let bandwidth = if entry.bytes == 0 {
            String::new()
        } else {
            format!(" {:8.1}", entry.bytes as f64 / entry.seconds / 1e9)
        };
        println!(
            "{:<18} {shape:>13} {:>10} {:>9.3} {:>5.1}%{bandwidth}",
            entry.name,
            entry.calls / u64::from(n_units),
            entry.seconds * 1e3 / units,
            100.0 * entry.seconds / total
        );
    }
    println!(
        "{:<18} {:>13} {:>10} {:>9.3}",
        "total",
        "",
        "",
        total * 1e3 / units
    );
}
