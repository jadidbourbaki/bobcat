//! bobcat-bench measures bobcat's prefill and decode speed on the Metal GPU.
//!
//! The protocol matches llama-bench and `mlx_lm.benchmark`: a prompt of fixed length, then a fixed
//! number of generated tokens.

#![expect(
    clippy::print_stdout,
    reason = "bobcat-bench reports its measurements on stdout"
)]

use std::process::ExitCode;

use clap::{Parser, ValueEnum};

/// The prompt repeats a fixed ordinary token. Speed does not depend on which tokens the prompt
/// holds.
#[cfg(target_os = "macos")]
const PROMPT_TOKEN: u32 = 1000;
#[cfg(target_os = "macos")]
const DECODE_CHUNK: usize = 8;

/// The element type of the KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum KvType {
    F16,
    F32,
}

/// The matrix kernel selection for benchmark comparisons.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum Matmul {
    Auto,
    Simd,
    Tensor,
}

/// Measure bobcat's prefill and decode speed on the Metal GPU.
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
    /// Measure warmed greedy streaming latency, then emit in eight-token chunks.
    #[arg(long)]
    latency: bool,
    /// Prefill matrix kernel selection for A/B comparisons.
    #[arg(long, value_enum, default_value_t = Matmul::Auto)]
    matmul: Matmul,
    /// KV cache type.
    #[arg(short, long, value_enum, default_value_t = KvType::F16)]
    kv: KvType,
}

fn main() -> ExitCode {
    let options = Options::parse();
    match run(&options) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            #[expect(clippy::print_stderr, reason = "bobcat-bench reports errors on stderr")]
            {
                eprintln!("bobcat-bench: {error}");
            }
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn run(_: &Options) -> Result<(), bobcat::Error> {
    Err(bobcat::Error::MetalUnsupported("macOS".to_owned()))
}

/// Load the model and time the runs `options` asks for.
#[cfg(target_os = "macos")]
fn run(options: &Options) -> Result<(), bobcat::Error> {
    use std::time::Instant;

    use bobcat::metal::Metal;
    use bobcat::{Lfm2Metal, Model};

    if options.latency
        && options.generate <= u32::try_from(DECODE_CHUNK).expect("chunk size fits u32")
    {
        return Err(bobcat::Error::Argument(
            "latency needs more than eight generated tokens",
        ));
    }

    let model = Model::load(&options.model)?;
    let mut metal = match options.matmul {
        Matmul::Auto => Metal::open()?,
        Matmul::Simd => Metal::open_simd_matmul()?,
        Matmul::Tensor => Metal::open_tensor_matmul()?,
    };
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
    println!("command submission: Metal 4");
    if metal.uses_tensor_matmul() {
        println!("prefill matrix path: Metal 4 tensors for expanded half weights");
    }
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
    if options.latency {
        let mut first = Vec::new();
        let mut per_output = Vec::new();
        let mut end_to_end = Vec::new();
        let mut inter_chunk = Vec::new();
        for rep in 0..=options.reps {
            let mut gpu = Lfm2Metal::new(&model, &mut metal, n_ctx, kv_half)?;
            let start = Instant::now();
            gpu.prefill(&prompt, None, None)?;
            let mut emissions = Vec::new();
            // A caller can observe the first token only after the first generate call returns.
            gpu.generate(&mut generated[..1])?;
            emissions.push(start.elapsed().as_secs_f64() * 1e3);
            for chunk in generated[1..].chunks_mut(DECODE_CHUNK) {
                gpu.generate(chunk)?;
                emissions.push(start.elapsed().as_secs_f64() * 1e3);
            }
            if rep > 0 {
                let ttft = emissions[0];
                let total = *emissions.last().expect("generation has output chunks");
                first.push(ttft);
                end_to_end.push(total);
                per_output.push((total - ttft) / f64::from(options.generate - 1));
                inter_chunk.extend(emissions.windows(2).map(|pair| pair[1] - pair[0]));
            }
        }
        println!(
            "\nstreaming latency, warmed model, pre-tokenized prompt, first token then greedy chunks of up to eight:"
        );
        print_milliseconds("TTFT", &first);
        print_milliseconds("TPOT", &per_output);
        print_milliseconds("end to end", &end_to_end);
        print_milliseconds("inter-chunk", &inter_chunk);
    }
    Ok(())
}

/// Print the mean and nearest-rank median and 95th percentile in milliseconds.
#[cfg(target_os = "macos")]
fn print_milliseconds(label: &str, values: &[f64]) {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let p50 = sorted[sorted.len().div_ceil(2) - 1];
    let p95 = sorted[(sorted.len() * 95).div_ceil(100) - 1];
    println!("{label:<12} mean {mean:8.3} ms, p50 {p50:8.3} ms, p95 {p95:8.3} ms");
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
fn print_profile(metal: &bobcat::metal::Metal, label: &str, n_units: u32) {
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
