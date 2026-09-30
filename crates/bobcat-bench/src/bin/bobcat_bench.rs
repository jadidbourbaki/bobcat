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
    /// Measure warmed greedy streaming latency.
    #[arg(long)]
    latency: bool,
    /// Skip throughput measurements when measuring streaming latency.
    #[arg(long, requires = "latency")]
    latency_only: bool,
    /// Output tokens per streaming call after the first token.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=1024))]
    stream_chunk: u32,
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

    if options.latency && options.generate < 2 {
        return Err(bobcat::Error::Argument(
            "latency needs at least two generated tokens",
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
    if !options.latency_only {
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
    }

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
        println!(
            "engine,run,prompt_tokens,generated_tokens,stream_chunk,ttft_ms,tpot_ms,end_to_end_ms,token_hash"
        );
        let mut first = Vec::new();
        let mut per_output = Vec::new();
        let mut end_to_end = Vec::new();
        let mut inter_chunk = Vec::new();
        for rep in 0..=options.reps {
            let mut gpu = Lfm2Metal::new(&model, &mut metal, n_ctx, kv_half)?;
            let start = Instant::now();
            gpu.prefill(&prompt, None, None)?;
            let mut emissions = Vec::new();
            if options.stream_chunk == 1 {
                // Each token goes out as soon as the step that selects it finishes, with later
                // steps already submitted.
                let mut emitted = 0;
                gpu.generate_stream(options.generate, |token| {
                    generated[emitted] = token;
                    emitted += 1;
                    emissions.push(start.elapsed().as_secs_f64() * 1e3);
                    true
                })?;
            } else {
                generated[0] = gpu.greedy_token()?;
                emissions.push(start.elapsed().as_secs_f64() * 1e3);
                for chunk in generated[1..].chunks_mut(options.stream_chunk as usize) {
                    gpu.generate_next(chunk)?;
                    emissions.push(start.elapsed().as_secs_f64() * 1e3);
                }
            }
            if rep > 0 {
                let ttft = emissions[0];
                let total = *emissions.last().expect("generation has output chunks");
                first.push(ttft);
                end_to_end.push(total);
                per_output.push((total - ttft) / f64::from(options.generate - 1));
                inter_chunk.extend(emissions.windows(2).map(|pair| pair[1] - pair[0]));
                let hash = generated.iter().fold(0_u64, |hash, &token| {
                    hash.wrapping_mul(1_000_003).wrapping_add(u64::from(token))
                });
                println!(
                    "bobcat,{rep},{},{},{},{ttft:.6},{:.6},{total:.6},{hash}",
                    options.prompt,
                    options.generate,
                    options.stream_chunk,
                    (total - ttft) / f64::from(options.generate - 1)
                );
            }
        }
        println!(
            "\nstreaming latency, warmed model, pre-tokenized prompt, first token then greedy chunks of up to {}:",
            options.stream_chunk
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
    let deviation = if sorted.len() > 1 {
        (sorted
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (sorted.len() - 1) as f64)
            .sqrt()
    } else {
        0.0
    };
    let p50 = sorted[sorted.len().div_ceil(2) - 1];
    let p95 = sorted[(sorted.len() * 95).div_ceil(100) - 1];
    println!(
        "{label:<12} mean {mean:8.3} ms, sd {deviation:8.3} ms, p50 {p50:8.3} ms, p95 {p95:8.3} ms"
    );
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

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::Options;

    #[test]
    fn latency_only_requires_latency() {
        assert!(Options::try_parse_from(["bobcat-bench", "model.gguf", "--latency-only"]).is_err());
    }

    #[test]
    fn streaming_chunk_rejects_empty_and_oversized_calls() {
        for chunk in ["0", "1025"] {
            assert!(
                Options::try_parse_from(["bobcat-bench", "model.gguf", "--stream-chunk", chunk])
                    .is_err()
            );
        }
    }
}
