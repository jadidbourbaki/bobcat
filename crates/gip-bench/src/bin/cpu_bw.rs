//! cpu-bw measures the memory read bandwidth that CPU threads reach.
//!
//! CPU decode speed is bounded by that bandwidth divided by the model size.

#![expect(
    clippy::print_stdout,
    reason = "cpu-bw reports its measurements on stdout"
)]

use std::hint::black_box;
use std::process::ExitCode;
use std::thread;
use std::time::Instant;

use clap::Parser;

const REPS: usize = 10;
const THREAD_COUNTS: [usize; 8] = [1, 2, 4, 6, 8, 10, 12, 14];

/// Sixteen floats per step feed sixteen independent accumulators, which LLVM keeps in four NEON
/// registers. The independent sums keep the loop bound by load throughput.
const LANES: usize = 16;

/// Measure the memory read bandwidth of CPU threads.
#[derive(Debug, Parser)]
#[command(version)]
struct Options {
    /// Read a buffer of this many megabytes.
    #[arg(short, long, default_value_t = 2048, value_parser = clap::value_parser!(u64).range(1..))]
    size: u64,
}

/// Return the sum of the floats in `slice`.
fn read_slice(slice: &[f32]) -> f32 {
    let mut lanes = [0.0_f32; LANES];
    for chunk in slice.as_chunks::<LANES>().0 {
        for (lane, &value) in lanes.iter_mut().zip(chunk) {
            *lane += value;
        }
    }
    lanes.iter().sum()
}

/// Return the best read bandwidth in GB/s over [`REPS`] passes in which `n_threads` threads split
/// `data`.
fn measure(data: &[f32], n_threads: usize) -> f64 {
    let per_thread = data.len() / n_threads / LANES * LANES;
    let mut best_seconds = f64::INFINITY;
    for _ in 0..REPS {
        let start = Instant::now();
        thread::scope(|scope| {
            for slice in data.chunks_exact(per_thread).take(n_threads) {
                scope.spawn(move || black_box(read_slice(slice)));
            }
        });
        best_seconds = best_seconds.min(start.elapsed().as_secs_f64());
    }
    let bytes_read = (per_thread * n_threads * size_of::<f32>()) as f64;
    bytes_read / best_seconds / 1e9
}

fn main() -> ExitCode {
    let options = Options::parse();
    let Some(count) = usize::try_from(options.size)
        .ok()
        .and_then(|mb| mb.checked_mul(1024 * 1024 / size_of::<f32>()))
    else {
        #[expect(clippy::print_stderr, reason = "cpu-bw reports errors on stderr")]
        {
            eprintln!("cpu-bw: invalid size: {}", options.size);
        }
        return ExitCode::FAILURE;
    };
    // Filling the buffer touches every page, which keeps page faults out of the timed passes.
    let data = vec![1.0_f32; count];

    println!("buffer {} MB, best of {REPS} passes", options.size);
    println!("threads  GB/s");
    for n_threads in THREAD_COUNTS {
        println!("{n_threads:7}  {:6.1}", measure(&data, n_threads));
    }
    ExitCode::SUCCESS
}
