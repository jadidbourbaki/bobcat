//! gpu-bw measures the memory read bandwidth that the Metal GPU reaches.
//!
//! GPU decode speed is bounded by that bandwidth divided by the model size.

#![expect(
    clippy::print_stdout,
    reason = "gpu-bw reports its measurements on stdout"
)]

use std::process::ExitCode;

use clap::Parser;

const REPS: usize = 10;
const GRID_SIZES: [usize; 4] = [1 << 14, 1 << 16, 1 << 18, 1 << 20];

/// Measure the memory read bandwidth of the Metal GPU.
#[derive(Debug, Parser)]
#[command(version)]
struct Options {
    /// Read a buffer of this many megabytes.
    #[arg(short, long, default_value_t = 2048, value_parser = clap::value_parser!(u64).range(1..=16384))]
    size: u64,
}

#[cfg(target_os = "macos")]
fn main() -> ExitCode {
    let options = Options::parse();
    let buffer_bytes = usize::try_from(options.size).unwrap_or(usize::MAX) * 1024 * 1024;
    match gip::metal::measure_bandwidth(buffer_bytes, &GRID_SIZES, REPS) {
        Ok(bandwidth) => {
            println!(
                "{}, buffer {} MB, best of {REPS} dispatches",
                bandwidth.device, options.size
            );
            println!("threads     GB/s");
            for sample in bandwidth.samples {
                println!("{:7}  {:7.1}", sample.threads, sample.gigabytes_per_second);
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            #[expect(clippy::print_stderr, reason = "gpu-bw reports errors on stderr")]
            {
                eprintln!("gpu-bw: {error}");
            }
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn main() -> ExitCode {
    let _ = Options::parse();
    #[expect(clippy::print_stderr, reason = "gpu-bw reports errors on stderr")]
    {
        eprintln!("gpu-bw: this build has no Metal backend");
    }
    ExitCode::FAILURE
}
