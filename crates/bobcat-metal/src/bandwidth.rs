//! A measurement of the memory read bandwidth the Metal GPU reaches.
//!
//! GPU decode speed is bounded by that bandwidth divided by the model size.

#![expect(
    unsafe_code,
    reason = "Metal's API is Objective-C, reached through objc2-metal"
)]

use objc2::rc::autoreleasepool;
use objc2_metal::{
    MTL4CompilerDescriptor, MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily,
    MTLResourceOptions,
};

use super::commands::Commands;
use super::{Arg, Buffer, Dispatch, Error, compile_library, describe, make_pipeline};

/// Each thread sums float4 values at a stride of the grid size, so neighboring threads read
/// neighboring addresses.
const SOURCE: &str = "
#include <metal_stdlib>
using namespace metal;
kernel void read_sum (device const float4 *src [[buffer (0)]],
                      device float *out [[buffer (1)]],
                      constant uint &n [[buffer (2)]],
                      uint gid [[thread_position_in_grid]],
                      uint grid [[threads_per_grid]])
{
  float4 acc = 0.0f;
  for (uint i = gid; i < n; i += grid)
    acc += src[i];
  out[gid] = acc.x + acc.y + acc.z + acc.w;
}
";

const THREADGROUP_SIZE: usize = 256;
const VECTOR_BYTES: usize = 16;

/// The bandwidth one grid size reached.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BandwidthSample {
    /// The number of threads that read the buffer.
    pub threads: usize,
    /// The best read bandwidth in GB/s.
    pub gigabytes_per_second: f64,
}

/// A measurement of the GPU's read bandwidth.
#[derive(Debug, Clone, PartialEq)]
pub struct Bandwidth {
    /// The GPU's name.
    pub device: String,
    /// One sample per grid size.
    pub samples: Vec<BandwidthSample>,
}

/// Return the best read bandwidth over `reps` passes of the GPU over a buffer of `buffer_bytes`
/// bytes, for each grid size in `grid_sizes`.
pub fn measure_bandwidth(
    buffer_bytes: usize,
    grid_sizes: &[usize],
    reps: usize,
) -> Result<Bandwidth, Error> {
    autoreleasepool(|_| {
        let device = MTLCreateSystemDefaultDevice().ok_or(Error::NoDevice)?;
        if !device.supportsFamily(MTLGPUFamily::Metal4) {
            return Err(Error::Metal4Unsupported);
        }
        let compiler = device
            .newCompilerWithDescriptor_error(&MTL4CompilerDescriptor::new())
            .map_err(|error| Error::Compile(describe(&error)))?;
        let library = compile_library(&compiler, SOURCE)?;
        let pipeline = make_pipeline(&compiler, &library, "read_sum", None)?;
        let mut commands = Commands::new(&device)?;

        let n_vectors = u32::try_from(buffer_bytes / VECTOR_BYTES)
            .map_err(|_| Error::Allocation(buffer_bytes))?;
        if n_vectors == 0 {
            return Err(Error::BandwidthInput("the buffer must hold a float4"));
        }
        if reps == 0 {
            return Err(Error::BandwidthInput(
                "the repetition count must be positive",
            ));
        }
        if grid_sizes
            .iter()
            .any(|&threads| threads == 0 || u32::try_from(threads).is_err())
        {
            return Err(Error::BandwidthInput(
                "grid sizes must be positive u32 values",
            ));
        }
        let source = device
            .newBufferWithLength_options(buffer_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::Allocation(buffer_bytes))?;
        // Touching every page first keeps page faults out of the timed dispatches.
        let contents = source.contents().cast::<u8>();
        // SAFETY: `contents` points at the `source.length()` bytes of a buffer no command buffer
        // has used yet.
        unsafe { contents.write_bytes(1, source.length()) };
        let source = Buffer {
            owner: 0,
            address: source.gpuAddress(),
            raw: source,
            len: buffer_bytes,
        };

        let max_threads = grid_sizes.iter().copied().max().unwrap_or(1);
        let out_bytes = max_threads
            .checked_mul(size_of::<f32>())
            .ok_or(Error::Allocation(usize::MAX))?;
        let out = device
            .newBufferWithLength_options(out_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::Allocation(out_bytes))?;
        let out = Buffer {
            owner: 0,
            address: out.gpuAddress(),
            raw: out,
            len: out_bytes,
        };

        let mut samples = Vec::with_capacity(grid_sizes.len());
        for &threads in grid_sizes {
            let mut best_seconds = f64::INFINITY;
            for _ in 0..reps {
                commands.begin()?;
                let args = [
                    Arg::Buffer {
                        view: source.at(0),
                        bytes: buffer_bytes,
                    },
                    Arg::Buffer {
                        view: out.at(0),
                        bytes: threads * size_of::<f32>(),
                    },
                    Arg::U32(n_vectors),
                ];
                if let Err(error) = commands.launch(
                    &pipeline,
                    &args,
                    &Dispatch::Threads([threads, 1, 1], [THREADGROUP_SIZE, 1, 1]),
                    false,
                ) {
                    commands.discard();
                    return Err(error);
                }
                let ticket = commands.commit()?;
                // Commit feedback timestamps exclude host encoding and scheduling time.
                let seconds = commands.wait(ticket)?;
                best_seconds = best_seconds.min(seconds);
            }
            samples.push(BandwidthSample {
                threads,
                gigabytes_per_second: f64::from(n_vectors) * VECTOR_BYTES as f64
                    / best_seconds
                    / 1e9,
            });
        }
        Ok(Bandwidth {
            device: device.name().to_string(),
            samples,
        })
    })
}
