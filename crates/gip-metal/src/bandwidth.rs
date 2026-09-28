//! A measurement of the memory read bandwidth the Metal GPU reaches.
//!
//! GPU decode speed is bounded by that bandwidth divided by the model size.

#![expect(
    unsafe_code,
    reason = "Metal's API is Objective-C, reached through objc2-metal"
)]

use std::ptr::NonNull;

use objc2::rc::autoreleasepool;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use crate::backend::Error;

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
        let queue = device.newCommandQueue().ok_or(Error::Queue)?;
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
            .map_err(|error| Error::Compile(error.localizedDescription().to_string()))?;
        let pipeline_error = |message| Error::Pipeline {
            name: "read_sum",
            message,
        };
        let function = library
            .newFunctionWithName(&NSString::from_str("read_sum"))
            .ok_or_else(|| pipeline_error("no such kernel".to_owned()))?;
        let pipeline = device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| pipeline_error(error.localizedDescription().to_string()))?;

        let n_vectors = u32::try_from(buffer_bytes / VECTOR_BYTES)
            .map_err(|_| Error::Allocation(buffer_bytes))?;
        let source = device
            .newBufferWithLength_options(buffer_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::Allocation(buffer_bytes))?;
        // Touching every page first keeps page faults out of the timed dispatches.
        let contents = source.contents().cast::<u8>();
        // SAFETY: `contents` points at the `source.length()` bytes of a buffer no command buffer
        // has used yet.
        unsafe { contents.write_bytes(1, source.length()) };

        let max_threads = grid_sizes.iter().copied().max().unwrap_or(1);
        let out_bytes = max_threads * size_of::<f32>();
        let out = device
            .newBufferWithLength_options(out_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::Allocation(out_bytes))?;

        let mut samples = Vec::with_capacity(grid_sizes.len());
        for &threads in grid_sizes {
            let mut best_seconds = f64::INFINITY;
            for _ in 0..reps {
                let seconds = dispatch(&queue, &pipeline, &source, &out, n_vectors, threads)?;
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

/// Run one pass of `threads` threads over the `n_vectors` float4 values of `source` and return
/// its GPU time in seconds.
fn dispatch(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    source: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n_vectors: u32,
    threads: usize,
) -> Result<f64, Error> {
    let command_buffer = queue.commandBuffer().ok_or(Error::CommandBuffer)?;
    let encoder = command_buffer
        .computeCommandEncoder()
        .ok_or(Error::CommandBuffer)?;
    encoder.setComputePipelineState(pipeline);
    // SAFETY: the kernel reads `n_vectors` float4 values, which fill `source`.
    unsafe { encoder.setBuffer_offset_atIndex(Some(source), 0, 0) };
    // SAFETY: the kernel writes one float per thread, and `out` holds one for the largest grid.
    unsafe { encoder.setBuffer_offset_atIndex(Some(out), 0, 1) };
    let count = NonNull::from(&n_vectors).cast();
    // SAFETY: `count` points at a live `u32`, and Metal copies it before returning.
    unsafe { encoder.setBytes_length_atIndex(count, size_of::<u32>(), 2) };
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADGROUP_SIZE,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    // GPU timestamps exclude host encoding and scheduling time.
    Ok(command_buffer.GPUEndTime() - command_buffer.GPUStartTime())
}
