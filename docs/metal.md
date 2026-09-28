# Metal

gip runs models on the GPU through Metal, Apple's GPU API. The
`gip-metal` crate compiles gip's kernels and launches them. The
`Lfm2Metal` type in `crates/gip/src/lfm2_metal.rs` strings the launches
together into the LFM2 forward pass.

## Terms

| Term | Meaning |
|---|---|
| Kernel | A function that runs on the GPU, written in the Metal Shading Language |
| Thread | One run of a kernel, usually on one element or one row |
| Simdgroup | 32 threads that run in lockstep and can exchange values in registers |
| Threadgroup | A few simdgroups that share fast threadgroup memory and can wait on each other |
| Pipeline | A compiled kernel, ready to launch |
| Command buffer | A list of launches that the CPU records and the GPU runs in order |
| Matvec | A matrix times one vector, the core of decoding one token |
| Matmul | A matrix times many vectors, the core of prefilling a prompt |

## Why decode speed is memory speed

Decoding one token multiplies every weight matrix by one vector. Each
weight is read once and used for one multiply and one add, so the GPU
spends its time waiting on memory. The fastest possible decode reads
every weight byte once per token at the GPU's full memory bandwidth.
`gpu-bw` measures that bandwidth. The decode ceiling is the bandwidth
divided by the model's bytes.

Quantization raises the ceiling. A Q4_K_M file holds about half the
bytes of a Q8_0 file, so each token reads half as much memory.

Prefill is different. A prompt of 512 tokens multiplies each matrix by
512 vectors, so each weight is read once and used 512 times. Prefill is
bound by arithmetic. The prefill kernels are built around the GPU's
matrix units.

## Compiling the kernels

The kernels live in seven files in `crates/gip-metal/src`.

| File | Kernels |
|---|---|
| `common.metal` | Shared constants, function constants, and reduction helpers |
| `quant.metal` | Readers for the Q4_0, Q4_K, and Q6_K block layouts |
| `matvec.metal` | Matrix-vector products for decoding |
| `norm.metal` | RMS normalization, rotary embeddings, embedding lookup, copies, and argmax |
| `attention.metal` | Attention over the KV cache |
| `conv.metal` | The short convolution |
| `matmul.metal` | Matrix-matrix products for prefill |

`include_str!` embeds the files in the library. `Metal::open` joins
them into one source string and compiles it with
`newLibraryWithSource` when the program starts. Building gip therefore
needs no Xcode, only the Command Line Tools. macOS caches the compiled
result between runs.

`Metal::open` builds every pipeline at load time. Metal function
constants specialize one kernel source into several pipelines, and the
compiler removes the branches each pipeline never takes.

| Function constant | Meaning |
|---|---|
| `rows_per_threadgroup` | How many matrix rows one threadgroup computes |
| `fuse_norm` | Whether the matvec applies RMS normalization to its input first |
| `accumulate` | Whether the kernel adds its product to the output |
| `swiglu_store` | Whether the matmul stores SiLU of the stored gate times its product |

Each quantization format has its own set of pipelines. `Format` names
the four formats the kernels read: Q8_0, Q4_0, Q4_K, and Q6_K. The
Q4_K and Q6_K matvec kernels follow the layout of llama.cpp's
`kernel_mul_mv_q4_K_f32` and `kernel_mul_mv_q6_K_f32`, where each
simdgroup owns whole rows and unpacks each super-block's scales once.

## Buffers and safety

`Lfm2Metal::new` wraps the whole memory-mapped GGUF file in one Metal
buffer with `newBufferWithBytesNoCopy`, so the GPU reads the weights
from the page cache in place. The buffer's deallocator holds an `Arc` of the mapping,
so the mapping lives until the last command buffer that reads it
finishes. Each tensor is a `View`, an offset into that buffer. A file
that the system refused to map arrives in heap memory and is copied
into a new buffer.

Metal's API is Objective-C, reached through the `objc2-metal` crate,
and every call into it is unsafe Rust. `gip-metal` keeps the unsafe
code behind a safe API. Every launch checks that each buffer range it
binds lies inside its buffer, and a buffer from another `Metal` is
rejected. The CPU reads and writes shared buffers only while no command
buffer is in flight.

## One decode step

`Lfm2Metal::step` writes the token id into a buffer and records the
whole step into one command buffer. Host time spent recording is part
of every token, so one command buffer per step keeps that cost small.

1. `embed` dequantizes the token's row of the embedding matrix into the
   residual stream.
2. A convolution layer runs its input projection with the layer's RMS
   normalization fused into the matvec. `short_conv` then updates the
   convolution history in place. The output projection adds its result
   straight into the residual stream.
3. An attention layer runs the q, k, and v projections with the
   normalization fused. `norm_rope` normalizes and rotates q in place
   and writes the rotated k straight into the KV cache.
   `attention_chunk` computes partial attention over chunks of 64
   cached positions in parallel, and `attention_combine` merges the
   chunks. The output projection adds into the residual stream.
4. `matvec_swiglu` reads the gate and up rows together with the
   feed-forward normalization fused, and `ffn_down` adds its result
   into the residual stream.
5. The output matvec, with the final normalization fused, writes the
   logits.

The encoder is concurrent, so launches may overlap on the GPU. A
memory barrier separates each launch from the launches it depends on.
The q, k, and v projections run at the same time, for example.

Fusion saves memory traffic. A fused normalization computes the input's
root mean square inside the matvec, so the normalized vector is never
written to memory and read back. An accumulating matvec adds into the
residual stream, so the block output needs no separate add kernel.

`gip respond` and `gip chat` keep the KV cache in half precision,
which halves the bytes each attention step reads. The reference tests allow a larger
error for the half-precision cache, as [LFM2](lfm2.md) describes.

## Pipelined greedy decoding

`Lfm2Metal::generate` decodes greedily without waiting on the CPU. Each
step's command buffer starts with an `argmax` kernel that writes the
most likely token into the token buffer, and the step's `embed` reads
that token. The CPU never has to see a token before recording the next
step, so it keeps up to three command buffers in flight and reads all
the generated ids at the end.

Sampling with a temperature needs the logits on the CPU, so the
sampled path calls `step` once per token and waits for it.

## Prefill

`Lfm2Metal::prefill` runs the prompt in batches of up to 512 tokens.
Each matrix is multiplied by the whole batch with a `matmul` kernel.

A `matmul` threadgroup of four simdgroups computes a tile of 64 matrix
rows by 32 tokens. The kernel dequantizes a slice of weights to half
precision in threadgroup memory. The simdgroups multiply the slice with
`simdgroup_multiply_accumulate`, which runs on the GPU's 8 by 8 matrix
units. The feed-forward block runs the gate matmul, then an up matmul
that applies SwiGLU as it stores, then a down matmul that accumulates
into the residual stream.

The convolution layers run `short_conv_batch` over every token and
channel at once, then `short_conv_history` saves the last tokens for
the next call. Attention covers all queries of the batch. Only the
batch's last token goes through the output matrix, since only its
logits choose the next token. Batches run back to back, and the CPU
waits only for the last one.

## State on the GPU

`Lfm2Metal::new` allocates every buffer once, so a step creates no
Metal objects other than its command buffer.

| Buffer | Holds |
|---|---|
| `k_cache`, `v_cache` | Keys and values of every attention layer, by layer, then position, then head |
| `conv_state` | The last `conv_kernel - 1` inputs of every convolution layer |
| `scores` | Partial attention results for each query, head, and chunk of positions |
| Scratch buffers | The residual stream, normalized inputs, and projections for one batch |
| `logits` | One score per vocabulary token |
| `tokens` | Token ids, written by the CPU or by `argmax` |

## Profiling

`gip-bench` measures prefill and decode speed and reports the time the
CPU spends recording against the time the GPU spends running. With
`--profile`, `Metal` runs each launch in its own command buffer and
records its time under the kernel's name and shape. The profile shows
where a token's time goes.
