# bobcat design

### Goals

bobcat is an inference engine for Apple silicon Macs. bobcat aims to
run every popular local model that fits in a Mac's memory, as fast as
possible and with as little energy as possible, on any Mac from a
MacBook Air to a Mac Studio.

New architectures appear every few months, so adding one must be cheap.
bobcat builds each model from shared operations, such as quantized
matrix products, attention, and normalization. A new architecture needs
only its graph and the operations it adds.

### Performance model

Local inference decodes one stream, or a few when several agents share
a model. Each decode step reads most of the active weights from memory
and does little arithmetic per weight. GPU read bandwidth divided by
the active weight bytes therefore gives the decode ceiling, the
fastest rate that streaming the weights allows. Cache traffic and the
fixed cost of each step keep real decoding below the ceiling. The
fixed cost matters most for small models, because a small model reads
few bytes per token.

Prefill processes the whole prompt as a batch. Each weight read serves
many tokens, so prefill can become bound by arithmetic. A kernel that
reloads weights for every small tile still stays bound by memory.

The design follows from this model. bobcat reduces the bytes read per
token, removes the fixed cost per step, and raises weight reuse in
prefill. Time to first token, time per output token, throughput, and
energy per token all guide the design. Profiling identifies the
limiting cost before each optimization.

### Unified memory

Every Apple silicon Mac puts the CPU, the GPU, and the Neural Engine on
one chip with one pool of memory. The design relies on three
consequences.

1. **Shared weight storage.** The GPU reads weights directly from the
   memory-mapped model file. A repacked layout needs its own
   allocation, made once at load time.
2. **Shared CPU and GPU buffers.** The CPU and the GPU exchange data
   through the same allocation, with no copies. Synchronization still
   costs time. The Neural Engine is reachable only through Core ML,
   which sets its own storage and scheduling rules.
3. **Large models fit.** A model can use most of the machine's memory,
   which on many Macs exceeds the memory of a consumer GPU.

Macs differ in bandwidth, GPU size, and memory capacity. bobcat relies
on the properties that every Mac shares and chooses chip-dependent
kernel layouts by measurement on the machine itself.

### Memory traffic

Each technique in this section raises the number of tokens that bobcat
generates per byte of weights read.

- **Quantization and weight layout.** A 4-bit weight takes a quarter of
  the bytes of a 16-bit weight, plus a share of its block's scales.
  bobcat reads quantized weights directly. bobcat can also repack them
  once at load time into the order in which the GPU reads them, and
  cache the repacked file. A repacked layout earns its storage only
  through less traffic or cheaper unpacking.
- **Batching.** When several agents share a model, bobcat decodes their
  streams together. One read of each weight serves every stream in the
  batch. The scheduler bounds each stream's latency.
- **Speculative decoding.** A small draft model proposes several
  tokens. The target verifies them in one batched pass, so the accepted
  tokens share the target's weight reads. Exact acceptance preserves
  the target's sampling distribution. The draft's cost and its
  acceptance rate set the gain. The target discards rejected tokens by
  truncating its key-value cache. Convolution and state-space layers
  also restore their state to the last accepted token, from a
  checkpoint or by recomputation.
- **Mixture of experts.** A mixture-of-experts layer sends each token to
  a few experts, so bobcat reads only their weights. Unified memory
  holds every expert. Routing runs on the GPU, next to the matrix
  kernels. Tokens in one batch can choose different experts, which
  lowers weight reuse.

### Per-step overhead

A step's fixed cost has three parts: encoding commands on the CPU,
dispatching kernels on the GPU, and synchronizing the two. CPU encoding
already overlaps GPU work, so profiling of GPU dispatch and stalls
guides this section.

- **The GPU runs the whole step.** The GPU chooses each token and feeds
  it to the next step. The CPU submits several steps ahead and
  synchronizes only to stream text or to stop.
- **Stable commands are recorded once.** Most of a decode step repeats
  for every token. Attention length and cache offsets change as the
  context grows, so the engine passes them as data. Indirect command
  buffers let the engine record the stable dispatches once and replay
  them.
- **Kernels fuse where fusion pays.** Combining small operations saves
  dispatches and intermediate traffic. Fusion can also raise register
  pressure and lower occupancy, so measured latency decides which
  fusions bobcat keeps.
- **Decoding allocates nothing.** bobcat allocates every cache and
  buffer at load time.

### Metal 4

Metal 4 runs on every Apple silicon Mac and gives the engine more
control over the GPU. Explicit command allocators let bobcat reuse one
command allocation for every step. Argument tables replace per-dispatch
buffer binding. Residency sets keep the weights and caches resident for
the life of the model. Allocation reuse and command replay are
separate mechanisms. bobcat uses each one where the machine supports
it. Metal 4 also adds tensor operations, which the next
section covers.

### Prefill

Prefill runs the prompt through matrix kernels in large batches. Weight
reuse, the cost of dequantization, and accumulation accuracy shape the
kernels. Metal 4 tensor operations are a candidate path. According to
[Apple's documentation](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4),
tensor operations run on neural accelerators in newer GPUs and on the
shader cores of earlier ones, so the speedup on each chip needs
measurement. Native low-bit tensor formats set their own layout and
scaling rules, so GGUF blocks may need repacking before a tensor
operation reads them. The current SIMD-group matrix path remains for
chips where tensor operations lose.

### CPU and Neural Engine

The CPU encodes commands and converts tokens to text while the GPU
decodes. Sampling stays on the GPU whenever the next step consumes its
result. CPU drafting is an experiment, because the CPU and the GPU
compete for the same memory bandwidth.

[Core ML](https://developer.apple.com/documentation/coreml/mlcomputeunits)
is the supported route to the Neural Engine. Model conversion,
operation placement, and state management constrain that route. A
small speculative draft is the first candidate, because it leaves the
GPU free to verify. Drafting and verification depend on each other, so
running them concurrently needs explicit scheduling. bobcat judges
Neural Engine drafting and prefill by end-to-end latency and energy
per token.

### Correctness

A scalar implementation of each operation defines the correct output.
bobcat checks every GPU kernel against the scalar version and every
model graph against the reference implementation in Hugging Face
transformers. The checks guard new architectures and kernel
optimizations. Lower-precision arithmetic needs explicit error bounds
and full-model validation. Speculative decoding needs tests of
acceptance and state restoration.

### Related work

**Engines for Apple silicon.**
[BaseRT](https://arxiv.org/abs/2607.00501) builds native Metal kernels
for unified-memory inference, and
[its follow-up](https://arxiv.org/abs/2607.19438) adds Metal 4 tensor
kernels on the M5 Pro.
[MetalRT](https://huggingface.co/blog/runanywhere/metalrt-fastest-inference-apple-silicon)
is another native Metal inference engine.

**Speculative decoding.**
[Leviathan et al.](https://arxiv.org/abs/2211.17192) introduced
speculative decoding at ICML 2023.
[Medusa](https://proceedings.mlr.press/v235/cai24b.html) and
[EAGLE](https://proceedings.mlr.press/v235/li24bt.html), both at ICML
2024, draft from the target model's own features.
[The Mamba in the Llama](https://arxiv.org/abs/2408.15237), at NeurIPS
2024, verifies drafts for Mamba and hybrid models on NVIDIA GPUs.

**Heterogeneous on-device inference.**
[llm.npu](https://dl.acm.org/doi/10.1145/3669940.3707239), at ASPLOS
2025, offloads prefill to a phone's NPU.
[LLM in a Flash](https://aclanthology.org/2024.acl-long.678/), at ACL
2024, streams weights from flash storage on Apple devices.

**Quantized kernels.**
[Marlin](https://dblp.org/rec/conf/ppopp/FrantarCCHA25.html), at PPoPP
2025, and [QServe](https://arxiv.org/abs/2405.04532), at MLSys 2025,
co-design weight layouts and GPU kernels on NVIDIA hardware.

### Non-goals

bobcat does not serve requests across machines or train models. Metal
is the execution backend. Core ML serves only experimental Neural
Engine work.
