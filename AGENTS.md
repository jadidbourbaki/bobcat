# AGENTS.md

Guidance for AI agents working on the gip codebase. The CLAUDE.md
symlink resolves to this file. Read top to bottom on first session.

## Project context

gip, the general inference program, is a local LLM inference engine
written in Rust. The first goal is the fastest LLM inference on the Mac
GPU, measured against mlx-lm and llama.cpp's Metal backend. The first
models are Liquid AI's LFM2.5 family at Q8_0, and the first machine is
an Apple M4 Pro with a 20-core GPU. LFM2.5-1.2B and LFM2.5-2.6B carry
the benchmark headline. LFM2.5-350M keeps the tests fast and will serve
as the draft model for speculative decoding of the 1.2B model, whose
vocabulary it shares.

A scalar Rust implementation of every op defines the correct output and
checks every GPU kernel. The GPU kernels are Metal Shading Language
files that the engine embeds and compiles at load time. SIMD CPU kernels
for phones and Linux servers come later behind the same Rust API.

The engine reads GGUF files directly, so every quantized model already
published on Hugging Face runs without conversion. The engine is a
library first. Command-line tools and any future server call the
library through its public Rust API. A C API crate generated with
cbindgen comes later, so C, Swift, and other languages can link the
engine.

## Quality gate

Before declaring any change complete, run:

```bash
just check
```

`just check` checks formatting, runs clippy with warnings as errors,
builds the documentation with warnings as errors, runs every test, and
checks that each `Cargo.toml` lists its dependencies in sorted order. A
change that fails `just check` locally is unfinished.

Cargo builds gip, and [just](https://just.systems/) holds the everyday
commands.

| Recipe | Meaning |
|---|---|
| `just` | List the recipes |
| `just build` | Build every crate and tool |
| `just test` | Run the tests |
| `just check` | Run the full quality gate |
| `just fmt` | Format the Rust sources, the manifests, and the Metal kernels |
| `just bench` | Measure GPU bandwidth, gip, and the baseline engines |
| `just bench-cpu` | Measure CPU bandwidth and the llama.cpp CPU baseline |
| `just clean` | Remove the build output |

The clippy invocation matches the one Astral's ruff and uv use:

```bash
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
```

## Toolchain

`rust-toolchain.toml` pins the Rust release and its `clippy` and
`rustfmt` components, so every machine lints and formats the same way.
rustup reads the file and installs the pinned release on first use.
Moving to a newer release is a commit of its own that fixes every new
lint.

Homebrew's `rust` formula installs its own `cargo` in
`/opt/homebrew/bin`, which ignores `rust-toolchain.toml`. Put
`~/.cargo/bin` ahead of `/opt/homebrew/bin` in `PATH`, or uninstall the
formula, so rustup's `cargo` runs.

The Command Line Tools include no Metal shader compiler. gip embeds its
`.metal` sources in the library with `include_str!` and compiles them at
load time with `newLibraryWithSource`, so building gip needs no Xcode.
macOS caches the compiled shaders between runs.

`clang-format` from the Command Line Tools formats the Metal kernels.
Homebrew's LLVM at `/opt/homebrew/opt/llvm` provides `clang-format` on
machines without the Command Line Tools.

## Repository layout

```
Cargo.toml                 workspace members, shared dependencies, lints, and profiles
Cargo.lock                 exact dependency versions, committed
rust-toolchain.toml        pinned Rust release
rustfmt.toml, clippy.toml  formatter and linter settings
justfile                   everyday commands
crates/
  gip-gguf/                GGUF parser in safe Rust
  gip-metal/               Metal host code through objc2-metal
    src/kernels.metal      Metal kernels, embedded at build time
  gip/                     model loading, scalar reference ops, and the LFM2 graph
    src/scalar.rs          reference implementation of every op
    src/lfm2.rs            LFM2 on the CPU with the scalar ops
    src/lfm2_metal.rs      LFM2 on the Metal GPU
    tests/                 reference and kernel tests
  gip-bench/               gip-bench, gpu-bw, and cpu-bw programs
tools/                     Python reference dumps and benchmark scripts
bench/                     baseline engine checkouts, gitignored
models/                    downloaded GGUF files, gitignored
```

## Writing prose

The following style rules apply to all prose in the repo: README,
docs, design notes, commit message bodies, and code comments. They
come from the maintainer's stated preference and must be honored.

### Hard rules

- **No em-dashes.** The character `—` does not appear in prose. Split
  the sentence in two or use a comma. The same goes for en-dashes
  (`–`) in prose.
- **No semicolons in prose.** Use a period and start a new sentence.
- **No unnecessary parentheses.** A parenthetical aside that pauses
  the reader belongs in its own sentence. A one-word gloss such as an
  abbreviation on first use is fine. Parentheses never substitute for
  a comma or a period.
- **Don't pack settings into parentheses.** Put flags, environment
  variables, and their defaults in a table with a column for the
  name, the default, and the meaning.
- **No ASCII diagrams.** Describe relationships in prose. A directory
  listing is fine.
- **No emoji** unless the user explicitly asks for them.
- **Don't define a thing by what it is not.** "X, not Y" and its
  variants, "X rather than Y" and "X and never Y", tell the reader
  what to erase instead of what to hold. Say what the thing does and
  stop. "The scalar kernel is a reference, not a fast path" becomes
  "The scalar kernel defines the correct output for every op." State
  a limit outright when it is load-bearing, as its own sentence, in
  terms of what happens: "The decode loop makes no allocation."
- **No vague back-references.** Do not open a sentence with "This",
  "That", "These", "Those", "Their", or "It" pointing at a noun from
  an earlier sentence. Name the noun again. The reader should never
  have to look backward to resolve a pronoun.

### Soft rules

- Write short, direct sentences. If a sentence has more than one
  comma, consider whether it should be two sentences.
- Lead with the noun. "The loader validates every offset" beats
  "When a file opens, the loader validates every offset."
- Define jargon on first use, even if you think the reader knows it.
- Do not write in fragments or in a punchy, aphoristic style. "No
  malloc, no syscalls, no surprises" is wrong. "The decode loop reads
  preallocated buffers, so it never allocates" is right.

### Examples

Wrong:

> The loader memory-maps the file (falling back to read on failure);
> tensors then point into the mapping.

Right:

> The loader memory-maps the file. If mmap fails, the loader reads the
> file into a heap buffer. Tensors point into whichever buffer holds
> the file.

Wrong:

> A Q8_0 block holds 32 weights, the scale is stored as fp16.

Right:

> A Q8_0 block holds 32 int8 weights and one fp16 scale.

## Rust style

gip follows the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
and the conventions of Astral's [uv](https://github.com/astral-sh/uv)
and [ruff](https://github.com/astral-sh/ruff). Where this document is
silent, those sources decide. Where this document and those sources
disagree, this document wins.

### Edition and formatting

Every crate uses the 2024 edition. `rustfmt.toml` sets
`style_edition = "2024"`, and `cargo fmt` decides all layout. Wrap
comments and doc comments at 100 columns, the same width rustfmt uses
for code.

Imports go at the top of the file, in three groups separated by blank
lines: `std`, external crates, then `crate` and `super`. Import inside
a function body only to keep a platform-specific name out of other
platforms' builds. Glob imports appear only as `use super::*` in a test
module.

### Lints

`Cargo.toml` declares the lints once in `[workspace.lints]`, and every
crate opts in with `[lints] workspace = true`. The set follows uv's:

- `clippy::pedantic` at warn, with a few lints allowed where they fight
  kernel code or add noise. Each allowance carries a comment giving the
  reason.
- `unreachable_pub` at warn, so public items are exactly the crate's
  API.
- `unsafe_code` at deny. The unsafe code policy below names the modules
  that may opt out.
- Restriction lints at warn: `allow_attributes`,
  `allow_attributes_without_reason`, `dbg_macro`, `exit`, `get_unwrap`,
  `print_stdout`, `print_stderr`, `todo`, `unimplemented`,
  `unwrap_used`, `undocumented_unsafe_blocks`, and
  `multiple_unsafe_ops_per_block`.

Silence a lint with `#[expect(lint, reason = "...")]` on the smallest
item that needs it. `#[allow]` is itself a lint error, because an
`#[expect]` warns once the code no longer triggers the lint.

Never assume a warning predates your change. Fix every warning that
`just check` reports.

### Keep the codebase small

Every line of code is a line someone maintains. Add code only when a
caller needs it now.

- Add a public item only when another crate or a test calls it. Delete
  an item once its last caller goes.
- Write one helper for a sequence of launches that two paths share.
  Keep a single-use computation inline.
- Skip speculative options, builders, and trait abstractions until a
  second implementation exists.
- Prefer the standard library and the crates already in the workspace
  over new code or new dependencies.
- Before proposing a change, reread the diff and remove anything the
  change does not need.

### Constructs

- Match exhaustively on enums gip defines. A wildcard arm hides the
  next variant someone adds.
- Use let chains, `let ... else`, and `?` to keep the happy path at the
  left margin.
- Keep each struct next to its `impl` blocks.
- Declare items with the narrowest visibility that works. Prefer
  `pub(crate)` for items shared inside a crate.
- Take slices and return owned values. `&[f32]` and `&mut [f32]` carry
  the length that a C pointer leaves implicit, so kernels index within
  bounds the caller established.
- Encode invariants in types. A GGUF tensor type is an enum, a token id
  is a `u32`, and a `Result` carries every failure. Reach for `expect`
  only for an invariant the surrounding code has already established,
  and let the message name the invariant.
- Convert between integer types with `From` and `TryFrom`. Use `as` only
  for a conversion that cannot lose information on every supported
  target, such as `u32` to `usize` on 64-bit hosts, or for float
  conversions that kernel math intends.

### Naming

Follow [RFC 430](https://rust-lang.github.io/rfcs/0430-finalizing-naming-conventions.html)
casing. Give items descriptive names. The accepted domain abbreviations
are `kv`, `qk`, `ffn`, `rms`, `rope`, `gguf`, `simd`, quant format
names such as `q8_0`, and the `n_` prefix for counts. Getters take the
field's name with no `get_` prefix. Conversions follow the `as_`,
`to_`, and `into_` conventions of the API guidelines.

### Comments and documentation

- Start every file with a `//!` comment that says what the module is
  for. A binary's file says what the program does.
- Give every public item a `///` comment. The first sentence says what
  the item does. Later sentences explain the arguments and the result
  when the signature leaves them unclear. Link other items as
  `[TypeName]` or `[function_name]`.
- Write comments in complete sentences that start with a capital
  letter.
- Inside function bodies, comment only the non-obvious why: a hidden
  constraint, an invariant, a numeric tolerance, or a workaround for a
  specific compiler or upstream bug. Skip comments that describe what
  the next line does.
- Describe the present code. Skip history such as "renamed from",
  conversation context, and references to callers, issues, or pull
  requests. `git log` holds the history.
- Skip banners and decorative separators.

### Errors

- Each library crate defines its error enum with `thiserror`. Variants
  carry the values a reader needs, such as a tensor name and its
  shape.
- Error messages start lower case and end with no period, as the API
  guidelines ask. Tools print errors as `PROGRAM: message` and follow
  the `source` chain.
- The library returns an error on bad input. The library never prints,
  exits, or panics on bad input, because it runs inside other people's
  processes.
- The library keeps no mutable global state.
- An impossible condition means a bug in gip. Use `unreachable!` with a
  message that explains why the condition cannot happen.
- Tools parse options with `clap` and support `--help` and `--version`.
  Exit status is 0 on success and 1 on failure.

### Dependencies

- Declare every dependency once in `[workspace.dependencies]` with its
  version, and refer to it from each crate with `workspace = true`.
- Turn off default features where the crate allows it, and list only
  the features gip uses.
- Commit `Cargo.lock`. Change a locked version with
  `cargo update --precise`, one dependency at a time.
- Before adding a dependency, check that it is maintained and widely
  used. State in the commit message what it replaces.

## Unsafe code

`unsafe_code` is denied across the workspace. `gip-gguf` goes further
with `#![forbid(unsafe_code)]`, so the parser that reads untrusted files
holds no unsafe code at all. Three places opt out with
`#[expect(unsafe_code, reason = "...")]`:

| Module | Why it needs unsafe |
|---|---|
| `gip-metal` | Metal's API is Objective-C, reached through `objc2-metal`. The crate also holds the GPU bandwidth probe that `gpu-bw` runs |
| `gip::storage` | `memmap2` maps model files |
| `cpu-bw` | macOS sets a thread's quality of service class through `libc` |

Adding unsafe code anywhere else needs the user's approval first.

Rules for unsafe code:

- Every `unsafe` block holds one unsafe operation and follows a
  `// SAFETY:` comment that says why each condition of that operation's
  contract holds.
- Every `unsafe fn` has a `# Safety` section in its documentation that
  lists the caller's obligations.
- A safe wrapper checks what the Objective-C API assumes. Every kernel
  launch checks that each bound range lies inside its buffer. CPU reads
  and writes of a shared buffer happen only while no command buffer is
  in flight.
- The GPU reads weights from the memory-mapped file through
  `newBufferWithBytesNoCopy`. The buffer's deallocator block holds an
  `Arc` of the mapping, so the mapping outlives every command buffer
  that reads it.

## Untrusted input

Users download GGUF files from strangers, so every byte of a model file
is untrusted. llama.cpp has shipped memory-safety bugs in GGUF parsing.
Safe Rust closes off that class of bug, and the rules below close off
the denial-of-service bugs that remain.

- Validate every count, offset, dimension, and string length against
  the file size before using it.
- Compute every size with `checked_add` and `checked_mul`, and reject
  the file on overflow.
- Read multi-byte fields with `from_le_bytes`. GGUF is little-endian.
- A malformed file yields an error. `gip-gguf` warns on
  `clippy::indexing_slicing` and `clippy::arithmetic_side_effects`, so
  no input can reach a panic through an index or an overflow.
- Reserve memory from a count in the file only after the count passes
  the file-size check.
- Keep all parsing in `gip-gguf`. Every change to that crate runs the
  malformed-file tests and, once it exists, `just fuzz`.

## Memory and threads

- The engine allocates everything at load time: weight mappings, the
  KV cache, convolution state, and scratch buffers. The decode loop
  makes no allocation and no system call other than waiting on the
  GPU.
- The loader tries `mmap` on each file and falls back to reading the
  file into memory when `mmap` fails for that file.
- The Metal backend wraps the memory-mapped GGUF file in one
  `MTLBuffer`, so the GPU reads weights from the page cache with no
  copy. The mapping is page-aligned, and each tensor is an offset into
  that buffer.
- The KV cache, convolution state, and scratch space live in
  `MTLBuffer`s created at load time. A decode step creates no Metal
  objects other than its command buffer.
- The CPU thread pool starts at load time and keeps its threads alive.
  Workers spin on barriers between ops, because waking a parked thread
  costs microseconds and a decode step runs hundreds of ops. Decode
  threads run on performance cores. Linux pins them with
  `sched_setaffinity`. macOS offers no pinning, so workers set
  `QOS_CLASS_USER_INTERACTIVE`.

## Kernels

- Every op has a scalar implementation in `crates/gip/src/scalar.rs`.
  The scalar version defines the correct output.
- Every Metal kernel and every SIMD kernel is tested against the
  scalar version on random inputs, with the tolerance stated in the
  test.

Metal kernels:

- Metal Shading Language code follows the GNU C formatting style in
  `.clang-format`, with a column limit of 79. `just fmt` applies it.
- Create every compute pipeline state at load time. Specialize kernels
  for fixed shapes such as head size with Metal function constants.
- Encode each decode step into one command buffer. Host encoding time
  is part of every token, so measure it along with GPU time.
- Keep threadgroup sizes and tile sizes in named constants with a
  comment giving the measurement that chose them. A constant shared by
  the host and a kernel appears in both files with a comment naming
  the other.

CPU kernels, which come after the Metal backend:

- The loader detects CPU features once and picks a function for each
  op. Hot loops call through that choice once per op.
- Prefer `std::arch` intrinsics. Use inline assembly only for an
  instruction the intrinsics cannot express, and say which instruction
  in a comment.
- Apple's M4 has SME and lacks SVE outside streaming mode. SVE code
  outside a streaming function crashes with `SIGILL` on the M4.

## Build profiles

| Profile | Settings | Use |
|---|---|---|
| `dev` | defaults | everyday builds |
| `test` | `opt-level = 2` | the scalar reference pass runs a whole model |
| `release` | `lto = "fat"`, `codegen-units = 1`, `overflow-checks = true` | benchmarks and shipping builds |

Release builds keep integer overflow checks on, because host code takes
a small share of each token. Turning them off needs `just bench`
numbers that show their cost.

## Writing tests

Write only tests that catch a real regression. A test that checks a
constant or restates the implementation is noise.

The tests that matter:

- Each Metal kernel and each SIMD kernel against the scalar kernel.
- Hidden states after every layer against dumps from `transformers`,
  produced by `tools/ref_dump.py`.
- Greedy decoding that matches `transformers` token for token.
- The parser rejecting malformed files, from crafted cases and the
  fuzz corpus.

Unit tests live in a `#[cfg(test)] mod tests` at the bottom of their
file. Tests that load a model or open the GPU live in each crate's
`tests/` directory. Name a test for the behavior it checks, with no
`test_` prefix. A test that needs a file from `models/` or a Metal
device prints `skip:` and the reason to standard error and returns
when the file or device is missing. Tests return `Result` and use `?`
for setup that can fail. gip uses no test framework beyond the
standard library.

## Benchmarks

The headline claim is the fastest LLM inference on a Mac. The engines
to beat are mlx-lm and llama.cpp's Metal backend. mistral.rs and
candle, the leading Rust engines, run the same GGUF files and serve as
further baselines. Every performance claim comes with numbers from this
protocol:

- Use the same GGUF file for gip, llama.cpp, mistral.rs, and candle.
- Run mlx-lm on Liquid's official MLX 8-bit weights. Report the model
  bytes of each engine's weights, since MLX's 8-bit format differs
  from GGUF Q8_0.
- Record each baseline's commit hash or version and its build flags.
- Measure prefill with a 512-token prompt and decode with 128
  generated tokens.
- Run each setting 5 times and report the mean and standard
  deviation.
- Report the decode ceiling: measured GPU read bandwidth divided by
  model bytes.
- Run on an idle machine with no build in progress, on AC power.
  Spotlight indexing and video calls distort the numbers, so exclude
  `models/` from Spotlight and close other applications first.

CPU benchmarks follow the same protocol against llama.cpp's CPU build
and sweep 1, 4, 8, and 10 threads on performance cores. `cpu-bw` gives
the CPU decode ceiling.

A change that slows any benchmark gets reported with before and after
numbers when it is proposed for commit.

## Adding a model architecture

1. Read the architecture's modeling file in `transformers` and its
   graph in llama.cpp. Write down every op in one layer, in order.
2. Dump reference token ids, per-layer hidden states, and logits with
   `tools/ref_dump.py`.
3. Write the scalar forward pass. Compare against the dump layer by
   layer and stop at the first layer that diverges.
4. Match greedy decoding token for token.
5. Only then write Metal kernels, SIMD kernels, or fused paths, each
   tested against the scalar pass.

## Python tools

Python appears only in `tools/`, for reference dumps and benchmark
scripts. The engine never depends on Python.

- **uv** manages environments and dependencies. Pin every direct
  dependency with `==` and commit `uv.lock`.
- **ruff** lints and formats. **ty** type-checks.
- Type every function signature.
- Raise exceptions on failure and catch narrowly. A script that hits
  a bad state stops with a traceback.

## Don't reinvent the wheel

The kernels, model graphs, KV cache, and scheduler are the product,
and gip writes them. Everything around them gets the usual scrutiny.
Before writing any other non-trivial logic, search the standard library
first, then the crates already in `[workspace.dependencies]`, then
established crates on crates.io.

Before writing a kernel or a model graph, read how the reference
engines below do the same job. Those projects have already found the
tricks worth copying and the traps worth avoiding.

| Engine | What to read it for |
|---|---|
| [MLX](https://github.com/ml-explore/mlx) | Metal kernels for quantized matmul, attention, and normalization in `mlx/backend/metal/kernels` |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) and ggml | GGUF, the Metal backend in `ggml/src/ggml-metal`, quantized CPU kernels, weight repacking, and the LFM2 graph |
| [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | CPU kernels from the author of ggml's K-quants and I-quants |
| [KleidiAI](https://github.com/ARM-software/kleidiai) | Arm's micro-kernels for `dotprod`, `i8mm`, and SME |
| [llamafile](https://github.com/mozilla-ai/llamafile) | tinyBLAS register tiling for prefill |
| [XNNPACK](https://github.com/google/XNNPACK) | Google's ARM and x86 CPU kernels |
| [MNN](https://github.com/alibaba/MNN) | On-device LLM inference on ARM phones |
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | Compact model definitions, KV cache design, and the generation loop to beat |
| [mistral.rs](https://github.com/EricLBuehler/mistral.rs) | Paged attention, speculative decoding, and a Rust engine API |
| [candle](https://github.com/huggingface/candle) | Rust Metal host code through `objc2-metal` and a second implementation of ggml's quant block layouts |
| [Cactus](https://github.com/cactus-compute/cactus) | A C API over a C++ core, built for phones |
| [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) and [ExecuTorch](https://github.com/pytorch/executorch) | On-device runtime structure |
| [tinygrad](https://github.com/tinygrad/tinygrad) | Compiler-generated kernels and a compact GGUF loader in `tinygrad/llm` |
| [transformers](https://github.com/huggingface/transformers) | The correct output of every model |

## Commit messages

[Conventional commits](https://www.conventionalcommits.org/en/v1.0.0/),
one sentence each, with no body unless absolutely necessary.

```
feat: add Q8_0 NEON matrix-vector kernel
fix: reject GGUF tensors whose offset exceeds the file size
perf: fuse RMSNorm into the following projection
test: compare LFM2 hidden states against transformers dumps
docs: document the benchmark protocol
```

Rules:

- One sentence subject in present-tense imperative.
- Lowercase the type and the first word after the colon, unless that
  word is a proper noun or acronym.
- No commit body unless the change is non-obvious and the reason
  cannot fit in the subject. Skip "Test plan" and "Summary"
  boilerplate.
- **No `Co-Authored-By: Claude` trailer.** Ever. The rule holds even
  when the user has authorized commits in advance.
- Do not amend or rewrite published commits without explicit user
  consent.

## Git practices

- **Never commit without asking. Never push without asking.** Each act
  needs its own approval. Propose the commit, name the files and the
  message, then wait for the user to say yes. The user verifies the
  results before saying yes, so leave the work uncommitted until then.
- **Approval never carries forward.** A yes for one commit covers only
  that commit, and a yes to commit leaves pushing unapproved. Ask
  again every time.
- **Approval to do work leaves committing unapproved.** "Sounds good",
  "go ahead", and "yes" in reply to a plan mean write the code. When
  the user approves an implementation, finish it, run `just check`,
  and stop with a summary and a commit proposal.
- Use whatever git identity the user has configured. Never pass
  `-c user.email` or `-c user.name`.
- Before any destructive operation such as `git reset --hard`, a
  force-push, or a branch delete, confirm with the user. Force-push
  only with `--force-with-lease`.

## Working with the user

### Risk and reversibility

Local, reversible actions such as editing a file or running a test
need no preamble. Hard-to-reverse actions such as a force-push or
deleting files the user wrote need explicit confirmation each time.
Authorization for one action covers only that action.

### Confirmation patterns

- Lay out a plan before destructive multi-step work. Get a green
  light, then execute.
- When you spot a side-effect the user didn't ask for, such as a
  cleanup, a refactor, or a lint fix, name it and ask before doing it.

### Communication style

- Default to terse. The user reads diffs.
- Lead with the result, then the details if asked.
- End a turn with one or two sentences on what changed and what is
  next.
- Don't restate the user's request. Just answer.

### Scope

Match the scope of a change to what the user asked. A bug fix gets no
free refactor of the surrounding code. A side-improvement of one line
with no behavior change needs no ceremony. Anything larger goes to the
user as a separate option.

## When in doubt

Re-read this document, then the most recent code that touched the same
area. The patterns are meant to stay consistent across the codebase,
so match the existing pattern.
