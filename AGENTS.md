# AGENTS.md

Guidance for AI agents working on the gip codebase. The CLAUDE.md
symlink resolves to this file. Read top to bottom on first session.

## Project context

gip, the general inference program, is a local LLM inference engine
written in C. The first goal is the fastest LLM inference on the Mac
GPU, measured against mlx-lm and llama.cpp's Metal backend. The first
models are Liquid AI's LFM2.5 family at Q8_0, and the first machine is
an Apple M4 Pro with a 20-core GPU. LFM2.5-1.2B and LFM2.5-2.6B carry
the benchmark headline. LFM2.5-350M keeps the tests fast and will serve
as the draft model for speculative decoding of the 1.2B model, whose
vocabulary it shares.

A scalar C implementation of every op defines the correct output and
checks every GPU kernel. SIMD CPU kernels for phones and Linux servers
come later behind the same C API.

The engine reads GGUF files directly, so every quantized model already
published on Hugging Face runs without conversion. The engine is a
library first. Command-line tools and any future server link the
library through its public header.

## Quality gate

Before declaring any change complete, run:

```bash
just check
```

`just check` builds everything with warnings as errors, runs the test
programs, builds and runs them again under AddressSanitizer and
UndefinedBehaviorSanitizer, and checks formatting with `clang-format`.
A change that fails `just check` locally is unfinished.

[Meson](https://mesonbuild.com/) builds gip, and
[just](https://just.systems/) holds the everyday commands. The
`justfile` keeps three meson build directories under `build/`.

| Directory | Configuration |
|---|---|
| `build/default` | debug info with `-O2`, warnings as errors |
| `build/sanitize` | the default build plus ASan and UBSan |
| `build/release` | `-O3` for benchmarks |

| Recipe | Meaning |
|---|---|
| `just` | List the recipes |
| `just build` | Build the library and tools |
| `just test` | Run the tests |
| `just check` | Run the full quality gate |
| `just fmt` | Rewrite the sources in GNU style |
| `just bench` | Measure GPU bandwidth and the llama.cpp Metal and mlx-lm baselines |
| `just bench-cpu` | Measure CPU bandwidth and the llama.cpp CPU baseline |
| `just clean` | Remove the build directories |

`meson install` honors `--prefix` and `DESTDIR`, as the
[GNU Makefile Conventions](https://www.gnu.org/prep/standards/html_node/Makefile-Conventions.html)
ask of an install step. A `just fuzz` recipe arrives with the first
libFuzzer harness.

## Toolchain

The supported compilers are GCC 13 or newer and Clang 17 or newer.
Clang 17 is the first Apple clang with SME support, and it ships with
Xcode 26 and its Command Line Tools. The Metal backend builds with
Apple clang only, because its host code is Objective-C.

The Command Line Tools include no Metal shader compiler. gip embeds
its `.metal` sources in the library and compiles them at load time
with `newLibraryWithSource`, so building gip needs no Xcode. macOS
caches the compiled shaders between runs.

Apple clang 16 from the older Command Line Tools lacks the libFuzzer
runtime. Its AddressSanitizer also crashed at startup on the
maintainer's macOS 26 machine, while its UBSan worked. On macOS the
`justfile` therefore builds `build/sanitize` with Homebrew's LLVM at
`/opt/homebrew/opt/llvm`, passing the installed SDK with `-isysroot`.
Install it with `brew install llvm`. Homebrew's LLVM also includes
libFuzzer. Other systems use their default compiler for every build.

## Repository layout

```
meson.build           project options and warning flags
justfile              everyday commands
.clang-format         GNU style settings
include/gip.h         public C API and the only installed header
src/                  library sources
  gguf.c              GGUF reader
  model_lfm2.c        LFM2 graph
  kv_cache.c          KV cache and convolution state
  kernels/scalar.c    reference implementation of every op
  metal/backend.m     Metal host code behind a C interface
  metal/*.metal       Metal kernels, embedded at build time
  kernels/neon.c      ARM NEON kernels, later
  kernels/avx2.c      x86 AVX2 kernels, later
tests/                test programs run by meson test
tests/fuzz/           libFuzzer harnesses
tools/                standalone programs such as bw.c and ref_dump.py
bench/                llama.cpp baseline checkout, gitignored
models/               downloaded GGUF files, gitignored
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
  preallocated buffers, so it never calls malloc" is right.

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

## C style

gip follows the
[GNU Coding Standards](https://www.gnu.org/prep/standards/standards.html)
and the GNU C formatting style. Where this document is silent, the GNU
standards decide. Where this document and the GNU standards disagree,
this document wins, and each such case below gives its reason.

### Language standard

Compile as `-std=gnu17`. GNU extensions that both GCC and Clang
support are welcome. The useful ones are `__attribute__`,
`__builtin_expect`, the `__builtin_*_overflow` family, and vector
extensions. `meson.build` defines `_GNU_SOURCE`, as GNU section 5.5
recommends.

### Formatting

The GNU style is enforced by `.clang-format` with `BasedOnStyle: GNU`
and a column limit of 79. `just fmt` applies it, and
`.clang-format-include` limits it to gip's own source directories.
The rules that shape every file, from GNU section 5.1:

- Keep lines to 79 columns.
- Put the return type on its own line. Start the function name in
  column one. Put the opening brace of a function body in column one.
- Indent by two spaces. Put the braces of a block on their own lines,
  indented two spaces from the statement, and indent the body two
  more.
- Put a space before every open parenthesis in a call and after every
  comma.
- Split a long expression before an operator. Add parentheses so the
  indentation shows the nesting.
- Put `struct` and `enum` braces in column one unless the whole body
  fits on one line.

```c
/* Return the dot product of N_BLOCKS Q8_0 blocks in WEIGHTS with the
   float vector X, which holds N_BLOCKS * QK8_0 values.  */
static float
dot_q8_0 (const struct q8_0_block *restrict weights, const float *restrict x,
          size_t n_blocks)
{
  float sum = 0.0f;

  for (size_t i = 0; i < n_blocks; i++)
    {
      float block_sum = 0.0f;
      for (int j = 0; j < QK8_0; j++)
        block_sum += weights[i].quants[j] * x[i * QK8_0 + j];
      sum += fp16_to_fp32 (weights[i].scale) * block_sum;
    }
  return sum;
}
```

### Constructs

From GNU section 5.3:

- Declare each variable on its own line when a declaration would span
  lines. Give each distinct purpose its own local variable with a
  meaningful name, declared in the smallest scope that covers its
  uses.
- Never shadow a global identifier. `-Wshadow` enforces the rule.
- Put braces around an `if`-`else` nested inside another `if`.
- Keep assignments out of `if` conditions. Assignments in `while`
  conditions are fine.
- Use `enum` for integer constants. GDB and LLDB show enum names.
- Put declarations of external functions near the top of the file or
  in a header. Never write `extern` inside a function.
- Declare a struct tag separately from any variable or typedef that
  uses it.

For configuration known at build time, GNU section 3.5 prefers
`if (HAS_FOO)` over `#ifdef HAS_FOO`, so the compiler checks both
paths. gip follows that rule everywhere except ISA-specific code.
NEON, AVX2, and SME intrinsics come from headers that exist only on
their own architecture. Each ISA gets its own source file, and
`meson.build` compiles that file only for its architecture, with its
own ISA flags.

Every `#else` and `#endif` more than a few lines from its `#if`
carries a comment that states the condition and its sense, as in
`#endif /* not __aarch64__ */`.

Mark kernel pointer parameters `restrict` and inputs `const`. Use
fixed-width types such as `uint32_t` for file format fields, and use
`size_t` for sizes and indices.

### Objective-C and Metal Shading Language

Objective-C appears only in `src/metal/*.m`, because Metal's host API
is Objective-C. Compile those files with ARC. Every function they
export has a plain C signature declared in a C header, so the rest of
the engine stays C. The GNU formatting and comment rules apply to
Objective-C and to `.metal` files as well.

### Naming

From GNU sections 4.3 and 5.4:

- Use lower case with underscores. Reserve upper case for macros and
  `enum` constants.
- Give globals and functions descriptive names. Locals can be short.
- Keep abbreviations few. The accepted domain abbreviations are `kv`,
  `qk`, `ffn`, `rms`, `rope`, `gguf`, `simd`, quant format names such
  as `q8_0`, and the `n_` prefix for counts.
- Prefix every symbol with external linkage with `gip_`. Make every
  other function and file-scope variable `static`.

The GNU standards give undocumented external symbols a leading
underscore. C reserves identifiers with a leading underscore at file
scope, as [CERT DCL37-C](https://wiki.sei.cmu.edu/confluence/display/c/DCL37-C.+Do+not+declare+or+define+a+reserved+identifier)
explains. gip builds with `-fvisibility=hidden` and exports only the
functions declared in `include/gip.h`, marked with `GIP_API`.
Internal functions keep the `gip_` prefix and stay hidden.

### Comments

From GNU section 5.2, adjusted by the prose rules above:

- Start every source file with a comment that names the file and says
  in a line or two what it is for. A file with `main` says what the
  program does.
- Put a comment above every function saying what the function does,
  what its arguments mean, and what it returns. Refer to argument
  values by the argument name in upper case, as in "Dequantize
  N_BLOCKS blocks from SRC into DST." Skip restating the function name
  or the C types.
- Put a comment above every file-scope variable.
- Write `/* */` comments in complete sentences that start with a
  capital letter. Put two spaces after each period and two spaces
  before the closing `*/`. The two spaces let Emacs sentence commands
  work. Markdown prose keeps one space.
- Inside function bodies, comment only the non-obvious why: a hidden
  constraint, an invariant, a numeric tolerance, or a workaround for a
  specific compiler or upstream bug. Skip comments that describe what
  the next line does.
- Describe the present code. Skip history such as "renamed from" and
  references to callers, issues, or pull requests. `git log` holds the
  history.
- Skip banners and Doxygen tags such as `@param`.

GNU's own examples use semicolons inside comments. gip's prose rules
ban semicolons, and the ban applies to comments too.

### Errors and robustness

From GNU sections 4.2 through 4.4:

- Library functions report failure through an `enum gip_status`
  return value. On bad input the library returns an error. The library
  never prints, exits, or aborts on bad input, because it runs inside
  other people's processes.
- Library functions are reentrant. The library keeps no mutable global
  state.
- Check every system call and every `malloc` and `realloc` result.
  Error messages from a failed system call include the file name and
  the text from `strerror`.
- An "impossible" condition means a bug in gip. Call `abort` and
  explain the condition in a comment.
- Tools print errors as `PROGRAM: FILE: message`, starting lower case
  with no final period. A failed allocation in a tool is fatal.
- Tools parse options with `getopt_long`, provide long options, and
  support `--help` and `--version`. Exit status is 0 on success and 1
  on failure. An error count never becomes an exit status.
- Temporary files honor `TMPDIR` and open with `O_EXCL` or `mkstemp`.
- The GGUF loader tries `mmap` on each file and falls back to `read`
  when `mmap` fails for that file, as GNU section 5.11 asks.

### Untrusted input

Users download GGUF files from strangers, so every byte of a model
file is untrusted. llama.cpp has shipped memory-safety bugs in GGUF
parsing. The rules below close off that class of bug in gip's
parser.

- Validate every count, offset, dimension, and string length against
  the file size before using it.
- Compute every size with `__builtin_mul_overflow` and
  `__builtin_add_overflow`, and reject the file on overflow.
- Read multi-byte fields with `memcpy` into fixed-width types. GGUF is
  little-endian. Casting file bytes to a struct pointer breaks on
  misaligned data and violates strict aliasing, so the parser never
  does it.
- Keep all parsing in `src/gguf.c`. Every change to that file runs
  `just fuzz` before it is proposed for commit.

### Memory and threads

- The engine allocates everything at load time: weight mappings, the
  KV cache, convolution state, and scratch buffers. The decode loop
  makes no call to `malloc`, `free`, or any system call other than
  thread synchronization.
- Align every buffer a kernel reads to 64 bytes.
- The Metal backend wraps the memory-mapped GGUF file in one
  `MTLBuffer` with `newBufferWithBytesNoCopy`, so the GPU reads weights
  from the page cache with no copy. The mapping is page-aligned, and
  each tensor is an offset into that buffer.
- The KV cache, convolution state, and scratch space live in
  `MTLBuffer`s created at load time. A decode step creates no Metal
  objects.
- The CPU thread pool starts at load time and keeps its threads alive.
  Workers spin on barriers between ops, because waking a parked thread
  costs microseconds and a decode step runs hundreds of ops. Decode
  threads run on performance cores. Linux pins them with
  `sched_setaffinity`. macOS offers no pinning, so workers set
  `QOS_CLASS_USER_INTERACTIVE`.

### Kernels

- Every op has a scalar implementation in `src/kernels/scalar.c`. The
  scalar version defines the correct output.
- Every Metal kernel and every SIMD kernel is tested against the
  scalar version on random inputs, with the tolerance stated in the
  test.

Metal kernels:

- Create every compute pipeline state at load time. Specialize kernels
  for fixed shapes such as head size with Metal function constants.
- Encode each decode step into one command buffer. Host encoding time
  is part of every token, so measure it along with GPU time.
- Keep threadgroup sizes and tile sizes in named constants with a
  comment giving the measurement that chose them.

CPU kernels, which come after the Metal backend:
- The loader detects CPU features once and fills a table of function
  pointers. Hot loops call through the table once per op.
- Prefer intrinsics. Use inline assembly only for an instruction the
  intrinsics cannot express, and say which instruction in a comment.
- Apple's M4 has SME and lacks SVE outside streaming mode. SVE code
  outside a streaming function crashes with `SIGILL` on the M4.

### Compiler flags

Every build uses these warnings. Meson's `warning_level=2` supplies
`-Wall -Wextra`, and `meson.build` adds the rest:

```
-Wall -Wextra -Wshadow -Wformat=2 -Wimplicit-fallthrough
-Wstrict-prototypes -Wmissing-prototypes
-Werror=implicit -Werror=incompatible-pointer-types
-Werror=int-conversion -Werror=format-security
```

The `justfile` configures development builds with `-Dwerror=true`.
Release tarballs leave `-Werror` out, as the
[OpenSSF compiler hardening guide](https://best.openssf.org/Compiler-Hardening-Guides/Compiler-Options-Hardening-Guide-for-C-and-C++.html)
recommends, so a newer compiler's new warning does not break a user's
build.

gip leaves `-Wconversion` off. GNU section 5.3 asks programs to
avoid casts whose only job is to quiet extra warnings, and
`-Wconversion` would demand hundreds of them in kernel code. The
parser gets its integer safety from overflow builtins and fuzzing.

Release builds add the hardening flags from the OpenSSF guide that
leave kernel code generation alone:

| Flag | Platform |
|---|---|
| `-D_FORTIFY_SOURCE=3` | glibc |
| `-fstack-protector-strong` | all |
| `-fstack-clash-protection` | Linux |
| `-mbranch-protection=standard` | AArch64 |
| `-fcf-protection=full` | x86-64 |
| `-fPIC` | the shared library |
| `-Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack` | ELF targets |

The guide also recommends `-fno-strict-aliasing`,
`-fno-strict-overflow`, and `-ftrivial-auto-var-init=zero`. Those
three change how the compiler optimizes hot loops. gip adopts each
one only after `just bench` shows its cost.

The sanitizer build uses meson's `-Db_sanitize=address,undefined`.
Thread pool tests also run under `-Db_sanitize=thread` in a build
directory of their own.

## Writing tests

Write only tests that catch a real regression. A test that checks a
constant or restates the implementation is noise.

The tests that matter:

- Each Metal kernel and each SIMD kernel against the scalar kernel.
- Hidden states after every layer against dumps from `transformers`,
  produced by `tools/ref_dump.py`.
- Greedy decoding that matches `transformers` token for token for 100
  tokens.
- The parser rejecting malformed files, from crafted cases and the
  fuzz corpus.

Each test is a standalone C program in `tests/`, registered with
`test()` in `tests/meson.build`. The programs follow the Automake exit
status convention, which `meson test` understands. Exit 0 means pass,
77 means skip, and any other status means failure. A test that needs
a model file from `models/`, a Metal device, or a missing CPU feature
exits 77. gip uses no test framework.

## Benchmarks

The headline claim is the fastest LLM inference on a Mac. The engines
to beat are mlx-lm and llama.cpp's Metal backend. Every performance
claim comes with numbers from this protocol:

- Use the same GGUF file for gip and llama.cpp.
- Run mlx-lm on Liquid's official MLX 8-bit weights. Report the model
  bytes of each engine's weights, since MLX's 8-bit format differs
  from GGUF Q8_0.
- Record the llama.cpp commit hash, the mlx and mlx-lm versions, and
  the build flags.
- Measure prefill with a 512-token prompt and decode with 128
  generated tokens.
- Run each setting 5 times and report the mean and standard
  deviation.
- Report the decode ceiling: measured GPU read bandwidth divided by
  model bytes.
- Run on an idle machine with no build in progress, on AC power.

CPU benchmarks follow the same protocol against llama.cpp's CPU build
and sweep 1, 4, 8, and 10 threads on performance cores.
`tools/bw.c` gives the CPU decode ceiling.

A change that slows any benchmark gets reported with before and after
numbers when it is proposed for commit.

## Adding a model architecture

1. Read the architecture's modeling file in `transformers` and its
   graph in llama.cpp. Write down every op in one layer, in order.
2. Dump reference token ids, per-layer hidden states, and logits with
   `tools/ref_dump.py`.
3. Write the scalar forward pass. Compare against the dump layer by
   layer and stop at the first layer that diverges.
4. Match greedy decoding token for token for 100 tokens.
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
Before writing any other non-trivial logic, search libc and POSIX
first, then the libraries already vendored under `third_party/`, then
established C libraries. Vendor a library with its license file and a
pinned version.

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
| [mistral.rs](https://github.com/EricLBuehler/mistral.rs) | Paged attention, speculative decoding, and engine API |
| [candle](https://github.com/huggingface/candle) | A second implementation of ggml's quant block layouts |
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
