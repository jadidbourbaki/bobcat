# Overview

gip runs language models on the Mac's GPU. The engine reads GGUF files
from Hugging Face and runs them with Metal kernels that gip writes
itself. The first goal is the fastest inference on Apple silicon, so
the code serves one GPU family and one model family well before it
grows wider.

## Crates

The workspace in `Cargo.toml` holds five crates.

| Crate | Job |
|---|---|
| `gip-gguf` | Parses GGUF files in safe Rust |
| `gip-metal` | Compiles the Metal kernels and launches them |
| `gip` | Loads LFM2 models, holds the scalar reference ops, and runs LFM2 on the CPU and the GPU |
| `gip-cli` | Builds the `gip` command |
| `gip-bench` | Builds the benchmark and bandwidth programs |

The dependencies run in one direction. `gip-cli` and `gip-bench`
depend on `gip`. `gip-cli` also reads GGUF metadata through `gip-gguf`
to rebuild the tokenizer. `gip` depends on `gip-gguf` and `gip-metal`.
The two lowest crates know nothing of each other or of any model.

The engine is a library first. The `gip` command calls the same public
API that another program would. A C API for Swift and other languages
comes later.

## One prompt, start to finish

`gip respond -m lfm2.5:1.2b "Name three primes."` runs through every
crate.

1. `gip-cli` expands the alias to the Hugging Face name
   `LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M` and finds the file in
   the Hugging Face cache, downloading it on first use.
2. `gip::Model::load` memory-maps the file and hands the bytes to
   `gip-gguf`, which parses the metadata and the tensor headers. The
   loader checks every tensor's shape and records where each tensor
   lives in the mapping.
3. `gip-cli` rebuilds the tokenizer from the file's metadata, renders
   the chat template around the prompt, and encodes the result into
   token ids.
4. `gip-metal` opens the GPU and compiles the kernels. `gip::Lfm2Metal`
   wraps the file mapping in one Metal buffer, so the GPU reads the
   weights in place with no copy. `Lfm2Metal::new` allocates the KV
   cache and scratch buffers once.
5. `Lfm2Metal::prefill` runs the whole prompt through the model in
   batches of up to 512 tokens.
6. `Lfm2Metal::generate` decodes the answer one token per step. With
   greedy decoding, the GPU picks each token itself.
7. `gip-cli` decodes each token id back to text, separates the thinking
   from the answer, and writes the answer to standard output as it
   arrives.

## The reference and the tests

Every operation has a scalar version in `crates/gip/src/scalar.rs`
that uses plain loops and double-precision sums. The scalar version
defines the correct output. Tests hold every GPU kernel to the scalar
version. Other tests hold the scalar model to Hugging Face
transformers, layer by layer.

A new model architecture follows the same order. The scalar forward
pass comes first and has to match transformers. GPU kernels come after,
each tested against the scalar pass.

## Further reading

| Document | Subject |
|---|---|
| [GGUF](gguf.md) | The file format, the parser, and the quantization formats |
| [LFM2](lfm2.md) | The model architecture, the CPU forward pass, and the reference tests |
| [Metal](metal.md) | The GPU backend, the kernels, and how a token is computed |
| [The gip command](cli.md) | Subcommands, model names, the tokenizer, the chat template, and sampling |
