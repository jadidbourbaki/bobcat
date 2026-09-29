<p align="center">
  <img src="assets/banner.svg" width="500" alt="bobcat">
</p>

<p align="center">
  <a href="https://github.com/jadidbourbaki/bobcat/actions/workflows/ci.yml"><img src="https://github.com/jadidbourbaki/bobcat/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="rust-toolchain.toml"><img src="https://img.shields.io/badge/rust-1.98-orange.svg?logo=rust" alt="Rust 1.98"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
  <a href="https://github.com/sponsors/jadidbourbaki"><img src="https://img.shields.io/badge/Sponsor-❤️-pink.svg" alt="Sponsor"></a>
</p>

**bobcat** is an inference engine optimized for Apple silicon. Run
local models on your Mac at unbelievably fast speeds.

```console
$ bobcat chat -m lfm2.5:1.2b
> What is the capital of Japan? One sentence.
The capital of Japan is Tokyo.
```

## Quick start

Install bobcat on a Mac with Apple silicon and macOS 15 or newer:

```console
$ curl -fsSL https://raw.githubusercontent.com/jadidbourbaki/bobcat/main/install.sh | sh
```

Download a model:

```console
$ bobcat pull lfm2.5:1.2b
```

Chat with it:

```console
$ bobcat chat -m lfm2.5:1.2b
```

Or answer one prompt, for scripts and pipes:

```console
$ bobcat respond -m lfm2.5:1.2b "Name three primes."
```

## Supported models

- [LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF): `lfm2.5:350m`
- [LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF): `lfm2.5:1.2b`
- [LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF): `lfm2.5:2.6b`

Other LFM2 and LFM2.5 models on Hugging Face run by their full name, as
in `bobcat chat -m LiquidAI/LFM2-1.2B-GGUF`.
