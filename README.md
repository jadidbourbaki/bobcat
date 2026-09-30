<p align="center">
  <img src="assets/banner.svg" width="500" alt="bobcat">
</p>

<p align="center">
  <a href="https://github.com/jadidbourbaki/bobcat/actions/workflows/ci.yml"><img src="https://github.com/jadidbourbaki/bobcat/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/macOS-26%2B-black.svg?logo=apple" alt="macOS 26 or newer">
  <a href="rust-toolchain.toml"><img src="https://img.shields.io/badge/rust-1.98-orange.svg?logo=rust" alt="Rust 1.98"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
  <a href="https://github.com/sponsors/jadidbourbaki"><img src="https://img.shields.io/badge/Sponsor-❤️-pink.svg" alt="Sponsor"></a>
</p>

**bobcat** is an inference engine optimized for Apple silicon. Run
local models on your Mac at unbelievably fast speeds.

```console
bobcat chat -m lfm2.5:2.6b
> What is the capital of Japan? One sentence.
The capital of Japan is Tokyo.
```

## Quick start

Install bobcat on a Mac with Apple silicon and macOS 26 or newer:

```sh
curl -fsSL https://raw.githubusercontent.com/jadidbourbaki/bobcat/main/install.sh | sh
```

Download a model:

```sh
bobcat pull lfm2.5:2.6b
```

Chat with it:

```sh
bobcat chat -m lfm2.5:2.6b
```

Or answer one prompt, for scripts and pipes:

```sh
bobcat respond -m lfm2.5:2.6b "Name three primes."
```

## Performance results

![Prompt processing and generation throughput of bobcat, llama.cpp, mlx-lm, mistral.rs, and candle on LFM2.5-2.6B](docs/performance/lfm2.5-2.6b.svg)

LFM2.5-2.6B on an M4 Pro, with llama-bench's pp512 and tg128 tests. The
raw runs are in [`docs/performance`](docs/performance).

## bobcat ❤️ agents

Start the server, then point your agent at it:

```sh
bobcat serve -m lfm2.5:2.6b -c 65536
```

**[Claude Code](https://docs.claude.com/en/docs/claude-code/overview)**

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 \
ANTHROPIC_AUTH_TOKEN=bobcat \
ANTHROPIC_API_KEY="" \
CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536 \
claude --model lfm2.5:2.6b
```

**[opencode](https://opencode.ai)**, in `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "bobcat": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "bobcat",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": { "lfm2.5:2.6b": { "name": "LFM2.5-2.6B" } }
    }
  }
}
```

**[aider](https://aider.chat)**

```sh
OPENAI_API_BASE=http://127.0.0.1:8080/v1 \
OPENAI_API_KEY=bobcat \
aider --model openai/lfm2.5:2.6b
```

## Supported models

**[LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF)**

```sh
bobcat pull lfm2.5:350m
```

**[LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF)**

```sh
bobcat pull lfm2.5:1.2b
```

**[LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF)**

```sh
bobcat pull lfm2.5:2.6b
```

bobcat is still in alpha. We are rapidly adding support for more models
and model families. Please stay tuned!
