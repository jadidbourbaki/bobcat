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

<p align="center">
  <img src="docs/performance/lfm2.5-2.6b.svg" alt="Prompt processing and generation throughput on LFM2.5-2.6B">
</p>

LFM2.5-2.6B on an M4 Pro, with a 512-token prompt and 128 generated
tokens. The details are [here](docs/performance/eval.md).

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

<p align="center">
  <img src="docs/performance/lfm2.5-350m.svg" alt="Prompt processing and generation throughput on LFM2.5-350M">
</p>

**[LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF)**

```sh
bobcat pull lfm2.5:350m
```

<p align="center">
  <img src="docs/performance/lfm2.5-1.2b.svg" alt="Prompt processing and generation throughput on LFM2.5-1.2B-Instruct">
</p>

**[LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF)**

```sh
bobcat pull lfm2.5:1.2b
```

<p align="center">
  <img src="docs/performance/lfm2.5-2.6b.svg" alt="Prompt processing and generation throughput on LFM2.5-2.6B">
</p>

**[LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF)**

```sh
bobcat pull lfm2.5:2.6b
```

<p align="center">
  <img src="docs/performance/lfm2.5-8b.svg" alt="Prompt processing and generation throughput on LFM2.5-8B-A1B">
</p>

**[LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF)**

```sh
bobcat pull lfm2.5:8b
```

**[Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF)**

```sh
bobcat pull qwen3.5:0.8b
```

**[Qwen3.5-2B](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)**

```sh
bobcat pull qwen3.5:2b
```

**[Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)**

```sh
bobcat pull qwen3.5:4b
```

**[Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)**

```sh
bobcat pull qwen3.5:9b
```

**[Clef-Flash](https://huggingface.co/Cloudflare/clef-flash)**

```sh
hf download Cloudflare/clef-flash --local-dir clef-flash
python llama.cpp/convert_hf_to_gguf.py clef-flash --no-mtp --outtype q8_0 \
  --outfile clef-flash-backbone.gguf
cd tools && uv run python clef_gguf.py --backbone ../clef-flash-backbone.gguf \
  --release ../clef-flash --out ../clef-flash-Q8_0.gguf
```

```sh
echo '{"model": "clef-flash", "state": "Checkout is down.",
  "questions": {"outage": {"type": "noul"}}}' | bobcat decide -m clef-flash-Q8_0.gguf
```

`bobcat serve -m clef-flash-Q8_0.gguf` also answers `POST /v1/systemone`.

bobcat is still in alpha. We are rapidly adding support for more models
and model families. Please stay tuned!
