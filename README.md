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

## Supported models

Each model has a short name. Pull a model by its short name, then pass
the same name to `bobcat chat -m` or `bobcat respond -m`. `chat` and
`respond` also pull a model the first time they use it.

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

Other LFM2 and LFM2.5 models on Hugging Face pull by their full name:

```sh
bobcat pull LiquidAI/LFM2-1.2B-GGUF
```

bobcat is still in alpha. We are rapidly adding support for more models
and model families. Please stay tuned!

## Using bobcat with agents

`bobcat serve` answers requests in the OpenAI Chat Completions API and
the Anthropic Messages API, so coding agents and chat apps can run on a
local model:

```sh
bobcat serve -m lfm2.5:2.6b
```

The server listens on `http://127.0.0.1:8080` and answers one request
at a time. Agents send the whole conversation with every request, and
bobcat skips the part of the prompt it has already run.

| Flag | Default | Meaning |
|---|---|---|
| `-m`, `--model` | `BOBCAT_MODEL` | The model to serve |
| `--host` | `127.0.0.1` | The address to listen on |
| `--port` | `8080` | The port to listen on |
| `-c`, `--context` | `32768` | The most tokens in one conversation |
| `-n`, `--max-tokens` | `4096` | The most tokens in a reply when a request sets no limit |

**[Claude Code](https://docs.claude.com/en/docs/claude-code/overview)**

Claude Code's system prompt and tools take about 20,000 tokens, so give
the server a larger context:

```sh
bobcat serve -m lfm2.5:2.6b -c 65536
```

Then start Claude Code with the same context:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 \
ANTHROPIC_AUTH_TOKEN=bobcat \
ANTHROPIC_API_KEY="" \
CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536 \
claude --model lfm2.5:2.6b
```

**[opencode](https://opencode.ai)**

Add bobcat as a provider in `opencode.json`:

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

Then pick the model:

```sh
opencode -m bobcat/lfm2.5:2.6b
```

**[aider](https://aider.chat)**

```sh
OPENAI_API_BASE=http://127.0.0.1:8080/v1 \
OPENAI_API_KEY=bobcat \
aider --model openai/lfm2.5:2.6b
```

**Other OpenAI clients**

Point the client's base URL at `http://127.0.0.1:8080/v1` and give it
any API key. bobcat ignores the key.

Codex CLI talks only to the OpenAI Responses API, which bobcat does not
serve yet.
