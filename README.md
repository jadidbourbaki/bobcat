# gip

gip, the general inference program, runs language models on the Apple
silicon GPU. It reads GGUF files, the format most quantized models on
Hugging Face ship in, and runs them with its own Metal kernels. The
goal is the fastest local inference on a Mac.

```console
$ gip chat -m LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M
> What is the capital of Japan? One sentence.
The capital of Japan is Tokyo.
```

gip is early. It runs Liquid AI's LFM2 and LFM2.5 models today, in the
Q8_0, Q4_0, and Q4_K_M quantizations. More model families come next.

## Install

gip needs a Mac with Apple silicon, macOS 15 or newer, and a Rust
toolchain from [rustup](https://rustup.rs). It needs no Xcode, because
it compiles its GPU kernels when it starts.

```console
$ cargo install --locked --git https://github.com/jadidbourbaki/gip gip-cli
```

## Use

`gip chat` holds a conversation in the terminal. Ctrl-D ends it.
Models that reason before they answer, such as LFM2.5-2.6B, show their
reasoning in dim text first.

```console
$ gip chat -m LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M
```

`gip respond` answers one prompt and writes only the answer to
standard output, so it works in scripts and pipes. Text on standard
input follows the prompt.

```console
$ gip respond -m LiquidAI/LFM2.5-1.2B-Instruct-GGUF "Name three primes."
$ cat notes.md | gip respond -m LiquidAI/LFM2.5-1.2B-Instruct-GGUF "Summarize this."
```

A model name is a Hugging Face repository with an optional
quantization tag, which defaults to Q8_0. gip downloads a model the
first time a command names it, into the shared Hugging Face cache at
`~/.cache/huggingface`. `-m` also takes a path to a GGUF file.

| Command | Meaning |
|---|---|
| `gip pull NAME` | Download a model and print the path of its file |
| `gip list` | List the downloaded models and their sizes |
| `gip rm NAME` | Remove a downloaded model |

Replies sample with the settings the model file recommends, or with
the settings of the model's authors when the file names none.
`--temperature`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`,
and `--seed` override them. `gip help` lists every option.

## Speed

Tokens per second on an M4 Pro with a 20-core GPU, measured on
2026-09-28 against llama.cpp `4da6337` on the same GGUF files. Each
run prefills a 512-token prompt and then generates 128 tokens, and
each number is the mean of 5 runs.

| Model | gip decode | llama.cpp decode | gip prefill | llama.cpp prefill |
|---|---|---|---|---|
| LFM2.5-350M Q4_K_M | 612 | 554 | 7,516 | 9,793 |
| LFM2.5-2.6B Q4_K_M | 119 | 117 | 994 | 1,269 |

gip generates text faster than llama.cpp. Prompt processing is slower,
and it is the current focus of the kernel work. `just bench` in a
checkout reruns these measurements, along with mlx-lm, mistral.rs, and
candle.

## Develop

`AGENTS.md` describes the code layout, the conventions, and the
benchmark protocol. `just check` runs the formatters, the linters, and
the tests. The tests compare every layer of the forward pass with
transformers.

## License

MIT
