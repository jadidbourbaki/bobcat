# Evaluation

## Setting

- **Machine:** Apple M4 Pro with 14 CPU cores, 20 GPU cores, and 48 GB
  of memory, on macOS 26.5.1 and AC power. Zoom, Slack, and a browser
  stayed open during the runs.
- **Model:** LFM2.5-2.6B. bobcat, llama.cpp, mistral.rs, and candle read
  Liquid AI's `LFM2.5-2.6B-Q4_K_M.gguf`, 1.67 GB. mlx-lm reads Liquid
  AI's `LFM2.5-2.6B-MLX-4bit`, 1.58 GB.
- **Tests:** pp512 measures the rate of processing a 512-token prompt.
  tg128 measures the rate of generating 128 tokens greedily after a
  512-token prompt.
- **Tools:** each engine ran through its own benchmark tool.
  - bobcat: `bobcat-bench`.
  - llama.cpp: `llama-bench -p 512 -n 0` and `-p 0 -n 128 -d 512`.
  - mlx-lm: `mlx_lm.benchmark -p 512 -g 128`.
  - mistral.rs: `mistralrs bench --depth 512`.
  - candle: its quantized LFM2 example, through `tools/candle_bench.py`.
- **Prompts:** bobcat, llama.cpp, and mlx-lm feed BOS and then random
  tokens. candle's example takes text, so its prompt counts upward to
  512 tokens.
- **Rounds:** `tools/performance_bench.py` ran five rounds. Each round
  ran every engine once after that tool's own warmup. The engine order
  rotated from round to round. Each bar is the median round. The
  error bars span the slowest to the fastest round.

## Why this setting

- **pp512 and tg128 are llama-bench's default tests.** llama.cpp's
  Apple Silicon performance thread compares every Mac on them, and the
  other engines' tools report the same two rates.
- **Each engine's own tool measures the engine the way its authors
  do.** A shared harness would time every engine through bobcat's
  choices about warmup, synchronization, and sampling.
- **Four engines read one file.** The same GGUF file gives them the
  same weights and the same bytes to read per token. mlx-lm cannot read
  GGUF, so it reads Liquid AI's official MLX weights of the closest
  size. The MLX file is 5% smaller, which favors mlx-lm slightly in
  generation.
- **Generation runs after the prompt.** A reply always follows a prompt,
  so tg128 at a depth of 512 reflects a real reply. Generation from an
  empty context would skip the attention cost of the prompt.
- **Random tokens stand in for text.** For these dense models the cost
  of a step does not depend on which token it runs. Random tokens also
  match llama-bench exactly.
- **Rotation and medians absorb background load.** The open
  applications slow every engine, and the rotating order spreads that
  load across engines. The median discards the one-off slow round.

[`lfm2.5-2.6b.csv`](lfm2.5-2.6b.csv) holds every round.
[`machine.txt`](machine.txt) records the machine.
