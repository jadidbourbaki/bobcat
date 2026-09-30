# Evaluation

The runs used an Apple M4 Pro with 14 CPU cores, 20 GPU cores, and 48 GB
of memory, on macOS 26.5.1 and AC power. Zoom, Slack, and a browser
stayed open throughout.

Every engine ran LFM2.5-2.6B. bobcat, llama.cpp, mistral.rs, and candle
read Liquid AI's `LFM2.5-2.6B-Q4_K_M.gguf` of 1.67 GB. The same file
gives the four engines the same weights and the same bytes to read per
token. mlx-lm cannot read GGUF, so it read Liquid AI's
`LFM2.5-2.6B-MLX-4bit` of 1.58 GB, the official MLX weights closest in
size. The MLX file is 5% smaller, which favors mlx-lm slightly in
generation.

The two tests are llama-bench's defaults, the rates that llama.cpp's
Apple Silicon performance thread compares Macs on. pp512 is the rate of
processing a 512-token prompt. tg128 is the rate of generating 128
tokens greedily after a 512-token prompt. Generation follows the prompt
because a real reply always does. Generating from an empty context
would skip the attention cost of the prompt.

Each engine ran through its own benchmark tool, which measures the
engine the way its authors do. bobcat ran `bobcat-bench`. llama.cpp ran
`llama-bench -p 512 -n 0` and `llama-bench -p 0 -n 128 -d 512`. mlx-lm
ran `mlx_lm.benchmark -p 512 -g 128`. mistral.rs ran
`mistralrs bench --depth 512`. candle ran its quantized LFM2 example
through `tools/candle_bench.py`. A shared harness would have timed every
engine through bobcat's own choices about warmup, synchronization, and
sampling.

bobcat, llama.cpp, and mlx-lm fed BOS and then random tokens, as
llama-bench does. candle's example takes text, so its prompt counts
upward to 512 tokens. The cost of a step in these dense models does not
depend on which token it runs.

`tools/performance_bench.py` ran five rounds. Each round ran every engine
once after that tool's own warmup. The engine order rotated from round
to round, which spread the load of the open applications across the
engines. Each bar in the figure is the median round, which discards a
one-off slow round. The error bars span the slowest to the fastest
round. [`lfm2.5-2.6b.csv`](lfm2.5-2.6b.csv) holds every round.
