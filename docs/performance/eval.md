# Evaluation

The runs used an Apple M4 Pro with 14 CPU cores, 20 GPU cores, and 48 GB
of memory, on macOS 26.5.1 and AC power. Zoom, Slack, and a browser
stayed open throughout.

Every engine ran LFM2.5-2.6B on a prompt of 512 copies of one token and
then generated 128 tokens greedily without stopping at an end token. The
time to first token covers the whole prompt and the first output token.
The time per output token is the time from the first output token to the
last, divided by 127. The figure shows both as rates: 512 prompt tokens
over the time to first token, and one token over the time per output
token.

bobcat, llama.cpp, mistral.rs, and candle read Liquid AI's
`LFM2.5-2.6B-QAD-Q4_0.gguf` of 1.59 GB. The same file gives the four
engines the same weights and the same bytes to read per token. Liquid AI
trained those Q4_0 weights with quantization-aware distillation. mlx-lm read
Liquid AI's `LFM2.5-2.6B-MLX-4bit` of 1.58 GB. ExecuTorch read a 4-bit
export of 1.96 GB from its own LFM2 recipe, with bf16 compute and a bf16
embedding table that its pipeline cannot quantize. Cactus read a 1.40 GB
bundle in its own 4-bit CQ4 format, which its converter built with a
generic graph because Cactus has no profile for this model.

bobcat, llama.cpp, and mlx-lm ran through matched harnesses that time
each token as the caller receives it: `bobcat-bench --latency`,
`tools/llama_latency.cpp`, and `tools/mlx_latency.py`. ExecuTorch ran its
MLX backend on the GPU through `tools/executorch_latency.py`, which calls
the model once for the prompt and once per output token. Cactus ran on
the GPU through `tools/cactus_latency.py`, which reads Cactus's own timing
of `cactus_benchmark_tokens`. mistral.rs reported its own time to first
token and time per output token through `mistralrs bench`. candle's
example reported prompt and generation rates. The prompt's duration
stands in for its time to first token.

`tools/performance_bench.py` ran five rounds. Each round ran every engine
once after that tool's own warmup. The engine order rotated from round
to round, which spread the load of the open applications across the
engines. Each bar in the figure is the median round, which discards a
one-off slow round. The error bars span the slowest to the fastest
round. [`lfm2.5-2.6b.csv`](lfm2.5-2.6b.csv) holds every round.
