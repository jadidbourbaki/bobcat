# Evaluation

The runs used an Apple M4 Pro with 14 CPU cores, 20 GPU cores, and 48 GB
of memory, on macOS 26.5.1 and AC power. Zoom, Slack, and a browser
stayed open throughout.

Every engine ran each model on a prompt of 512 copies of one token and
then generated 128 tokens greedily without stopping at an end token. The
time to first token covers the whole prompt and the first output token.
The time per output token is the time from the first output token to the
last, divided by 127. The figure shows both as rates: 512 prompt tokens
over the time to first token, and one token over the time per output
token.

bobcat, llama.cpp, mistral.rs, and candle read one GGUF file per model.
For LFM2.5-350M, LFM2.5-1.2B-Instruct, and LFM2.5-2.6B the file is Liquid
AI's QAD-Q4_0 file, whose Q4_0 weights Liquid AI trained with
quantization-aware distillation. Liquid AI publishes no QAD file of
LFM2.5-8B-A1B, so its file is the plain `LFM2.5-8B-A1B-Q4_0.gguf`. The
same file gives the four engines the same weights and the same bytes to
read per token. mlx-lm read Liquid AI's MLX-4bit weights of each model.
ExecuTorch read 4-bit exports from its own LFM2 recipe, with bf16 compute
and a bf16 embedding table that its pipeline cannot quantize. Cactus read
bundles in its own 4-bit CQ4 format. Its converter has a profile for
LFM2.5-8B-A1B and builds the other models with a generic graph.

Three engines skip LFM2.5-8B-A1B. ExecuTorch's LFM2 model has no
mixture-of-experts layers. candle's LFM2 example reads only dense
models. mistral.rs loads the model but generated about 0.2 tokens per
second in a short trial, so five rounds would have taken hours.

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
round. The CSV of each model, such as
[`lfm2.5-2.6b.csv`](lfm2.5-2.6b.csv), holds every round.
