llama_bench_cpu := "bench/llama.cpp/build/bin/llama-bench"
llama_bench_metal := "bench/llama.cpp/build-metal/bin/llama-bench"
mistralrs := "bench/mistral.rs/target/release/mistralrs"
candle_lfm2 := "bench/candle/target/release/examples/quantized-lfm2"
gguf_models := "models/LFM2.5-350M-Q8_0.gguf models/LFM2.5-1.2B-Instruct-Q8_0.gguf models/LFM2.5-2.6B-Q8_0.gguf"
mlx_models := "models/LFM2.5-350M-MLX-8bit models/LFM2.5-1.2B-Instruct-MLX-8bit models/LFM2.5-2.6B-MLX-8bit"
kernels := "crates/bobcat-metal/src/common.metal crates/bobcat-metal/src/quant.metal crates/bobcat-metal/src/matvec.metal crates/bobcat-metal/src/matmul.metal crates/bobcat-metal/src/matmul_tensor.metal crates/bobcat-metal/src/attention.metal crates/bobcat-metal/src/conv.metal crates/bobcat-metal/src/norm.metal"

# List the recipes.
default:
    @just --list

# Build every crate and tool.
build:
    cargo build --workspace --all-targets --locked

# Run the tests.
test:
    cargo test --workspace --locked

# Run the full quality gate.
check:
    cargo fmt --all --check
    cargo sort --workspace --check
    clang-format --dry-run -Werror {{kernels}}
    cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
    cargo test --workspace --locked
    cd tools && uv run ruff format --check .
    cd tools && uv run ruff check .
    cd tools && uv run ty check .

# Format the Rust sources, the manifests, the Metal kernels, and the Python tools.
fmt:
    cargo fmt --all
    cargo sort --workspace
    clang-format -i {{kernels}}
    cd tools && uv run ruff format .

# Measure GPU bandwidth, bobcat, and the llama.cpp, mlx-lm, mistral.rs, and candle baselines.
bench:
    cargo build --release --locked -p bobcat-bench
    target/release/gpu-bw
    @echo "bobcat $(git rev-parse --short HEAD), Metal"
    for model in {{gguf_models}}; do target/release/bobcat-bench "$model"; done
    @echo "llama.cpp $(git -C bench/llama.cpp rev-parse --short HEAD), Metal"
    {{llama_bench_metal}} $(printf -- '-m %s ' {{gguf_models}}) -p 512 -n 0 -r 5 -o md
    {{llama_bench_metal}} $(printf -- '-m %s ' {{gguf_models}}) -p 0 -n 128 -d 512 -r 5 -o md
    @cd tools && uv run python -c "import mlx.core, mlx_lm; print('mlx', mlx.core.__version__, 'mlx-lm', mlx_lm.__version__)"
    for model in {{mlx_models}}; do echo "$model"; (cd tools && uv run python -m mlx_lm.benchmark --model "../$model" -p 512 -g 128 -n 5); done
    @echo "mistral.rs $(git -C bench/mistral.rs rev-parse --short HEAD), Metal"
    for model in {{gguf_models}}; do echo "$model"; {{mistralrs}} bench -f "$model" --prompt-len 512 --gen-len 128 --depth 512 --iterations 5 --warmup 1 2>/dev/null | grep -E 'TTFT \(|Decode \('; done
    @echo "candle $(git -C bench/candle rev-parse --short HEAD), Metal"
    for model in {{gguf_models}}; do (cd tools && uv run python candle_bench.py --binary "../{{candle_lfm2}}" --model "../$model" --tokenizer "../${model%-Q8_0.gguf}-MLX-8bit/tokenizer.json"); done

# Measure CPU bandwidth and the llama.cpp CPU baseline.
bench-cpu:
    cargo build --release --locked -p bobcat-bench
    target/release/cpu-bw
    @echo "llama.cpp $(git -C bench/llama.cpp rev-parse --short HEAD), CPU"
    {{llama_bench_cpu}} $(printf -- '-m %s ' {{gguf_models}}) -t 1,4,8,10 -p 512 -n 0 -r 5 -o md
    {{llama_bench_cpu}} $(printf -- '-m %s ' {{gguf_models}}) -t 1,4,8,10 -p 0 -n 128 -d 512 -r 5 -o md

# Measure warmed engine latency for greedy streaming output.
bench-latency model="models/LFM2.5-2.6B-Q4_K_M.gguf":
    cargo build --release --locked -p bobcat-bench
    target/release/bobcat-bench --latency --reps 20 {{model}}

# Remove the build output.
clean:
    cargo clean
