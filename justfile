build_dir := "build/default"
sanitize_dir := "build/sanitize"
release_dir := "build/release"
llama_bench_cpu := "bench/llama.cpp/build/bin/llama-bench"
llama_bench_metal := "bench/llama.cpp/build-metal/bin/llama-bench"
gguf_models := "-m models/LFM2.5-350M-Q8_0.gguf -m models/LFM2.5-1.2B-Instruct-Q8_0.gguf -m models/LFM2.5-2.6B-Q8_0.gguf"
mlx_models := "models/LFM2.5-350M-MLX-8bit models/LFM2.5-1.2B-Instruct-MLX-8bit models/LFM2.5-2.6B-MLX-8bit"

# Apple clang's AddressSanitizer crashes at startup on some macOS
# releases, so macOS sanitizer builds use Homebrew's LLVM with the
# installed SDK.  Other systems use their default compiler.  Homebrew's
# clang emits objc_msgSendClass stubs that only the Xcode 26 linker
# fills in, and the flag below turns them off for older linkers.
llvm_clang := "/opt/homebrew/opt/llvm/bin/clang"
sanitize_env := if os() == "macos" { "CC=" + llvm_clang + " OBJC=" + llvm_clang } else { "" }
sanitize_sdk := if os() == "macos" { `xcrun --show-sdk-path` } else { "" }
sanitize_args := if os() == "macos" { "-Dc_args='-isysroot " + sanitize_sdk + "' -Dc_link_args='-isysroot " + sanitize_sdk + "' -Dobjc_args='-isysroot " + sanitize_sdk + " -fno-objc-msgsend-class-selector-stubs' -Dobjc_link_args='-isysroot " + sanitize_sdk + "'" } else { "" }

# List the recipes.
default:
    @just --list

# Configure any build directory that does not exist yet.
setup:
    [ -d {{build_dir}} ] || meson setup {{build_dir}} -Dwerror=true
    [ -d {{sanitize_dir}} ] || {{sanitize_env}} meson setup {{sanitize_dir}} -Dwerror=true -Db_sanitize=address,undefined -Db_lundef=false {{sanitize_args}}
    [ -d {{release_dir}} ] || meson setup {{release_dir}} --buildtype=release

# Build the library and tools.
build: setup
    meson compile -C {{build_dir}}

# Run the tests.
test: build
    meson test -C {{build_dir}} --print-errorlogs

# Run the full quality gate.
check: setup
    meson compile -C {{build_dir}}
    meson test -C {{build_dir}} --print-errorlogs
    meson compile -C {{sanitize_dir}}
    meson test -C {{sanitize_dir}} --print-errorlogs
    ninja -C {{build_dir}} clang-format-check

# Rewrite the sources in GNU style.
fmt: setup
    ninja -C {{build_dir}} clang-format

# Measure GPU bandwidth and the llama.cpp Metal and mlx-lm baselines.
bench: setup
    meson compile -C {{release_dir}}
    {{release_dir}}/tools/gpu_bw
    @echo "llama.cpp $(git -C bench/llama.cpp rev-parse --short HEAD), Metal"
    {{llama_bench_metal}} {{gguf_models}} -p 512 -n 128 -r 5 -o md
    @cd tools && uv run python -c "import mlx.core, mlx_lm; print('mlx', mlx.core.__version__, 'mlx-lm', mlx_lm.__version__)"
    for model in {{mlx_models}}; do echo "$model"; (cd tools && uv run python -m mlx_lm.benchmark --model "../$model" -p 512 -g 128 -n 5); done

# Measure CPU bandwidth and the llama.cpp CPU baseline.
bench-cpu: setup
    meson compile -C {{release_dir}}
    {{release_dir}}/tools/bw
    @echo "llama.cpp $(git -C bench/llama.cpp rev-parse --short HEAD), CPU"
    {{llama_bench_cpu}} {{gguf_models}} -t 1,4,8,10 -p 512 -n 128 -r 5 -o md

# Remove the build directories.
clean:
    rm -rf build
