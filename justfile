build_dir := "build/default"
sanitize_dir := "build/sanitize"
release_dir := "build/release"
llama_bench := "bench/llama.cpp/build/bin/llama-bench"
bench_models := "-m models/LFM2.5-350M-Q8_0.gguf -m models/LFM2.5-1.2B-Instruct-Q8_0.gguf"

# List the recipes.
default:
    @just --list

# Configure any build directory that does not exist yet.
setup:
    [ -d {{build_dir}} ] || meson setup {{build_dir}} -Dwerror=true
    [ -d {{sanitize_dir}} ] || meson setup {{sanitize_dir}} -Dwerror=true -Db_sanitize=address,undefined -Db_lundef=false
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

# Measure memory bandwidth and the llama.cpp baseline.
bench: setup
    meson compile -C {{release_dir}}
    {{release_dir}}/tools/bw
    @echo "llama.cpp $(git -C bench/llama.cpp rev-parse --short HEAD)"
    {{llama_bench}} {{bench_models}} -t 1,4,8,10 -p 512 -n 128 -r 5 -o md

# Remove the build directories.
clean:
    rm -rf build
