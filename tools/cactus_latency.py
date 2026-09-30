"""Measure warmed Cactus greedy latency on a pre-tokenized prompt.

Run `uv run cactus_latency.py LIBRARY BUNDLE --prompt 512 --generate 128 --reps 5` from
tools/, where LIBRARY is Cactus's `libcactus_engine.dylib` and BUNDLE is a converted Cactus
model directory. CSV rows go to stdout with the bobcat-bench latency schema.

The script calls `cactus_benchmark_tokens`, the C entry point behind `cactus benchmark`. It
takes token ids, decodes greedily, and never stops at EOS. Cactus times each call with its
own clock. TTFT runs from the start of the call to the first sampled token, and the total
runs to the last one. Model loading and prompt construction precede timing. Cloud telemetry
and cloud handoff are disabled before the library loads.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

PROMPT_TOKEN = 1000
HASH_MASK = (1 << 64) - 1
RESPONSE_BYTES = 1 << 20


class BenchmarkResponse(BaseModel):
    success: bool
    time_to_first_token_ms: float
    total_time_ms: float
    prompt_tokens: int
    completion_tokens: int
    completion_token_ids: list[int]


@dataclass(frozen=True)
class Measurement:
    ttft_ms: float
    tpot_ms: float
    end_to_end_ms: float
    token_hash: int


def positive_int(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return count


def load_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.cactus_set_backend.argtypes = [ctypes.c_char_p]
    library.cactus_set_backend.restype = ctypes.c_int
    library.cactus_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool]
    library.cactus_init.restype = ctypes.c_void_p
    library.cactus_get_last_error.argtypes = []
    library.cactus_get_last_error.restype = ctypes.c_char_p
    library.cactus_benchmark_tokens.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.cactus_benchmark_tokens.restype = ctypes.c_int
    library.cactus_destroy.argtypes = [ctypes.c_void_p]
    library.cactus_destroy.restype = None
    library.cactus_telemetry_shutdown.argtypes = []
    library.cactus_telemetry_shutdown.restype = None
    return library


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--prompt", type=positive_int, default=512)
    parser.add_argument("--generate", type=positive_int, default=128)
    parser.add_argument("--reps", type=positive_int, default=5)
    parser.add_argument("--backend", choices=["metal", "cpu"], default="metal")
    args = parser.parse_args()
    if args.generate < 2:
        parser.error("--generate needs at least two tokens")
    if not (args.bundle / "config.txt").is_file():
        parser.error(f"{args.bundle} has no config.txt")

    # Cactus reads these flags when it initializes a model.
    os.environ["CACTUS_NO_CLOUD_TELE"] = "1"
    os.environ["CACTUS_DISABLE_CLOUD_HANDOFF"] = "1"
    library = load_library(args.library)
    # Graph nodes take the selected backend when the bundle loads, so selection comes first.
    if library.cactus_set_backend(args.backend.encode()) != 0:
        raise RuntimeError(f"Cactus backend {args.backend} is unavailable")
    model = library.cactus_init(str(args.bundle).encode(), None, False)
    if not model:
        raise RuntimeError(f"cactus_init failed: {library.cactus_get_last_error().decode()}")
    prompt = (ctypes.c_uint32 * args.prompt)(*([PROMPT_TOKEN] * args.prompt))
    response = ctypes.create_string_buffer(RESPONSE_BYTES)

    def measure() -> Measurement:
        written = library.cactus_benchmark_tokens(
            model, prompt, args.prompt, args.generate, response, RESPONSE_BYTES
        )
        if written < 0:
            raise RuntimeError(f"cactus_benchmark_tokens failed: {response.value.decode()}")
        result = BenchmarkResponse.model_validate_json(response.value)
        if not result.success or result.prompt_tokens != args.prompt:
            raise RuntimeError(f"unexpected benchmark response: {response.value.decode()}")
        count = len(result.completion_token_ids)
        if result.completion_tokens != args.generate or count != args.generate:
            raise RuntimeError(f"expected {args.generate} outputs, got {count}")
        token_hash = 0
        for token in result.completion_token_ids:
            token_hash = (token_hash * 1000003 + token) & HASH_MASK
        first_ms = result.time_to_first_token_ms
        total_ms = result.total_time_ms
        return Measurement(first_ms, (total_ms - first_ms) / (count - 1), total_ms, token_hash)

    print("Running warmup..", file=sys.stderr)
    measure()
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "engine",
            "run",
            "prompt_tokens",
            "generated_tokens",
            "stream_chunk",
            "ttft_ms",
            "tpot_ms",
            "end_to_end_ms",
            "token_hash",
        ]
    )
    for run in range(1, args.reps + 1):
        result = measure()
        writer.writerow(
            [
                "cactus",
                run,
                args.prompt,
                args.generate,
                1,
                f"{result.ttft_ms:.6f}",
                f"{result.tpot_ms:.6f}",
                f"{result.end_to_end_ms:.6f}",
                result.token_hash,
            ]
        )
    library.cactus_destroy(model)
    library.cactus_telemetry_shutdown()


if __name__ == "__main__":
    main()
