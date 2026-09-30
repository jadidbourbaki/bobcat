"""Measure warmed MLX greedy streaming latency on a pre-tokenized prompt.

Run `uv run mlx_latency.py MODEL --prompt 512 --generate 128 --reps 5`
from tools/. CSV rows go to stdout with the bobcat-bench latency schema.
Model loading and prompt construction precede timing. Timing includes
MLX's production streaming generator and its text detokenizer.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm import load, stream_generate

PROMPT_TOKEN = 1000
HASH_MASK = (1 << 64) - 1


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--prompt", type=positive_int, default=512)
    parser.add_argument("--generate", type=positive_int, default=128)
    parser.add_argument("--reps", type=positive_int, default=5)
    args = parser.parse_args()
    if args.generate < 2:
        parser.error("--generate needs at least two tokens")

    loaded = load(args.model)
    if len(loaded) != 2:
        raise RuntimeError("expected a model and tokenizer")
    model, tokenizer = loaded
    if tokenizer.vocab_size <= PROMPT_TOKEN:
        parser.error("prompt token 1000 is outside the vocabulary")
    # Fixed output counts let the engines compare without early EOS termination.
    tokenizer._eos_token_ids = set()
    prompt = mx.array([PROMPT_TOKEN] * args.prompt, dtype=mx.uint32)
    mx.eval(prompt)

    def measure() -> Measurement:
        mx.synchronize()
        start = time.perf_counter()
        first_ms = 0.0
        total_ms = 0.0
        token_hash = 0
        count = 0
        for response in stream_generate(model, tokenizer, prompt, max_tokens=args.generate):
            total_ms = (time.perf_counter() - start) * 1000
            if count == 0:
                first_ms = total_ms
            token_hash = (token_hash * 1000003 + response.token) & HASH_MASK
            count += 1
        # MLX submits a lookahead token. Drain it outside the measured emission
        # interval so its GPU work cannot spill into the next trial.
        mx.synchronize()
        if count != args.generate:
            raise RuntimeError(f"expected {args.generate} outputs, got {count}")
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
                "mlx-lm",
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


if __name__ == "__main__":
    main()
