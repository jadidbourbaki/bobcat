"""Measure warmed ExecuTorch greedy streaming latency on a pre-tokenized prompt.

Run `../bench/executorch/.venv/bin/python executorch_latency.py MODEL.pte --prompt 512
--generate 128 --reps 5` from tools/. The script runs in ExecuTorch's own environment under
bench/executorch, because tools/.venv does not carry the ExecuTorch runtime. CSV rows go to
stdout with the bobcat-bench latency schema. Each run loads a fresh program before timing,
because the exported model keeps its KV cache and convolution state in mutable buffers that
ExecuTorch cannot reset. Timing covers the prompt call, every decode call, and a host argmax
over the returned logits.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from executorch.runtime import Runtime  # ty: ignore[unresolved-import]
from pydantic import BaseModel, PositiveInt

PROMPT_TOKEN = 1000
HASH_MASK = (1 << 64) - 1


class Method(Protocol):
    def execute(self, inputs: Sequence[object]) -> Sequence[object]: ...


class Program(Protocol):
    def load_method(self, name: str) -> Method | None: ...


class ModelLimits(BaseModel):
    vocab_size: PositiveInt
    max_seq_len: PositiveInt
    max_context_len: PositiveInt


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


def load_method(program: Program, name: str) -> Method:
    method = program.load_method(name)
    if method is None:
        raise RuntimeError(f"the program has no {name} method")
    return method


def read_scalar(program: Program, name: str) -> int:
    outputs = load_method(program, name).execute([])
    if len(outputs) != 1 or not isinstance(outputs[0], int):
        raise RuntimeError(f"{name} returned {outputs!r}, expected one integer")
    return outputs[0]


def greedy_token(outputs: Sequence[object]) -> int:
    logits = outputs[0]
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"expected logits as a tensor, got {type(logits).__name__}")
    return int(logits.reshape(-1, logits.shape[-1])[-1].argmax())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--prompt", type=positive_int, default=512)
    parser.add_argument("--generate", type=positive_int, default=128)
    parser.add_argument("--reps", type=positive_int, default=5)
    args = parser.parse_args()
    if args.generate < 2:
        parser.error("--generate needs at least two tokens")

    runtime = Runtime.get()
    probe: Program = runtime.load_program(args.model)
    limits = ModelLimits(
        vocab_size=read_scalar(probe, "get_vocab_size"),
        max_seq_len=read_scalar(probe, "get_max_seq_len"),
        max_context_len=read_scalar(probe, "get_max_context_len"),
    )
    del probe
    if limits.vocab_size <= PROMPT_TOKEN:
        parser.error("prompt token 1000 is outside the vocabulary")
    # The export bounds one call to max_seq_len - 1 tokens.
    if args.prompt >= limits.max_seq_len:
        parser.error(f"a {args.prompt}-token prompt needs max_seq_len above {args.prompt}")
    if args.prompt + args.generate > limits.max_context_len:
        parser.error(f"the export holds {limits.max_context_len} tokens of context")
    prompt = torch.full((1, args.prompt), PROMPT_TOKEN, dtype=torch.long)
    prompt_position = torch.zeros(1, dtype=torch.long)

    def measure() -> Measurement:
        # Release the previous program first, since its delegate still holds a KV cache.
        gc.collect()
        forward = load_method(runtime.load_program(args.model), "forward")
        start = time.perf_counter()
        token = greedy_token(forward.execute([prompt, prompt_position]))
        first_ms = (time.perf_counter() - start) * 1000
        token_hash = token
        for position in range(args.prompt, args.prompt + args.generate - 1):
            step_inputs = [
                torch.tensor([[token]], dtype=torch.long),
                torch.tensor([position], dtype=torch.long),
            ]
            token = greedy_token(forward.execute(step_inputs))
            token_hash = (token_hash * 1000003 + token) & HASH_MASK
        total_ms = (time.perf_counter() - start) * 1000
        tpot_ms = (total_ms - first_ms) / (args.generate - 1)
        return Measurement(first_ms, tpot_ms, total_ms, token_hash)

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
                "executorch",
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
