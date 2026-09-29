"""Measure candle's prefill and decode speed with its quantized LFM2 example.

Usage:

    uv run python candle_bench.py --binary ../bench/candle/target/release/examples/quantized-lfm2 \
        --model ../models/LFM2.5-350M-Q8_0.gguf \
        --tokenizer ../models/LFM2.5-350M-MLX-8bit/tokenizer.json

The example runs one prompt per process, so the script runs it once to
warm the shader cache and then `--reps` more times. Each run decodes
greedily with no repeat penalty, the way bobcat-bench decodes. The script
prints the mean and sample standard deviation of each rate.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import statistics
import subprocess

import tokenizers

PROMPT_RE = re.compile(r"(\d+) prompt tokens processed: ([\d.]+) token/s")
DECODE_RE = re.compile(r"(\d+) tokens generated: ([\d.]+) token/s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark candle's LFM2 example.")
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--tokenizer", required=True, type=pathlib.Path)
    parser.add_argument("--prompt", type=int, default=512, help="prompt tokens")
    parser.add_argument("--generate", type=int, default=128, help="generated tokens")
    parser.add_argument("--reps", type=int, default=5)
    return parser.parse_args()


def make_prompt(tokenizer: tokenizers.Tokenizer, n_tokens: int) -> str:
    """Return a prompt that encodes to exactly `n_tokens` tokens.

    The prompt counts upward, so greedy decoding keeps counting and never
    stops early at an end-of-sequence token.
    """
    counting = ", ".join(str(i) for i in range(1, 4 * n_tokens))
    ids = tokenizer.encode(counting, add_special_tokens=False).ids
    for length in range(n_tokens, 0, -1):
        prompt = tokenizer.decode(ids[:length])
        if len(tokenizer.encode(prompt, add_special_tokens=True).ids) == n_tokens:
            return prompt
    raise ValueError(f"no counting prompt encodes to {n_tokens} tokens")


def run_once(args: argparse.Namespace, prompt: str) -> tuple[float, float]:
    """Run the example once and return its prefill and decode rates."""
    result = subprocess.run(
        [
            str(args.binary),
            "--model",
            str(args.model),
            "--tokenizer",
            str(args.tokenizer),
            "--prompt",
            prompt,
            # The first token comes from the prefill, and the example counts only the rest as
            # generated.
            "--sample-len",
            str(args.generate + 1),
            "--temperature",
            "0",
            "--repeat-penalty",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    prefill = PROMPT_RE.search(result.stdout)
    decode = DECODE_RE.search(result.stdout)
    if prefill is None or decode is None:
        raise RuntimeError(f"unexpected output from {args.binary}:\n{result.stdout}")
    if int(prefill.group(1)) != args.prompt or int(decode.group(1)) != args.generate:
        raise RuntimeError(
            f"ran {prefill.group(1)} prompt and {decode.group(1)} generated tokens"
        )
    return float(prefill.group(2)), float(decode.group(2))


def main() -> None:
    args = parse_args()
    tokenizer = tokenizers.Tokenizer.from_file(str(args.tokenizer))
    prompt = make_prompt(tokenizer, args.prompt)

    run_once(args, prompt)
    runs = [run_once(args, prompt) for _ in range(args.reps)]

    print(
        f"{args.model}, {args.prompt} prompt and {args.generate} generated tokens, "
        f"{args.reps} runs"
    )
    for label, rates in (
        ("prefill", [r[0] for r in runs]),
        ("decode", [r[1] for r in runs]),
    ):
        spread = statistics.stdev(rates) if len(rates) > 1 else 0.0
        print(f"{label:<8} {statistics.mean(rates):9.2f} ± {spread:.2f} tokens/s")


if __name__ == "__main__":
    main()
