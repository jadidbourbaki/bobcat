"""Measure prompt and generation throughput of bobcat and four baseline engines.

Usage:

    uv run python performance_bench.py --out ../docs/performance/lfm2.5-2.6b.csv

Every engine runs through its own benchmark tool with llama-bench's tests: pp512, the rate of
processing a 512-token prompt, and tg128, the rate of generating 128 tokens after that prompt.
The GGUF engines read the same `LFM2.5-2.6B-Q4_K_M.gguf`, and mlx-lm reads Liquid AI's 4-bit
MLX weights. Each round runs every engine once, after the tool's own warmup, and the order
rotates between rounds, so background load falls on every engine alike. The script writes one
CSV row per engine and round, and writes the machine description next to the CSV.
"""

from __future__ import annotations

import argparse
import csv
import io
import platform
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, PositiveFloat, PositiveInt

ROOT = Path(__file__).resolve().parent.parent
GGUF = ROOT / "models/LFM2.5-2.6B-Q4_K_M.gguf"
MLX = ROOT / "models/LFM2.5-2.6B-MLX-4bit"
PROMPT_TOKENS = 512
GENERATED_TOKENS = 128
MISTRALRS_RATE = re.compile(r"^│\s*(TTFT|Decode) \(.*?┆\s*([\d.]+) ±", re.MULTILINE)
# bobcat-bench and candle_bench.py print their rates in the same form.
PREFILL_DECODE_RATE = re.compile(r"^(prefill|decode)\s+([\d.]+) ±", re.MULTILINE)
MLX_RATE = re.compile(r"prompt_tps=([\d.]+), generation_tps=([\d.]+)")


class Run(BaseModel):
    """One measurement of one engine, as a row of the CSV."""

    engine: str
    version: str
    run: PositiveInt
    pp512_tok_s: PositiveFloat
    tg128_tok_s: PositiveFloat


@dataclass(frozen=True)
class Engine:
    """An engine, its version, and the command that measures it once."""

    name: str
    version: str
    measure: Callable[[], tuple[float, float]]


def output(command: list[str], cwd: Path = ROOT) -> str:
    """Return the standard output of `command`, which must succeed."""
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True).stdout


def git_version(path: Path) -> str:
    return output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"]).strip()


def matches(pattern: re.Pattern[str], text: str, command: str) -> dict[str, float]:
    found = {label: float(value) for label, value in pattern.findall(text)}
    if len(found) != 2:
        raise RuntimeError(f"unexpected output from {command}:\n{text}")
    return found


def bobcat() -> tuple[float, float]:
    text = output(
        [
            "target/release/bobcat-bench",
            "-r",
            "1",
            "-p",
            str(PROMPT_TOKENS),
            "-n",
            str(GENERATED_TOKENS),
            str(GGUF),
        ]
    )
    rates = matches(PREFILL_DECODE_RATE, text, "bobcat-bench")
    return rates["prefill"], rates["decode"]


def llama_rate(test: list[str]) -> float:
    text = output(
        [
            "bench/llama.cpp/build-metal/bin/llama-bench",
            "-m",
            str(GGUF),
            *test,
            "-r",
            "1",
            "-o",
            "csv",
        ]
    )
    rows = list(csv.DictReader(io.StringIO(text)))
    if len(rows) != 1:
        raise RuntimeError(f"unexpected output from llama-bench:\n{text}")
    return float(rows[0]["avg_ts"])


def llama_cpp() -> tuple[float, float]:
    prompt = llama_rate(["-p", str(PROMPT_TOKENS), "-n", "0"])
    generation = llama_rate(["-p", "0", "-n", str(GENERATED_TOKENS), "-d", str(PROMPT_TOKENS)])
    return prompt, generation


def mlx_lm() -> tuple[float, float]:
    text = output(
        [
            "uv",
            "run",
            "python",
            "-m",
            "mlx_lm.benchmark",
            "--model",
            str(MLX),
            "-p",
            str(PROMPT_TOKENS),
            "-g",
            str(GENERATED_TOKENS),
            "-n",
            "1",
        ],
        cwd=ROOT / "tools",
    )
    found = MLX_RATE.search(text)
    if found is None:
        raise RuntimeError(f"unexpected output from mlx_lm.benchmark:\n{text}")
    return float(found[1]), float(found[2])


def mistral_rs() -> tuple[float, float]:
    text = output(
        [
            "bench/mistral.rs/target/release/mistralrs",
            "bench",
            "-f",
            str(GGUF),
            "--prompt-len",
            str(PROMPT_TOKENS),
            "--gen-len",
            str(GENERATED_TOKENS),
            "--depth",
            str(PROMPT_TOKENS),
            "--iterations",
            "1",
            "--warmup",
            "1",
        ]
    )
    rates = matches(MISTRALRS_RATE, text, "mistralrs bench")
    return rates["TTFT"], rates["Decode"]


def candle() -> tuple[float, float]:
    text = output(
        [
            "uv",
            "run",
            "python",
            "candle_bench.py",
            "--binary",
            str(ROOT / "bench/candle/target/release/examples/quantized-lfm2"),
            "--model",
            str(GGUF),
            "--tokenizer",
            str(MLX / "tokenizer.json"),
            "--prompt",
            str(PROMPT_TOKENS),
            "--generate",
            str(GENERATED_TOKENS),
            "--reps",
            "1",
        ],
        cwd=ROOT / "tools",
    )
    rates = matches(PREFILL_DECODE_RATE, text, "candle_bench.py")
    return rates["prefill"], rates["decode"]


def mlx_version() -> str:
    return output(
        ["uv", "run", "python", "-c", "import mlx_lm; print(mlx_lm.__version__)"],
        cwd=ROOT / "tools",
    ).strip()


def machine() -> str:
    """Return the chip, cores, memory, and macOS version of this Mac."""
    chip = output(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    cpu_cores = output(["sysctl", "-n", "hw.ncpu"]).strip()
    memory = int(output(["sysctl", "-n", "hw.memsize"])) // 2**30
    gpu = output(["system_profiler", "SPDisplaysDataType"])
    gpu_cores = re.search(r"Total Number of Cores: (\d+)", gpu)
    macos = platform.mac_ver()[0]
    return (
        f"{chip}, {cpu_cores} CPU cores, "
        f"{gpu_cores[1] if gpu_cores else 'unknown'} GPU cores, {memory} GiB memory.\n"
        f"macOS {macos}.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the engines' throughput.")
    parser.add_argument("--out", required=True, type=Path, help="the CSV to write")
    parser.add_argument("--rounds", type=int, default=5)
    arguments = parser.parse_args()
    out: Path = arguments.out

    output(["cargo", "build", "--release", "--locked", "-p", "bobcat-bench"])
    engines = [
        Engine("bobcat", git_version(ROOT), bobcat),
        Engine("llama.cpp", git_version(ROOT / "bench/llama.cpp"), llama_cpp),
        Engine("mlx-lm", mlx_version(), mlx_lm),
        Engine("mistral.rs", git_version(ROOT / "bench/mistral.rs"), mistral_rs),
        Engine("candle", git_version(ROOT / "bench/candle"), candle),
    ]
    runs: list[Run] = []
    for round_index in range(arguments.rounds):
        shift = round_index % len(engines)
        for engine in engines[shift:] + engines[:shift]:
            prompt, generation = engine.measure()
            run = Run(
                engine=engine.name,
                version=engine.version,
                run=round_index + 1,
                pp512_tok_s=prompt,
                tg128_tok_s=generation,
            )
            print(run.model_dump_json(), flush=True)
            runs.append(run)

    # Rows follow the engine order, so the figure keeps bobcat first.
    order = {engine.name: index for index, engine in enumerate(engines)}
    runs.sort(key=lambda run: (order[run.engine], run.run))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(Run.model_fields))
        writer.writeheader()
        writer.writerows(run.model_dump() for run in runs)
    out.with_name("machine.txt").write_text(machine())


if __name__ == "__main__":
    main()
