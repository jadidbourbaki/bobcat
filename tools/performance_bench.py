"""Measure the time to first token and the time per output token of bobcat and six baselines.

Usage:

    uv run python performance_bench.py --model 2.6b

Every engine that runs the model runs it on a 512-token prompt and then generates 128 tokens
greedily. The CSV goes to `docs/performance/lfm2.5-MODEL.csv`.
bobcat, llama.cpp, and mlx-lm run through matched streaming harnesses. ExecuTorch and Cactus
run through their harnesses in tools/. mistral.rs and candle run through their own benchmark
tools. Each round runs every engine once, after the tool's own warmup, and the order rotates
between rounds, so background load falls on every engine alike. The script writes one CSV row
per engine and round, and writes the machine description next to the CSV. `eval.md` in the
output directory describes each engine's weights and timing.
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
TOOLS = ROOT / "tools"
MODELS = ROOT / "models"
EXECUTORCH = ROOT / "bench/executorch"
CACTUS = ROOT / "bench/cactus"
CACTUS_LIBRARY = CACTUS / "cactus-engine/build/libcactus_engine.dylib"
PROMPT_TOKENS = 512
GENERATED_TOKENS = 128
MISTRALRS_TTFT = re.compile(r"┆\s*([\d.]+) ms TTFT")
MISTRALRS_TPOT = re.compile(r"┆\s*([\d.]+) ms TPOT")
CANDLE_RATE = re.compile(r"^(prefill|decode)\s+([\d.]+) ±", re.MULTILINE)


class Run(BaseModel):
    """One measurement of one engine, as a row of the CSV."""

    engine: str
    version: str
    run: PositiveInt
    ttft_ms: PositiveFloat
    tpot_ms: PositiveFloat


class LatencyRow(BaseModel):
    """A row of the latency CSV that the streaming harnesses print."""

    engine: str
    ttft_ms: PositiveFloat
    tpot_ms: PositiveFloat


@dataclass(frozen=True)
class Weights:
    """The weights each engine reads for one model. An engine without weights skips the model."""

    gguf: Path
    mlx: Path
    executorch: Path | None
    cactus: Path | None
    mistral_rs: bool
    candle: bool


WEIGHTS = {
    "350m": Weights(
        gguf=MODELS / "LFM2.5-350M-QAD-Q4_0.gguf",
        mlx=MODELS / "LFM2.5-350M-MLX-4bit",
        executorch=EXECUTORCH / "lfm2_5_350m_mlx_4w.pte",
        cactus=CACTUS / "weights/lfm2.5-350m-cq4",
        mistral_rs=True,
        candle=True,
    ),
    "1.2b": Weights(
        gguf=MODELS / "LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf",
        mlx=MODELS / "LFM2.5-1.2B-Instruct-MLX-4bit",
        executorch=EXECUTORCH / "lfm2_5_1_2b_mlx_4w.pte",
        cactus=CACTUS / "weights/lfm2.5-1.2b-cq4",
        mistral_rs=True,
        candle=True,
    ),
    "2.6b": Weights(
        gguf=MODELS / "LFM2.5-2.6B-QAD-Q4_0.gguf",
        mlx=MODELS / "LFM2.5-2.6B-MLX-4bit",
        executorch=EXECUTORCH / "lfm2_5_2_6b_mlx_4w.pte",
        cactus=CACTUS / "weights/lfm2.5-2.6b-cq4",
        mistral_rs=True,
        candle=True,
    ),
    "8b": Weights(
        gguf=MODELS / "LFM2.5-8B-A1B-Q4_0.gguf",
        mlx=MODELS / "LFM2.5-8B-A1B-MLX-4bit",
        # ExecuTorch's LFM2 model has no mixture-of-experts layers. mistral.rs decodes the 8B at
        # about 0.2 tokens per second, and candle's LFM2 example reads only dense models.
        executorch=None,
        cactus=CACTUS / "weights/lfm2.5-8b-cq4",
        mistral_rs=False,
        candle=False,
    ),
}


@dataclass(frozen=True)
class Engine:
    """An engine, its version, and the command that measures it once on some weights."""

    name: str
    version: str
    measure: Callable[[Weights], tuple[float, float]]


def output(command: list[str], cwd: Path = ROOT) -> str:
    """Return the standard output of `command`, which must succeed."""
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True).stdout


def git_version(path: Path) -> str:
    return output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"]).strip()


def latency(command: list[str], engine: str, cwd: Path = ROOT) -> tuple[float, float]:
    """Run a harness that prints the latency CSV for one run and return its TTFT and TPOT."""
    rows = [
        LatencyRow.model_validate(row)
        for row in csv.DictReader(io.StringIO(output(command, cwd)))
        if row.get("engine") == engine
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one {engine} row from {command}")
    return rows[0].ttft_ms, rows[0].tpot_ms


def bobcat(weights: Weights) -> tuple[float, float]:
    return latency(
        [
            "target/release/bobcat-bench",
            "--latency",
            "--latency-only",
            "-r",
            "1",
            str(weights.gguf),
        ],
        "bobcat",
    )


def llama_cpp(weights: Weights) -> tuple[float, float]:
    return latency(
        [
            "target/llama-latency",
            str(weights.gguf),
            str(PROMPT_TOKENS),
            str(GENERATED_TOKENS),
            "1",
        ],
        "llama.cpp",
    )


def mlx_lm(weights: Weights) -> tuple[float, float]:
    return latency(
        [
            "uv",
            "run",
            "mlx_latency.py",
            str(weights.mlx),
            "--prompt",
            str(PROMPT_TOKENS),
            "--generate",
            str(GENERATED_TOKENS),
            "--reps",
            "1",
        ],
        "mlx-lm",
        TOOLS,
    )


def executorch(weights: Weights) -> tuple[float, float]:
    return latency(
        [
            str(EXECUTORCH / ".venv/bin/python"),
            "executorch_latency.py",
            str(weights.executorch),
            "--prompt",
            str(PROMPT_TOKENS),
            "--generate",
            str(GENERATED_TOKENS),
            "--reps",
            "1",
        ],
        "executorch",
        TOOLS,
    )


def cactus(weights: Weights) -> tuple[float, float]:
    return latency(
        [
            "uv",
            "run",
            "cactus_latency.py",
            str(CACTUS_LIBRARY),
            str(weights.cactus),
            "--prompt",
            str(PROMPT_TOKENS),
            "--generate",
            str(GENERATED_TOKENS),
            "--reps",
            "1",
        ],
        "cactus",
        TOOLS,
    )


def mistral_rs(weights: Weights) -> tuple[float, float]:
    text = output(
        [
            "bench/mistral.rs/target/release/mistralrs",
            "bench",
            "-f",
            str(weights.gguf),
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
    ttft = MISTRALRS_TTFT.search(text)
    tpot = MISTRALRS_TPOT.search(text)
    if ttft is None or tpot is None:
        raise RuntimeError(f"unexpected output from mistralrs bench:\n{text}")
    return float(ttft[1]), float(tpot[1])


def candle(weights: Weights) -> tuple[float, float]:
    text = output(
        [
            "uv",
            "run",
            "python",
            "candle_bench.py",
            "--binary",
            str(ROOT / "bench/candle/target/release/examples/quantized-lfm2"),
            "--model",
            str(weights.gguf),
            "--tokenizer",
            str(weights.mlx / "tokenizer.json"),
            "--prompt",
            str(PROMPT_TOKENS),
            "--generate",
            str(GENERATED_TOKENS),
            "--reps",
            "1",
        ],
        TOOLS,
    )
    rates = {label: float(value) for label, value in CANDLE_RATE.findall(text)}
    if set(rates) != {"prefill", "decode"}:
        raise RuntimeError(f"unexpected output from candle_bench.py:\n{text}")
    # candle reports rates, so the prompt's duration stands in for the time to first token.
    return PROMPT_TOKENS / rates["prefill"] * 1000, 1000 / rates["decode"]


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
    parser = argparse.ArgumentParser(description="Measure the engines' latency.")
    parser.add_argument("--model", required=True, choices=list(WEIGHTS))
    parser.add_argument("--rounds", type=int, default=5)
    arguments = parser.parse_args()
    weights = WEIGHTS[arguments.model]
    out = ROOT / f"docs/performance/lfm2.5-{arguments.model}.csv"

    output(["cargo", "build", "--release", "--locked", "-p", "bobcat-bench"])
    mlx_version = output(
        ["uv", "run", "python", "-c", "import mlx_lm; print(mlx_lm.__version__)"], TOOLS
    ).strip()
    executorch_version = output(
        [
            str(EXECUTORCH / ".venv/bin/python"),
            "-c",
            "import executorch.version; print(executorch.version.__version__)",
        ]
    ).strip()
    supported = {
        "ExecuTorch": weights.executorch is not None,
        "Cactus": weights.cactus is not None,
        "mistral.rs": weights.mistral_rs,
        "candle": weights.candle,
    }
    engines = [
        engine
        for engine in [
            Engine("bobcat", git_version(ROOT), bobcat),
            Engine("mlx-lm", mlx_version, mlx_lm),
            Engine("llama.cpp", git_version(ROOT / "bench/llama.cpp"), llama_cpp),
            Engine("ExecuTorch", executorch_version, executorch),
            Engine("Cactus", git_version(CACTUS), cactus),
            Engine("mistral.rs", git_version(ROOT / "bench/mistral.rs"), mistral_rs),
            Engine("candle", git_version(ROOT / "bench/candle"), candle),
        ]
        if supported.get(engine.name, True)
    ]
    runs: list[Run] = []
    for round_index in range(arguments.rounds):
        shift = round_index % len(engines)
        for engine in engines[shift:] + engines[:shift]:
            ttft_ms, tpot_ms = engine.measure(weights)
            run = Run(
                engine=engine.name,
                version=engine.version,
                run=round_index + 1,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
            )
            print(run.model_dump_json(), flush=True)
            runs.append(run)

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
