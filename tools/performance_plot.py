"""Draw the README's performance figure from the benchmark runs in docs/performance.

Usage:

    uv run python performance_plot.py ../docs/performance/lfm2.5-2.6b.csv

`performance_bench.py` writes the CSV, one row per engine and round. The script writes an SVG and
a PNG next to the CSV, with one panel for prompt processing and one for generation, both in
tokens per second. Each bar is the median run of an engine, and each red error bar spans the
slowest to the fastest run. Engines appear in the order of their first row in the CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from performance_bench import Run

# A fixed salt gives the SVG elements the same ids on every run, so an unchanged figure has an
# unchanged file.
matplotlib.rcParams["svg.hashsalt"] = "42"
# The figure follows the look of a USENIX systems paper: the Times-like STIX serif that ships with
# matplotlib, a closed frame, and inward ticks on a numeric y axis repeated on the right.
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
        # SVG text stays text, so a browser draws it as sharp as the page text.
        "svg.fonttype": "none",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": True,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "patch.linewidth": 0.8,
    }
)
# A figure 6.4 inches wide displays at 614 px in a browser, the width of a GitHub README column.
# The panels stack, so each spans the full width and the engine names fit level under the bars.
FIGURE_SIZE_INCHES = (6.4, 4.4)
PNG_DPI = 300
BAR_WIDTH = 0.6
BAR_EDGE_COLOR = "#000000"
# bobcat's bars are filled black, and every other engine's bars are open.
BOBCAT_FACE_COLOR = "#000000"
BASELINE_FACE_COLOR = "#ffffff"
# Every error bar is ANSI red, so the ranges stand out against both open and filled bars.
ERROR_BAR_COLOR = "#ff0000"
SURFACE_COLOR = "#ffffff"
TEXT_COLOR = "#000000"
# A solid light grid stays sharp on a screen, where a thin dotted line breaks into uneven pixels.
GRID_COLOR = "#d9d9d9"
METRICS = {
    "pp512_tok_s": "Prompt Processing (tokens/s)",
    "tg128_tok_s": "Generation (tokens/s)",
}


def summarize(runs: pl.DataFrame) -> pl.DataFrame:
    """Return the median, fastest, and slowest run of each metric per engine, in CSV order."""
    first_row = pl.col("row").min().alias("first_row")
    statistics = [
        expression
        for metric in METRICS
        for expression in (
            pl.col(metric).median().alias(f"{metric}_median"),
            pl.col(metric).min().alias(f"{metric}_min"),
            pl.col(metric).max().alias(f"{metric}_max"),
        )
    ]
    numbered = runs.with_row_index("row")
    by_engine = numbered.group_by("engine")
    summary = by_engine.agg(first_row, *statistics)
    return summary.sort("first_row")


def draw_panel(axes: Axes, summary: pl.DataFrame, metric: str) -> None:
    """Draw the median of each engine as a bar and its fastest to slowest run as an error bar."""
    engines = summary["engine"].to_list()
    medians = summary[f"{metric}_median"].to_list()
    minimums = summary[f"{metric}_min"].to_list()
    maximums = summary[f"{metric}_max"].to_list()
    below = [median - minimum for median, minimum in zip(medians, minimums, strict=True)]
    above = [maximum - median for median, maximum in zip(medians, maximums, strict=True)]
    positions = list(range(len(engines)))
    face_colors = [
        BOBCAT_FACE_COLOR if engine == "bobcat" else BASELINE_FACE_COLOR for engine in engines
    ]
    axes.set_facecolor(SURFACE_COLOR)
    axes.bar(
        positions,
        medians,
        BAR_WIDTH,
        color=face_colors,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=1.0,
        zorder=2,
    )
    axes.errorbar(
        positions,
        medians,
        yerr=[below, above],
        linestyle="none",
        ecolor=ERROR_BAR_COLOR,
        elinewidth=1.2,
        capsize=3,
        capthick=1.2,
        zorder=3,
    )
    axes.set_xticks(positions)
    axes.set_xticklabels(engines)
    axes.set_ylabel(METRICS[metric], color=TEXT_COLOR)
    # Bars start at 0 on a linear axis, so a bar half as tall shows half the rate.
    axes.set_ylim(bottom=0)
    axes.grid(
        axis="y",
        which="major",
        color=GRID_COLOR,
        linestyle="-",
        linewidth=0.8,
        zorder=0,
    )
    # The engines are categories, and the bars already mark where each engine sits.
    axes.tick_params(axis="x", which="both", bottom=False, top=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the README's performance figure.")
    parser.add_argument("csv", type=Path, help="the benchmark runs")
    arguments = parser.parse_args()
    csv_path: Path = arguments.csv
    with csv_path.open(newline="") as file:
        rows = [Run.model_validate(row).model_dump() for row in csv.DictReader(file)]
    runs = pl.DataFrame(rows)
    summary = summarize(runs)

    figure = Figure(figsize=FIGURE_SIZE_INCHES, facecolor=SURFACE_COLOR)
    panels = figure.subplots(len(METRICS), 1)
    for axes, metric in zip(panels, METRICS, strict=True):
        draw_panel(axes, summary, metric)
    # A browser draws the SVG text in its own serif, which can run a few pixels past the STIX
    # edges, so the padding keeps that text inside the figure.
    figure.tight_layout(pad=0.8)
    svg_path = csv_path.with_suffix(".svg")
    figure.savefig(svg_path, facecolor=SURFACE_COLOR, metadata={"Date": None})
    png_path = csv_path.with_suffix(".png")
    figure.savefig(png_path, facecolor=SURFACE_COLOR, dpi=PNG_DPI)


if __name__ == "__main__":
    main()
