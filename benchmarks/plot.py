"""Renders benchmark results as a chart (PNG) and a table (markdown).

Run from the repository root with the result files of both engines:

    uv run python -m benchmarks.plot latency benchmarks/results/latency-*-cpu.json \
        --output benchmarks/results/latency-cpu
"""

import argparse
import itertools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, NullLocator

# Engines in the order they are drawn and listed. Each keeps its color in every chart.
ENGINE_COLORS = {"cached": "#2a78d6", "naive": "#eb6834"}
ENGINE_LABELS = {"cached": "KV cache", "naive": "naive (no cache)"}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED_TEXT = "#52514e"
GRID = "#e4e3df"
# Settings that must be equal for two runs to be comparable.
COMPARED_SETTINGS = ("model", "revision", "prompt_tokens")
COMPARED_ENVIRONMENT = ("device", "dtype", "cpu", "torch")

Result = dict[str, Any]


def load_latency_results(paths: Sequence[Path]) -> dict[str, Result]:
    """Reads latency result files, one per engine.

    Returns:
        The results keyed by engine, in the order of ENGINE_COLORS.

    Raises:
        ValueError: If a file is not a latency result, an engine appears twice or is
            unknown, or the runs differ in model, prompt, device or software.
    """
    results: dict[str, Result] = {}
    for path in paths:
        result = json.loads(path.read_text())
        engine = result.get("engine")
        if result.get("benchmark") != "latency" or engine not in ENGINE_COLORS:
            raise ValueError(f"{path} is not a latency result of a known engine")
        if engine in results:
            raise ValueError(f"{path} is a second result for the {engine} engine")
        results[engine] = result
    if not results:
        raise ValueError("no result files given")

    first = next(iter(results.values()))
    for result in results.values():
        for key in COMPARED_SETTINGS:
            if result[key] != first[key]:
                raise ValueError(f"the runs differ in {key}: {first[key]} and {result[key]}")
        for key in COMPARED_ENVIRONMENT:
            ours, theirs = first["environment"][key], result["environment"][key]
            if ours != theirs:
                raise ValueError(f"the runs differ in {key}: {ours} and {theirs}")
    return {engine: results[engine] for engine in ENGINE_COLORS if engine in results}


def medians(result: Result, metric: str) -> dict[int, float]:
    """Returns the median of a metric for each output length of one result."""
    return {entry["output_tokens"]: entry["median"][metric] for entry in result["results"]}


def growth(values: dict[int, float]) -> dict[int, float]:
    """Returns how many times larger each value is than the one at the previous length.

    With lengths that double, a factor of 2 means linear growth and 4 means quadratic.
    """
    lengths = sorted(values)
    return {
        length: values[length] / values[previous]
        for previous, length in itertools.pairwise(lengths)
    }


def describe(results: dict[str, Result]) -> str:
    """Returns one line naming the model, the machine and the method of the runs."""
    result = next(iter(results.values()))
    environment = result["environment"]
    model = str(result["model"]).rsplit("/", 1)[-1]
    hardware = environment.get("gpu") or environment["cpu"]
    repeats = result["repeats"]
    runs = "1 run" if repeats == 1 else f"median of {repeats} runs"
    return (
        f"{model}, {environment['device']} {environment['dtype']}, {hardware}, "
        f"{result['prompt_tokens']}-token prompt, greedy, {runs}"
    )


def latency_table(results: dict[str, Result]) -> str:
    """Renders latency results as a markdown table with one row per output length.

    The "x previous" columns give the total time relative to the row above.
    """
    speeds = {engine: medians(result, "output_tokens_per_s") for engine, result in results.items()}
    times = {engine: medians(result, "total_s") for engine, result in results.items()}
    factors = {engine: growth(values) for engine, values in times.items()}

    header = ["output tokens"]
    for engine in results:
        header += [f"{engine} tokens/s", f"{engine} total s", f"{engine} x previous"]
    with_speedup = "cached" in results and "naive" in results
    if with_speedup:
        header.append("speedup")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    for length in sorted({length for values in times.values() for length in values}):
        row = [str(length)]
        for engine in results:
            row += [
                _format(speeds[engine].get(length), ".1f"),
                _format(times[engine].get(length), ".2f"),
                _format(factors[engine].get(length), ".2f"),
            ]
        if with_speedup:
            both = length in times["cached"] and length in times["naive"]
            speedup = times["naive"][length] / times["cached"][length] if both else None
            row.append(_format(speedup, ".1f"))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def plot_latency(results: dict[str, Result], path: Path) -> Path:
    """Draws decoding speed and total time against output length, one line per engine."""
    figure = Figure(figsize=(10.5, 4.6), dpi=160, facecolor=SURFACE)
    speed_axes, time_axes = figure.subplots(1, 2)
    lengths = sorted({entry["output_tokens"] for r in results.values() for entry in r["results"]})

    for engine, result in results.items():
        speeds = medians(result, "output_tokens_per_s")
        times = medians(result, "total_s")
        _draw_series(speed_axes, speeds, engine, f"{speeds[max(speeds)]:.1f}")
        _draw_series(time_axes, times, engine, f"{times[max(times)]:,.1f} s")

    _style_axes(speed_axes, lengths, "Output tokens per second")
    speed_axes.set_ylim(bottom=0)
    speed_axes.legend(frameon=False, labelcolor=TEXT, loc="center left", fontsize=9)

    _style_axes(time_axes, lengths, "Total time in seconds (log scale)")
    time_axes.set_yscale("log")
    time_axes.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    time_axes.yaxis.set_minor_locator(NullLocator())
    # The longest doubling states the growth in numbers: 2 is linear, 4 is quadratic.
    notes: list[str] = []
    for engine, result in results.items():
        factors = growth(medians(result, "total_s"))
        if factors:
            notes.append(f"{ENGINE_LABELS[engine]} x{factors[max(factors)]:.1f}")
    if len(lengths) > 1 and notes:
        doubling = f"{lengths[-2]} to {lengths[-1]} tokens: "
        time_axes.text(
            0.0, 1.03, doubling + ", ".join(notes), transform=time_axes.transAxes,
            color=MUTED_TEXT, fontsize=9, va="bottom",
        )  # fmt: skip

    figure.suptitle(
        "Decoding one request with and without the KV cache",
        x=0.06, y=0.97, ha="left", color=TEXT, fontsize=13, fontweight="bold",
    )  # fmt: skip
    figure.text(0.06, 0.885, describe(results), color=MUTED_TEXT, fontsize=9, ha="left")
    figure.subplots_adjust(left=0.06, right=0.97, top=0.74, bottom=0.12, wspace=0.16)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    return path


def _draw_series(axes: Axes, values: dict[int, float], engine: str, end_label: str) -> None:
    lengths = sorted(values)
    axes.plot(
        lengths,
        [values[length] for length in lengths],
        color=ENGINE_COLORS[engine],
        label=ENGINE_LABELS[engine],
        linewidth=2,
        marker="o",
        markersize=6.5,
        # A ring in the surface color keeps markers apart where lines meet.
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    # The value at the longest output, next to the end of the line. The line carries the
    # color; the text stays in the text color.
    axes.annotate(
        end_label,
        (lengths[-1], values[lengths[-1]]),
        xytext=(8, 0),
        textcoords="offset points",
        color=TEXT,
        fontsize=9,
        va="center",
    )


def _style_axes(axes: Axes, lengths: Sequence[int], title: str) -> None:
    axes.set_facecolor(SURFACE)
    axes.set_title(title, loc="left", color=TEXT, fontsize=10, pad=24)
    axes.set_xlabel("output tokens", color=MUTED_TEXT, fontsize=9)
    # Lengths double from point to point, so a log scale spaces them evenly.
    axes.set_xscale("log", base=2)
    axes.set_xticks(list(lengths), [str(length) for length in lengths])
    axes.xaxis.set_minor_locator(NullLocator())
    # Room on the right for the labels at the ends of the lines.
    axes.set_xlim(lengths[0] / 1.25, lengths[-1] * 2.1)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED_TEXT, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)


def _format(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Renders benchmark results.")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    latency = commands.add_parser("latency", help="decoding speed against output length")
    latency.add_argument("results", type=Path, nargs="+", help="latency result files")
    latency.add_argument(
        "--output", type=Path, required=True, help="path of the chart and table, without suffix"
    )
    args = parser.parse_args(argv)

    results = load_latency_results(args.results)
    chart = plot_latency(results, args.output.with_name(args.output.name + ".png"))
    table = args.output.with_name(args.output.name + ".md")
    table.write_text(f"{describe(results)}\n\n{latency_table(results)}")
    print(f"saved {chart}")
    print(f"saved {table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
