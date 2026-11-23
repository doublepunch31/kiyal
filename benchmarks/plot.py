"""Renders benchmark results as a chart (PNG) and a table (markdown).

Run from the repository root, with the latency results of both engines, or one throughput,
batching or capacity result:

    uv run python -m benchmarks.plot latency benchmarks/results/latency-*-cpu.json \
        --output benchmarks/results/latency-cpu
    uv run python -m benchmarks.plot throughput benchmarks/results/throughput-static-cpu.json \
        --output benchmarks/results/throughput-cpu
    uv run python -m benchmarks.plot batching benchmarks/results/batching-cpu.json \
        --output benchmarks/results/batching-cpu
    uv run python -m benchmarks.plot capacity benchmarks/results/capacity-cpu.json \
        --output benchmarks/results/capacity-cpu
"""

import argparse
import itertools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator

# Engines in the order they are drawn and listed. Each keeps its color in every chart.
ENGINE_COLORS = {"cached": "#2a78d6", "naive": "#eb6834"}
ENGINE_LABELS = {"cached": "KV cache", "naive": "naive (no cache)"}
# Static batching decodes with the KV cache, so it keeps the cache's color. Continuous
# batching takes the next free hue, so no two engines share a color in any chart.
STATIC_COLOR = ENGINE_COLORS["cached"]
BATCHING_COLORS = {"static": STATIC_COLOR, "continuous": "#1baf7a"}
BATCHING_LABELS = {"static": "static batching", "continuous": "continuous batching"}
# The contiguous pool is continuous batching's pool before paging, so it keeps that color. The
# paged pool takes the next hue of the palette.
CAPACITY_COLORS = {"contiguous": BATCHING_COLORS["continuous"], "paged": "#eda100"}
CAPACITY_LABELS = {"contiguous": "contiguous", "paged": "paged"}
# Without the cache is the paged engine of the capacity chart; with it takes the next slot of
# the palette, magenta.
PREFIX_COLORS = {"off": CAPACITY_COLORS["paged"], "on": "#e87ba4"}
PREFIX_LABELS = {"off": "without prefix caching", "on": "with prefix caching"}
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
    workload = f"{result['prompt_tokens']}-token prompt"
    # Latency results vary the output length; throughput results fix it.
    if "output_tokens" in result:
        workload = (
            f"{result['prompt_tokens']}-token prompts, {result['output_tokens']} output tokens"
        )
    return (
        f"{model}, {environment['device']} {environment['dtype']}, {hardware}, "
        f"{workload}, greedy, {runs}"
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
        color, label = ENGINE_COLORS[engine], ENGINE_LABELS[engine]
        _draw_series(speed_axes, speeds, color, label, f"{speeds[max(speeds)]:.1f}")
        _draw_series(time_axes, times, color, label, f"{times[max(times)]:,.1f} s")

    _style_axes(speed_axes, lengths, "Output tokens per second", "output tokens")
    speed_axes.set_ylim(bottom=0)
    speed_axes.legend(frameon=False, labelcolor=TEXT, loc="center left", fontsize=9)

    _style_axes(time_axes, lengths, "Total time in seconds (log scale)", "output tokens")
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


def load_throughput_result(path: Path) -> Result:
    """Reads a throughput result file.

    Raises:
        ValueError: If the file is not a throughput result.
    """
    result = json.loads(path.read_text())
    if result.get("benchmark") != "throughput":
        raise ValueError(f"{path} is not a throughput result")
    return result


def batch_medians(result: Result, metric: str) -> dict[int, float]:
    """Returns the median of a metric for each batch size of a throughput result."""
    return {entry["batch_size"]: entry["median"][metric] for entry in result["results"]}


def throughput_table(result: Result) -> str:
    """Renders a throughput result as a markdown table with one row per batch size.

    The "x batch" columns compare each batch size with the smallest one.
    """
    speeds = batch_medians(result, "output_tokens_per_s")
    steps = batch_medians(result, "step_s")
    ttfts = batch_medians(result, "ttft_s")
    smallest = min(speeds)
    header = [
        "batch size",
        "output tokens/s",
        f"x batch {smallest}",
        "step ms",
        f"x batch {smallest}",
        "TTFT ms",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for size in sorted(speeds):
        row = [
            str(size),
            f"{speeds[size]:.1f}",
            f"{speeds[size] / speeds[smallest]:.2f}",
            f"{steps[size] * 1e3:.1f}",
            f"{steps[size] / steps[smallest]:.2f}",
            f"{ttfts[size] * 1e3:.0f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def plot_throughput(result: Result, path: Path) -> Path:
    """Draws throughput and the time per decoding step against batch size."""
    figure = Figure(figsize=(10.5, 4.6), dpi=160, facecolor=SURFACE)
    speed_axes, step_axes = figure.subplots(1, 2)
    speeds = batch_medians(result, "output_tokens_per_s")
    steps_ms = {size: step * 1e3 for size, step in batch_medians(result, "step_s").items()}
    sizes = sorted(speeds)
    label = "static batching, KV cache"

    _draw_series(speed_axes, speeds, STATIC_COLOR, label, f"{speeds[sizes[-1]]:,.0f}")
    _style_axes(speed_axes, sizes, "Output tokens per second", "batch size")
    speed_axes.set_ylim(bottom=0)
    _draw_series(step_axes, steps_ms, STATIC_COLOR, label, f"{steps_ms[sizes[-1]]:.1f} ms")
    _style_axes(step_axes, sizes, "Time per decoding step in ms", "batch size")
    step_axes.set_ylim(bottom=0)
    # The trade-off in numbers: how much more throughput, for how much slower steps.
    if len(sizes) > 1:
        first, last = sizes[0], sizes[-1]
        note = (
            f"batch {first} to {last}: throughput x{speeds[last] / speeds[first]:.1f}, "
            f"step time x{steps_ms[last] / steps_ms[first]:.1f}"
        )
        step_axes.text(
            0.0, 1.03, note, transform=step_axes.transAxes,
            color=MUTED_TEXT, fontsize=9, va="bottom",
        )  # fmt: skip

    figure.suptitle(
        "Static batching: throughput against batch size",
        x=0.06, y=0.97, ha="left", color=TEXT, fontsize=13, fontweight="bold",
    )  # fmt: skip
    figure.text(0.06, 0.885, describe({"static": result}), color=MUTED_TEXT, fontsize=9)
    figure.subplots_adjust(left=0.06, right=0.97, top=0.74, bottom=0.12, wspace=0.16)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    return path


def load_batching_result(path: Path) -> Result:
    """Reads a batching result file.

    Raises:
        ValueError: If the file is not a batching result.
    """
    result = json.loads(path.read_text())
    if result.get("benchmark") != "batching":
        raise ValueError(f"{path} is not a batching result")
    return result


def engine_medians(result: Result, engine: str, metric: str) -> dict[int, float]:
    """Returns the median of a metric for each batch size of one engine in a batching result."""
    return {
        entry["batch_size"]: entry["engines"][engine]["median"][metric]
        for entry in result["results"]
    }


def describe_batching(result: Result) -> str:
    """Returns one line naming the model, the machine and the workload of a batching run."""
    environment = result["environment"]
    model = str(result["model"]).rsplit("/", 1)[-1]
    hardware = environment.get("gpu") or environment["cpu"]
    repeats = result["repeats"]
    runs = "1 run" if repeats == 1 else f"median of {repeats} runs"
    (prompt_min, prompt_max), (output_min, output_max) = (
        result["prompt_tokens"],
        result["output_tokens"],
    )
    return (
        f"{model}, {environment['device']} {environment['dtype']}, {hardware}, "
        f"{result['num_requests']} requests at once, prompts {prompt_min}-{prompt_max} and "
        f"outputs {output_min}-{output_max} tokens, greedy, {runs}"
    )


def batching_table(result: Result) -> str:
    """Renders a batching result as a markdown table with one row per batch size.

    "decode use" is the share of static batching's decoding slots that go to outputs that
    have not ended; continuous batching uses all of them.
    """
    header = ["batch size"]
    for metric in ("tokens/s", "p50 E2E s", "p99 E2E s", "p50 TTFT s"):
        header += [f"static {metric}", f"continuous {metric}"]
    header += ["throughput x", "static decode use"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    metrics = [
        ("output_tokens_per_s", ".1f"),
        ("e2e_p50_s", ".2f"),
        ("e2e_p99_s", ".2f"),
        ("ttft_p50_s", ".2f"),
    ]
    for entry in sorted(result["results"], key=lambda e: e["batch_size"]):
        engines = entry["engines"]
        row = [str(entry["batch_size"])]
        for metric, spec in metrics:
            row += [format(engines[engine]["median"][metric], spec) for engine in BATCHING_COLORS]
        speeds = [engines[engine]["median"]["output_tokens_per_s"] for engine in BATCHING_COLORS]
        row += [f"{speeds[1] / speeds[0]:.2f}", f"{entry['static_decode_utilization']:.0%}"]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def plot_batching(result: Result, path: Path) -> Path:
    """Draws throughput and request latency against batch size, one color per engine.

    Throughput and latency have different units, so each gets its own panel rather than a
    second y-axis.
    """
    figure = Figure(figsize=(10.5, 4.6), dpi=160, facecolor=SURFACE)
    speed_axes, latency_axes = figure.subplots(1, 2)
    sizes = sorted(entry["batch_size"] for entry in result["results"])
    end_labels: list[tuple[int, float, str]] = []

    for engine, color in BATCHING_COLORS.items():
        label = BATCHING_LABELS[engine]
        speeds = engine_medians(result, engine, "output_tokens_per_s")
        _draw_series(speed_axes, speeds, color, label, f"{speeds[sizes[-1]]:,.0f}")
        p50 = engine_medians(result, engine, "e2e_p50_s")
        p99 = engine_medians(result, engine, "e2e_p99_s")
        # The median's end label is placed below, apart from the other engine's.
        _draw_series(latency_axes, p50, color, f"{label}, median", None)
        end_labels.append((sizes[-1], p50[sizes[-1]], f"{p50[sizes[-1]]:.1f} s"))
        # The tail in the same color, told apart by dashes without markers.
        tail = _draw_series(latency_axes, p99, color, f"{label}, p99", None)
        tail.set_linestyle("--")
        tail.set_marker("")

    _style_axes(speed_axes, sizes, "Output tokens per second", "batch size")
    speed_axes.set_ylim(bottom=0)
    speed_axes.legend(frameon=False, labelcolor=TEXT, loc="upper left", fontsize=9)
    _style_axes(
        latency_axes, sizes, "Request latency in seconds (arrival to last token)", "batch size"
    )
    latency_axes.set_ylim(bottom=0)
    _spread_end_labels(latency_axes, end_labels)
    # Long handles, so that the dashes show in the legend.
    latency_axes.legend(
        frameon=False, labelcolor=TEXT, loc="upper right", fontsize=8, handlelength=3.5
    )
    # The gain in numbers at each end of the range.
    if sizes:
        static = engine_medians(result, "static", "output_tokens_per_s")
        continuous = engine_medians(result, "continuous", "output_tokens_per_s")
        gains = [
            f"batch {size}: x{continuous[size] / static[size]:.2f}"
            for size in (sizes[0], sizes[-1])
        ]
        speed_axes.text(
            0.0, 1.03, "continuous over static, " + ", ".join(dict.fromkeys(gains)),
            transform=speed_axes.transAxes, color=MUTED_TEXT, fontsize=9, va="bottom",
        )  # fmt: skip

    figure.suptitle(
        "Static against continuous batching, outputs of different lengths",
        x=0.06, y=0.97, ha="left", color=TEXT, fontsize=13, fontweight="bold",
    )  # fmt: skip
    figure.text(0.06, 0.885, describe_batching(result), color=MUTED_TEXT, fontsize=9)
    figure.subplots_adjust(left=0.06, right=0.97, top=0.74, bottom=0.12, wspace=0.16)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    return path


def load_capacity_result(path: Path) -> Result:
    """Reads a capacity result file.

    Raises:
        ValueError: If the file is not a capacity result.
    """
    result = json.loads(path.read_text())
    if result.get("benchmark") != "capacity":
        raise ValueError(f"{path} is not a capacity result")
    return result


def layout_medians(result: Result, layout: str, metric: str) -> dict[int, float]:
    """Returns the median of a metric for each KV budget of one layout in a capacity result."""
    return {
        entry["kv_tokens"]: entry["layouts"][layout]["median"][metric]
        for entry in result["results"]
    }


def capacity_table(result: Result) -> str:
    """Renders a capacity result as a markdown table with one row per KV budget.

    "requests" counts those holding KV memory after a step: the most at once and the mean
    over steps.
    """
    header = [
        "KV tokens",
        "KV MB",
        "contiguous requests",
        "paged most requests",
        "contiguous mean requests",
        "paged mean requests",
        "paged preemptions",
        "contiguous tokens/s",
        "paged tokens/s",
        "throughput x",
        "contiguous p99 E2E s",
        "paged p99 E2E s",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for entry in sorted(result["results"], key=lambda e: e["kv_tokens"]):
        contiguous = entry["layouts"]["contiguous"]["median"]
        paged = entry["layouts"]["paged"]["median"]
        row = [
            str(entry["kv_tokens"]),
            f"{entry['kv_bytes'] / 2**20:.0f}",
            f"{contiguous['peak_running']:.0f}",
            f"{paged['peak_running']:.0f}",
            f"{contiguous['mean_running']:.1f}",
            f"{paged['mean_running']:.1f}",
            f"{paged['preemptions']:.0f}",
            f"{contiguous['output_tokens_per_s']:.1f}",
            f"{paged['output_tokens_per_s']:.1f}",
            f"{paged['output_tokens_per_s'] / contiguous['output_tokens_per_s']:.2f}",
            f"{contiguous['e2e_p99_s']:.2f}",
            f"{paged['e2e_p99_s']:.2f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def plot_capacity(result: Result, path: Path) -> Path:
    """Draws the requests that hold KV memory and the throughput against the KV budget."""
    figure = Figure(figsize=(10.5, 4.6), dpi=160, facecolor=SURFACE)
    seqs_axes, speed_axes = figure.subplots(1, 2)
    budgets = sorted(entry["kv_tokens"] for entry in result["results"])
    end_labels: list[tuple[int, float, str]] = []

    for layout, color in CAPACITY_COLORS.items():
        label = CAPACITY_LABELS[layout]
        mean = layout_medians(result, layout, "mean_running")
        peak = layout_medians(result, layout, "peak_running")
        _draw_series(seqs_axes, mean, color, f"{label}, mean over steps", None)
        end_labels.append((budgets[-1], mean[budgets[-1]], f"{mean[budgets[-1]]:.1f}"))
        # The most at once in the same color, told apart by dashes without markers.
        most = _draw_series(seqs_axes, peak, color, f"{label}, most at once", None)
        most.set_linestyle("--")
        most.set_marker("")
        speeds = layout_medians(result, layout, "output_tokens_per_s")
        _draw_series(speed_axes, speeds, color, label, f"{speeds[budgets[-1]]:,.0f}")

    _style_axes(seqs_axes, budgets, "Requests holding KV memory", "KV memory budget in tokens")
    seqs_axes.set_ylim(bottom=0)
    _spread_end_labels(seqs_axes, end_labels)
    seqs_axes.text(
        0.0, 1.03,
        f"contiguous: a row of {result['max_len']} tokens per request, "
        f"paged: blocks of {result['results'][0]['layouts']['paged']['block_size']}",
        transform=seqs_axes.transAxes, color=MUTED_TEXT, fontsize=9, va="bottom",
    )  # fmt: skip
    # Long handles, so that the dashes show in the legend.
    seqs_axes.legend(frameon=False, labelcolor=TEXT, loc="upper left", fontsize=8, handlelength=3.5)
    _style_axes(speed_axes, budgets, "Output tokens per second", "KV memory budget in tokens")
    speed_axes.set_ylim(bottom=0)
    speed_axes.legend(frameon=False, labelcolor=TEXT, loc="upper left", fontsize=9)
    # The gain in numbers at each end of the range.
    contiguous = layout_medians(result, "contiguous", "output_tokens_per_s")
    paged = layout_medians(result, "paged", "output_tokens_per_s")
    gains = [
        f"{budget} tokens: x{paged[budget] / contiguous[budget]:.2f}"
        for budget in (budgets[0], budgets[-1])
    ]
    speed_axes.text(
        0.0, 1.03, "paged over contiguous, " + ", ".join(dict.fromkeys(gains)),
        transform=speed_axes.transAxes, color=MUTED_TEXT, fontsize=9, va="bottom",
    )  # fmt: skip

    figure.suptitle(
        "Contiguous against paged KV memory under the same budget",
        x=0.06, y=0.97, ha="left", color=TEXT, fontsize=13, fontweight="bold",
    )  # fmt: skip
    figure.text(0.06, 0.885, describe_batching(result), color=MUTED_TEXT, fontsize=9)
    figure.subplots_adjust(left=0.06, right=0.97, top=0.74, bottom=0.12, wspace=0.16)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    return path


def load_prefix_result(path: Path) -> Result:
    """Reads a prefix caching result file.

    Raises:
        ValueError: If the file is not a prefix caching result.
    """
    result = json.loads(path.read_text())
    if result.get("benchmark") != "prefix":
        raise ValueError(f"{path} is not a prefix caching result")
    return result


def prefix_medians(result: Result, arrival: str, cache: str, metric: str) -> dict[int, float]:
    """Returns the median of a metric for each system prompt length of one arrival pattern
    and cache setting in a prefix caching result."""
    return {
        entry["system_tokens"]: entry["arrivals"][arrival][cache]["median"][metric]
        for entry in result["results"]
    }


def describe_prefix(result: Result) -> str:
    """Returns one line naming the model, the machine and the workload of a prefix run."""
    environment = result["environment"]
    model = str(result["model"]).rsplit("/", 1)[-1]
    hardware = environment.get("gpu") or environment["cpu"]
    repeats = result["repeats"]
    runs = "1 run" if repeats == 1 else f"median of {repeats} runs"
    user_min, user_max = result["user_tokens"]
    return (
        f"{model}, {environment['device']} {environment['dtype']}, {hardware}, "
        f"{result['num_requests']} requests of one system prompt and {user_min}-{user_max} "
        f"other tokens, {result['output_tokens']} output tokens, greedy, {runs}"
    )


def prefix_table(result: Result) -> str:
    """Renders a prefix caching result as a markdown table with one row per system prompt
    length. TTFT is in milliseconds; "x" is the TTFT without the cache over the TTFT with
    it."""
    header = [
        "system tokens",
        "one at a time, off p50 ms",
        "one at a time, on p50 ms",
        "x",
        "at once, off p50 ms",
        "at once, on p50 ms",
        "x",
        "at once, off p99 ms",
        "at once, on p99 ms",
        "at once, off tokens/s",
        "at once, on tokens/s",
        "hit rate",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for entry in sorted(result["results"], key=lambda e: e["system_tokens"]):
        alone = entry["arrivals"]["one_at_a_time"]
        together = entry["arrivals"]["all_at_once"]
        alone_off, alone_on = alone["off"]["median"], alone["on"]["median"]
        off, on = together["off"]["median"], together["on"]["median"]
        row = [
            str(entry["system_tokens"]),
            f"{alone_off['ttft_p50_s'] * 1000:.1f}",
            f"{alone_on['ttft_p50_s'] * 1000:.1f}",
            f"{alone_off['ttft_p50_s'] / alone_on['ttft_p50_s']:.1f}",
            f"{off['ttft_p50_s'] * 1000:.0f}",
            f"{on['ttft_p50_s'] * 1000:.0f}",
            f"{off['ttft_p50_s'] / on['ttft_p50_s']:.1f}",
            f"{off['ttft_p99_s'] * 1000:.0f}",
            f"{on['ttft_p99_s'] * 1000:.0f}",
            f"{off['output_tokens_per_s']:.1f}",
            f"{on['output_tokens_per_s']:.1f}",
            f"{on['prefix_hit_rate']:.0%}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def plot_prefix(result: Result, path: Path) -> Path:
    """Draws the time to first token against the system prompt length, without and with
    prefix caching, for requests that arrive one at a time and all at once."""
    figure = Figure(figsize=(10.5, 4.6), dpi=160, facecolor=SURFACE)
    alone_axes, together_axes = figure.subplots(1, 2)
    lengths = sorted(entry["system_tokens"] for entry in result["results"])
    end_labels: list[tuple[int, float, str]] = []

    def milliseconds(arrival: str, cache: str, metric: str) -> dict[int, float]:
        return {n: s * 1000 for n, s in prefix_medians(result, arrival, cache, metric).items()}

    for cache, color in PREFIX_COLORS.items():
        label = PREFIX_LABELS[cache]
        alone = milliseconds("one_at_a_time", cache, "ttft_p50_s")
        _draw_series(alone_axes, alone, color, label, f"{alone[lengths[-1]]:,.0f} ms")
        p50 = milliseconds("all_at_once", cache, "ttft_p50_s")
        p99 = milliseconds("all_at_once", cache, "ttft_p99_s")
        _draw_series(together_axes, p50, color, f"{label}, p50", None)
        end_labels.append((lengths[-1], p50[lengths[-1]], f"{p50[lengths[-1]]:,.0f} ms"))
        # p99 in the same color, told apart by dashes without markers.
        slowest = _draw_series(together_axes, p99, color, f"{label}, p99", None)
        slowest.set_linestyle("--")
        slowest.set_marker("")

    xlabel = "system prompt tokens shared by every request"
    _style_axes(alone_axes, lengths, "Median time to first token, one request at a time", xlabel)
    alone_axes.set_ylim(bottom=0)
    alone_axes.legend(frameon=False, labelcolor=TEXT, loc="upper left", fontsize=9)
    _style_axes(
        together_axes,
        lengths,
        f"Time to first token, {result['num_requests']} requests at once",
        xlabel,
    )
    together_axes.set_ylim(bottom=0)
    _spread_end_labels(together_axes, end_labels)
    # Long handles, so that the dashes show in the legend.
    together_axes.legend(
        frameon=False, labelcolor=TEXT, loc="upper left", fontsize=8, handlelength=3.5
    )
    for axes, arrival in ((alone_axes, "one_at_a_time"), (together_axes, "all_at_once")):
        off = prefix_medians(result, arrival, "off", "ttft_p50_s")
        on = prefix_medians(result, arrival, "on", "ttft_p50_s")
        ratios = [f"{n} tokens x{off[n] / on[n]:.1f}" for n in (lengths[0], lengths[-1])]
        axes.text(
            0.0, 1.03, "p50 without over with the cache: " + ", ".join(dict.fromkeys(ratios)),
            transform=axes.transAxes, color=MUTED_TEXT, fontsize=9, va="bottom",
        )  # fmt: skip

    figure.suptitle(
        "Time to first token without and with prefix caching",
        x=0.06, y=0.97, ha="left", color=TEXT, fontsize=13, fontweight="bold",
    )  # fmt: skip
    figure.text(0.06, 0.885, describe_prefix(result), color=MUTED_TEXT, fontsize=9)
    figure.subplots_adjust(left=0.06, right=0.97, top=0.74, bottom=0.12, wspace=0.16)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    return path


def _draw_series(
    axes: Axes, values: dict[int, float], color: str, label: str, end_label: str | None
) -> Line2D:
    lengths = sorted(values)
    (line,) = axes.plot(
        lengths,
        [values[length] for length in lengths],
        color=color,
        label=label,
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
    if end_label is None:
        return line
    axes.annotate(
        end_label,
        (lengths[-1], values[lengths[-1]]),
        xytext=(8, 0),
        textcoords="offset points",
        color=TEXT,
        fontsize=9,
        va="center",
    )
    return line


def _spread_end_labels(axes: Axes, labels: list[tuple[int, float, str]]) -> None:
    """Writes labels next to the ends of lines, moved apart vertically where they would
    overlap. Call it after the axis limits are final."""
    min_gap = 12.0  # points, a little more than the 9-point text
    to_points = 72 / axes.figure.dpi
    # Vertical position of each line end in points, lowest first.
    ends = sorted(
        (axes.transData.transform((x, y))[1] * to_points, x, y, text) for x, y, text in labels
    )
    placed: list[float] = []
    for height, _, _, _ in ends:
        placed.append(height if not placed else max(height, placed[-1] + min_gap))
    # Center the moved group on the line ends, so no label strays far from its line.
    shift = (sum(h for h, *_ in ends) - sum(placed)) / len(placed) if placed else 0.0
    for (height, x, y, text), position in zip(ends, placed, strict=True):
        axes.annotate(
            text,
            (x, y),
            xytext=(8, position + shift - height),
            textcoords="offset points",
            color=TEXT,
            fontsize=9,
            va="center",
        )


def _style_axes(axes: Axes, lengths: Sequence[int], title: str, xlabel: str) -> None:
    axes.set_facecolor(SURFACE)
    axes.set_title(title, loc="left", color=TEXT, fontsize=10, pad=24)
    axes.set_xlabel(xlabel, color=MUTED_TEXT, fontsize=9)
    # Values double from point to point, so a log scale spaces them evenly.
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
    throughput = commands.add_parser("throughput", help="static batching across batch sizes")
    throughput.add_argument("result", type=Path, help="throughput result file")
    throughput.add_argument(
        "--output", type=Path, required=True, help="path of the chart and table, without suffix"
    )
    batching = commands.add_parser("batching", help="static against continuous batching")
    batching.add_argument("result", type=Path, help="batching result file")
    batching.add_argument(
        "--output", type=Path, required=True, help="path of the chart and table, without suffix"
    )
    capacity = commands.add_parser("capacity", help="contiguous against paged KV memory")
    capacity.add_argument("result", type=Path, help="capacity result file")
    capacity.add_argument(
        "--output", type=Path, required=True, help="path of the chart and table, without suffix"
    )
    prefix = commands.add_parser("prefix", help="time to first token with prefix caching")
    prefix.add_argument("result", type=Path, help="prefix caching result file")
    prefix.add_argument(
        "--output", type=Path, required=True, help="path of the chart and table, without suffix"
    )
    args = parser.parse_args(argv)

    chart = args.output.with_name(args.output.name + ".png")
    table = args.output.with_name(args.output.name + ".md")
    if args.command == "latency":
        results = load_latency_results(args.results)
        plot_latency(results, chart)
        table.write_text(f"{describe(results)}\n\n{latency_table(results)}")
    elif args.command == "throughput":
        result = load_throughput_result(args.result)
        plot_throughput(result, chart)
        table.write_text(f"{describe({'static': result})}\n\n{throughput_table(result)}")
    elif args.command == "batching":
        result = load_batching_result(args.result)
        plot_batching(result, chart)
        table.write_text(f"{describe_batching(result)}\n\n{batching_table(result)}")
    elif args.command == "capacity":
        result = load_capacity_result(args.result)
        plot_capacity(result, chart)
        table.write_text(f"{describe_batching(result)}\n\n{capacity_table(result)}")
    else:
        result = load_prefix_result(args.result)
        plot_prefix(result, chart)
        table.write_text(f"{describe_prefix(result)}\n\n{prefix_table(result)}")
    print(f"saved {chart}")
    print(f"saved {table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
