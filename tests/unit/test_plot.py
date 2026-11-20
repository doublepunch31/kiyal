import json
from pathlib import Path

import pytest
from matplotlib.figure import Figure
from matplotlib.text import Annotation

from benchmarks import plot

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def latency_result(engine: str, total_s: dict[int, float], **overrides: object) -> plot.Result:
    """A latency result in the shape that benchmarks.latency saves."""
    result: plot.Result = {
        "benchmark": "latency",
        "engine": engine,
        "model": "org/tiny-model",
        "revision": None,
        "prompt_tokens": 128,
        "repeats": 3,
        "environment": {"device": "cpu", "dtype": "float32", "cpu": "Test CPU", "torch": "2.15.0"},
        "results": [
            {
                "output_tokens": length,
                "median": {"total_s": total, "output_tokens_per_s": length / total},
            }
            for length, total in total_s.items()
        ],
    }
    return result | overrides


def write(path: Path, result: plot.Result) -> Path:
    path.write_text(json.dumps(result))
    return path


@pytest.fixture
def results() -> dict[str, plot.Result]:
    return {
        "cached": latency_result("cached", {32: 0.5, 64: 1.0, 128: 2.0}),
        "naive": latency_result("naive", {32: 2.0, 64: 5.0, 128: 20.0}),
    }


def test_results_are_loaded_by_engine_in_a_fixed_order(
    tmp_path: Path, results: dict[str, plot.Result]
) -> None:
    naive = write(tmp_path / "a.json", results["naive"])
    cached = write(tmp_path / "b.json", results["cached"])
    loaded = plot.load_latency_results([naive, cached])
    assert list(loaded) == ["cached", "naive"]
    assert loaded == results


def test_a_single_engine_can_be_loaded(tmp_path: Path, results: dict[str, plot.Result]) -> None:
    loaded = plot.load_latency_results([write(tmp_path / "a.json", results["naive"])])
    assert list(loaded) == ["naive"]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"benchmark": "throughput"}, "not a latency result"),
        ({"engine": "paged"}, "not a latency result"),
        ({"engine": "cached"}, "second result for the cached engine"),
        ({"model": "org/other-model"}, "differ in model"),
        ({"prompt_tokens": 64}, "differ in prompt_tokens"),
    ],
)
def test_results_that_cannot_be_compared_raise(
    tmp_path: Path, results: dict[str, plot.Result], changes: dict[str, object], message: str
) -> None:
    cached = write(tmp_path / "cached.json", results["cached"])
    other = write(tmp_path / "other.json", results["naive"] | changes)
    with pytest.raises(ValueError, match=message):
        plot.load_latency_results([cached, other])


def test_results_from_different_devices_raise(
    tmp_path: Path, results: dict[str, plot.Result]
) -> None:
    environment = results["naive"]["environment"] | {"device": "mps"}
    cached = write(tmp_path / "cached.json", results["cached"])
    other = write(tmp_path / "other.json", results["naive"] | {"environment": environment})
    with pytest.raises(ValueError, match="differ in device: cpu and mps"):
        plot.load_latency_results([cached, other])


def test_no_files_raise() -> None:
    with pytest.raises(ValueError, match="no result files"):
        plot.load_latency_results([])


def test_growth_is_relative_to_the_previous_length() -> None:
    assert plot.growth({128: 8.0, 32: 1.0, 64: 2.0}) == {64: 2.0, 128: 4.0}
    assert plot.growth({32: 1.0}) == {}


def test_the_description_names_model_machine_and_method(results: dict[str, plot.Result]) -> None:
    expected = "tiny-model, cpu float32, Test CPU, 128-token prompt, greedy, median of 3 runs"
    assert plot.describe(results) == expected
    environment = results["cached"]["environment"] | {"device": "cuda:0", "gpu": "Test GPU"}
    on_gpu = {"cached": results["cached"] | {"environment": environment, "repeats": 1}}
    assert plot.describe(on_gpu) == (
        "tiny-model, cuda:0 float32, Test GPU, 128-token prompt, greedy, 1 run"
    )


def test_the_table_lists_speed_time_growth_and_speedup(results: dict[str, plot.Result]) -> None:
    assert plot.latency_table(results).splitlines() == [
        "| output tokens | cached tokens/s | cached total s | cached x previous "
        "| naive tokens/s | naive total s | naive x previous | speedup |",
        "|---|---|---|---|---|---|---|---|",
        "| 32 | 64.0 | 0.50 | - | 16.0 | 2.00 | - | 4.0 |",
        "| 64 | 64.0 | 1.00 | 2.00 | 12.8 | 5.00 | 2.50 | 5.0 |",
        "| 128 | 64.0 | 2.00 | 2.00 | 6.4 | 20.00 | 4.00 | 10.0 |",
    ]


def test_the_table_marks_lengths_an_engine_did_not_run() -> None:
    results = {
        "cached": latency_result("cached", {32: 0.5, 64: 1.0}),
        "naive": latency_result("naive", {32: 2.0}),
    }
    assert plot.latency_table(results).splitlines()[-1] == (
        "| 64 | 64.0 | 1.00 | 2.00 | - | - | - | - |"
    )


def test_the_table_of_one_engine_has_no_speedup(results: dict[str, plot.Result]) -> None:
    lines = plot.latency_table({"naive": results["naive"]}).splitlines()
    assert lines[0] == "| output tokens | naive tokens/s | naive total s | naive x previous |"
    assert lines[-1] == "| 128 | 6.4 | 20.00 | 4.00 |"


def test_the_chart_is_written_as_a_png(tmp_path: Path, results: dict[str, plot.Result]) -> None:
    path = plot.plot_latency(results, tmp_path / "charts" / "latency.png")
    assert path.read_bytes().startswith(PNG_SIGNATURE)


def test_main_writes_the_chart_and_the_table(
    tmp_path: Path, results: dict[str, plot.Result], capsys: pytest.CaptureFixture[str]
) -> None:
    files = [str(write(tmp_path / f"{engine}.json", result)) for engine, result in results.items()]
    output = tmp_path / "out" / "latency-cpu"
    assert plot.main(["latency", *files, "--output", str(output)]) == 0

    chart, table = output.with_name("latency-cpu.png"), output.with_name("latency-cpu.md")
    assert chart.read_bytes().startswith(PNG_SIGNATURE)
    assert table.read_text() == f"{plot.describe(results)}\n\n{plot.latency_table(results)}"
    assert capsys.readouterr().out == f"saved {chart}\nsaved {table}\n"


def throughput_result(timings: dict[int, tuple[float, float, float]]) -> plot.Result:
    """A throughput result in the shape that benchmarks.throughput saves.

    Args:
        timings: Output tokens/s, step time and TTFT in seconds for each batch size.
    """
    return {
        "benchmark": "throughput",
        "engine": "static",
        "model": "org/tiny-model",
        "revision": None,
        "prompt_tokens": 128,
        "output_tokens": 64,
        "repeats": 3,
        "environment": {"device": "cpu", "dtype": "float32", "cpu": "Test CPU", "torch": "2.15.0"},
        "results": [
            {
                "batch_size": size,
                "median": {"output_tokens_per_s": speed, "step_s": step, "ttft_s": ttft},
            }
            for size, (speed, step, ttft) in timings.items()
        ],
    }


@pytest.fixture
def batches() -> plot.Result:
    return throughput_result(
        {1: (100.0, 0.01, 0.05), 2: (180.0, 0.011, 0.09), 4: (300.0, 0.0125, 0.2)}
    )


def test_a_throughput_result_is_loaded(tmp_path: Path, batches: plot.Result) -> None:
    assert plot.load_throughput_result(write(tmp_path / "t.json", batches)) == batches


def test_a_latency_result_is_not_a_throughput_result(
    tmp_path: Path, results: dict[str, plot.Result]
) -> None:
    with pytest.raises(ValueError, match="not a throughput result"):
        plot.load_throughput_result(write(tmp_path / "l.json", results["cached"]))


def test_the_throughput_description_names_the_output_length(batches: plot.Result) -> None:
    assert plot.describe({"static": batches}) == (
        "tiny-model, cpu float32, Test CPU, 128-token prompts, 64 output tokens, greedy, "
        "median of 3 runs"
    )


def test_the_throughput_table_compares_each_batch_with_the_smallest(batches: plot.Result) -> None:
    assert plot.throughput_table(batches).splitlines() == [
        "| batch size | output tokens/s | x batch 1 | step ms | x batch 1 | TTFT ms |",
        "|---|---|---|---|---|---|",
        "| 1 | 100.0 | 1.00 | 10.0 | 1.00 | 50 |",
        "| 2 | 180.0 | 1.80 | 11.0 | 1.10 | 90 |",
        "| 4 | 300.0 | 3.00 | 12.5 | 1.25 | 200 |",
    ]


def test_main_writes_the_throughput_chart_and_table(
    tmp_path: Path, batches: plot.Result, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "out" / "throughput-cpu"
    assert (
        plot.main(["throughput", str(write(tmp_path / "t.json", batches)), "--output", str(output)])
        == 0
    )

    chart, table = output.with_name("throughput-cpu.png"), output.with_name("throughput-cpu.md")
    assert chart.read_bytes().startswith(PNG_SIGNATURE)
    expected = f"{plot.describe({'static': batches})}\n\n{plot.throughput_table(batches)}"
    assert table.read_text() == expected
    assert capsys.readouterr().out == f"saved {chart}\nsaved {table}\n"


def batching_result(speeds: dict[int, tuple[float, float]]) -> plot.Result:
    """A batching result in the shape that benchmarks.batching saves.

    Args:
        speeds: Static and continuous output tokens/s per batch size. Latencies are derived
            from them so that each value is distinct.
    """

    def summary(speed: float) -> dict[str, float]:
        return {
            "output_tokens_per_s": speed,
            "e2e_p50_s": 1000 / speed,
            "e2e_p99_s": 2000 / speed,
            "ttft_p50_s": 500 / speed,
        }

    return {
        "benchmark": "batching",
        "model": "org/tiny-model",
        "revision": None,
        "num_requests": 64,
        "prompt_tokens": [32, 256],
        "output_tokens": [16, 256],
        "seed": 0,
        "max_num_batched_tokens": 2048,
        "repeats": 3,
        "warmup": 1,
        "environment": {"device": "cpu", "dtype": "float32", "cpu": "Test CPU", "torch": "2.15.0"},
        "results": [
            {
                "batch_size": size,
                "static_decode_utilization": 0.5,
                "engines": {
                    "static": {"median": summary(static)},
                    "continuous": {"median": summary(continuous)},
                },
            }
            for size, (static, continuous) in speeds.items()
        ],
    }


@pytest.fixture
def engines() -> plot.Result:
    return batching_result({8: (100.0, 125.0), 16: (200.0, 250.0)})


def test_a_batching_result_is_loaded(tmp_path: Path, engines: plot.Result) -> None:
    assert plot.load_batching_result(write(tmp_path / "b.json", engines)) == engines
    with pytest.raises(ValueError, match="not a batching result"):
        plot.load_batching_result(write(tmp_path / "t.json", throughput_result({1: (1, 1, 1)})))


def test_the_batching_description_names_the_workload(engines: plot.Result) -> None:
    assert plot.describe_batching(engines) == (
        "tiny-model, cpu float32, Test CPU, 64 requests at once, prompts 32-256 and outputs "
        "16-256 tokens, greedy, median of 3 runs"
    )


def test_the_batching_table_compares_the_engines_at_each_batch_size(
    engines: plot.Result,
) -> None:
    lines = plot.batching_table(engines).splitlines()
    assert lines[0] == (
        "| batch size | static tokens/s | continuous tokens/s | static p50 E2E s | "
        "continuous p50 E2E s | static p99 E2E s | continuous p99 E2E s | static p50 TTFT s | "
        "continuous p50 TTFT s | throughput x | static decode use |"
    )
    assert lines[2:] == [
        "| 8 | 100.0 | 125.0 | 10.00 | 8.00 | 20.00 | 16.00 | 5.00 | 4.00 | 1.25 | 50% |",
        "| 16 | 200.0 | 250.0 | 5.00 | 4.00 | 10.00 | 8.00 | 2.50 | 2.00 | 1.25 | 50% |",
    ]


def test_main_writes_the_batching_chart_and_table(
    tmp_path: Path, engines: plot.Result, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "batching-cpu"
    path = write(tmp_path / "b.json", engines)
    assert plot.main(["batching", str(path), "--output", str(output)]) == 0
    chart, table = output.with_name("batching-cpu.png"), output.with_name("batching-cpu.md")
    assert chart.read_bytes().startswith(PNG_SIGNATURE)
    assert (
        table.read_text() == f"{plot.describe_batching(engines)}\n\n{plot.batching_table(engines)}"
    )
    assert capsys.readouterr().out == f"saved {chart}\nsaved {table}\n"


def test_end_labels_that_would_overlap_are_moved_apart() -> None:
    figure = Figure(dpi=100)
    axes = figure.subplots()
    axes.set_ylim(0, 100)
    # Two line ends 1 unit apart, a few points on screen, and one far away.
    plot._spread_end_labels(axes, [(1, 20.0, "a"), (1, 21.0, "b"), (1, 80.0, "c")])
    to_points = 72 / figure.dpi
    annotations = [text for text in axes.texts if isinstance(text, Annotation)]
    assert len(annotations) == 3
    heights = {
        a.get_text(): axes.transData.transform(a.xy)[1] * to_points + a.xyann[1]
        for a in annotations
    }
    assert heights["b"] - heights["a"] >= 12
    # A label with room stays next to its line end.
    assert heights["c"] == pytest.approx(axes.transData.transform((1, 80.0))[1] * to_points, abs=8)


def capacity_result(layouts: dict[int, tuple[float, float]]) -> plot.Result:
    """A capacity result in the shape that benchmarks.capacity saves.

    Args:
        layouts: Contiguous and paged output tokens/s per KV budget. The other metrics are
            derived from them so that each value is distinct.
    """

    def summary(speed: float, kv_tokens: int, paged: bool) -> dict[str, float]:
        return {
            "output_tokens_per_s": speed,
            "e2e_p99_s": 1000 / speed,
            "peak_running": kv_tokens / (64 if paged else 256),
            "mean_running": kv_tokens / (100 if paged else 300),
            "preemptions": 5.0 if paged else 0.0,
        }

    return {
        **batching_result({}),
        "benchmark": "capacity",
        "max_len": 256,
        "results": [
            {
                "kv_tokens": kv_tokens,
                "kv_bytes": kv_tokens * 2**10,
                "layouts": {
                    "contiguous": {
                        "block_size": 256,
                        "median": summary(contiguous, kv_tokens, False),
                    },
                    "paged": {"block_size": 16, "median": summary(paged, kv_tokens, True)},
                },
            }
            for kv_tokens, (contiguous, paged) in layouts.items()
        ],
    }


@pytest.fixture
def layouts() -> plot.Result:
    return capacity_result({2048: (100.0, 150.0), 4096: (200.0, 250.0)})


def test_a_capacity_result_is_loaded(tmp_path: Path, layouts: plot.Result) -> None:
    assert plot.load_capacity_result(write(tmp_path / "c.json", layouts)) == layouts
    with pytest.raises(ValueError, match="not a capacity result"):
        plot.load_capacity_result(write(tmp_path / "b.json", batching_result({})))


def test_the_capacity_table_compares_the_layouts_at_each_budget(layouts: plot.Result) -> None:
    lines = plot.capacity_table(layouts).splitlines()
    assert lines[0] == (
        "| KV tokens | KV MB | contiguous requests | paged most requests | "
        "contiguous mean requests | paged mean requests | paged preemptions | "
        "contiguous tokens/s | paged tokens/s | throughput x | contiguous p99 E2E s | "
        "paged p99 E2E s |"
    )
    assert lines[2:] == [
        "| 2048 | 2 | 8 | 32 | 6.8 | 20.5 | 5 | 100.0 | 150.0 | 1.50 | 10.00 | 6.67 |",
        "| 4096 | 4 | 16 | 64 | 13.7 | 41.0 | 5 | 200.0 | 250.0 | 1.25 | 5.00 | 4.00 |",
    ]


def test_main_writes_the_capacity_chart_and_table(
    tmp_path: Path, layouts: plot.Result, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "capacity-cpu"
    path = write(tmp_path / "c.json", layouts)
    assert plot.main(["capacity", str(path), "--output", str(output)]) == 0
    chart, table = output.with_name("capacity-cpu.png"), output.with_name("capacity-cpu.md")
    assert chart.read_bytes().startswith(PNG_SIGNATURE)
    assert (
        table.read_text() == f"{plot.describe_batching(layouts)}\n\n{plot.capacity_table(layouts)}"
    )
    assert capsys.readouterr().out == f"saved {chart}\nsaved {table}\n"
