import json
from pathlib import Path

import pytest

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
