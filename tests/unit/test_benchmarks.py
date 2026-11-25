import dataclasses
import json
import math
import subprocess
from functools import partial
from pathlib import Path

import pytest
import torch

from benchmarks import batching, capacity, latency, perplexity, prefix, serving, throughput
from benchmarks.common import REPO_ROOT, capture_environment, median, percentile, save_result
from benchmarks.latency import build_prompt
from kiyal.config import EngineConfig
from kiyal.engine.generator import TextGenerator
from kiyal.engine.llm_engine import LLMEngine
from kiyal.model.llama import LlamaModel
from kiyal.model.loader import load_model
from kiyal.tokenizer import Tokenizer

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def generator(tiny_model_dir: Path, tiny_tokenizer: Tokenizer) -> TextGenerator:
    model = load_model(tiny_model_dir, device=CPU, dtype=torch.float32)
    return TextGenerator(model, tiny_tokenizer)


def test_environment_names_the_code_and_the_machine() -> None:
    environment = capture_environment(CPU, torch.float32)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert environment["kiyal_commit"] == head.stdout.strip()
    assert isinstance(environment["kiyal_uncommitted_changes"], bool)
    assert environment["torch"] == torch.__version__
    assert (environment["device"], environment["dtype"]) == ("cpu", "float32")
    assert environment["cpu"]
    assert "gpu" not in environment
    assert environment["power_source"] is None or isinstance(environment["power_source"], str)
    json.dumps(environment)


def test_median_is_the_middle_run() -> None:
    assert median([3.0, 1.0, 10.0]) == 3.0


def test_percentiles_interpolate_between_values() -> None:
    values = [float(v) for v in range(1, 102)]  # 1 to 101
    assert percentile(values, 50) == median(values) == 51.0
    assert percentile(values, 99) == pytest.approx(100.0)
    assert percentile([1.0, 3.0], 50) == 2.0
    with pytest.raises(ValueError, match=r"percent must be in \[1, 99\]"):
        percentile(values, 100)


def test_save_result_writes_json_in_a_new_directory(tmp_path: Path) -> None:
    path = save_result({"benchmark": "latency", "results": [1, 2]}, tmp_path / "raw" / "run.json")
    assert json.loads(path.read_text()) == {"benchmark": "latency", "results": [1, 2]}


@pytest.mark.parametrize("num_tokens", [1, 40, 128, 500])
def test_prompt_has_the_requested_length(tiny_tokenizer: Tokenizer, num_tokens: int) -> None:
    prompt = latency.build_prompt(tiny_tokenizer, num_tokens)
    assert len(prompt) == num_tokens
    assert prompt == latency.build_prompt(tiny_tokenizer, 500)[:num_tokens]


def test_run_timing_derives_the_per_token_metrics() -> None:
    timing = latency.RunTiming(output_tokens=11, total_s=2.0, ttft_s=0.5)
    assert timing.tpot_s == pytest.approx(0.15)
    assert timing.output_tokens_per_s == pytest.approx(5.5)


def test_time_generation_times_the_requested_number_of_tokens(
    generator: TextGenerator, tiny_tokenizer: Tokenizer
) -> None:
    prompt = latency.build_prompt(tiny_tokenizer, 16)
    timing = latency.time_generation(generator, prompt, 8, CPU)
    assert timing.output_tokens == 8
    assert 0 < timing.ttft_s < timing.total_s


def test_time_generation_rejects_lengths_it_cannot_measure(
    generator: TextGenerator, tiny_tokenizer: Tokenizer
) -> None:
    prompt = latency.build_prompt(tiny_tokenizer, 16)
    with pytest.raises(ValueError, match="at least 2 output tokens"):
        latency.time_generation(generator, prompt, 1, CPU)
    # The tiny model's context holds 128 tokens.
    with pytest.raises(ValueError, match="context is too short"):
        latency.time_generation(generator, prompt, 200, CPU)


def test_measure_reports_every_run_and_the_medians(
    generator: TextGenerator, tiny_tokenizer: Tokenizer
) -> None:
    prompt = latency.build_prompt(tiny_tokenizer, 16)
    results = latency.measure(generator, prompt, [4, 8], CPU, repeats=3)
    assert [entry["output_tokens"] for entry in results] == [4, 8]
    for entry in results:
        runs, summary = entry["runs"], entry["median"]
        assert isinstance(runs, list)
        assert isinstance(summary, dict)
        assert len(runs) == 3
        assert summary["total_s"] == median([run["total_s"] for run in runs])
        assert set(summary) == {"total_s", "ttft_s", "tpot_s", "output_tokens_per_s"}


@pytest.mark.parametrize("engine", latency.ENGINES)
def test_main_saves_a_result_file(
    tiny_model_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], engine: str
) -> None:
    output = tmp_path / "latency.json"
    options = ["--device", "cpu", "--dtype", "fp32", "--prompt-tokens", "16", "--repeats", "2"]
    args = ["--model", str(tiny_model_dir), "--engine", engine, *options]
    assert latency.main([*args, "--output-tokens", "4", "8", "--output", str(output)]) == 0

    result = json.loads(output.read_text())
    assert (result["benchmark"], result["engine"]) == ("latency", engine)
    assert (result["prompt_tokens"], result["repeats"], result["warmup"]) == (16, 2, 1)
    assert result["environment"]["device"] == "cpu"
    assert [entry["output_tokens"] for entry in result["results"]] == [4, 8]
    assert all(len(entry["runs"]) == 2 for entry in result["results"])
    printed = capsys.readouterr().out
    assert "tokens/s" in printed
    assert f"saved {output}" in printed


@pytest.mark.parametrize(
    ("engine", "prefill_and_first_step"), [("cached", [16, 1]), ("naive", [16, 17])]
)
def test_main_runs_the_requested_engine(
    tiny_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
    prefill_and_first_step: list[int],
) -> None:
    lengths: list[int] = []
    forward = LlamaModel.forward

    def recording_forward(self: LlamaModel, input_ids: torch.Tensor, *args, **kwargs):
        lengths.append(input_ids.shape[1])
        return forward(self, input_ids, *args, **kwargs)

    monkeypatch.setattr(LlamaModel, "forward", recording_forward)
    options = ["--device", "cpu", "--dtype", "fp32", "--prompt-tokens", "16", "--repeats", "1"]
    args = ["--model", str(tiny_model_dir), "--engine", engine, *options]
    assert latency.main([*args, "--output-tokens", "4", "--output", str(tmp_path / "x.json")]) == 0
    assert lengths[:2] == prefill_and_first_step


def test_the_default_run_covers_32_to_2048_output_tokens_with_the_cache(
    tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: list[list[int]] = []

    def fake_measure(generator, prompt, output_lengths, device, repeats):
        requested.append(list(output_lengths))
        return []

    monkeypatch.setattr(latency, "measure", fake_measure)
    output = tmp_path / "latency.json"
    args = ["--model", str(tiny_model_dir), "--device", "cpu", "--warmup", "0"]
    assert latency.main([*args, "--output", str(output)]) == 0
    assert requested == [[32, 64, 128, 256, 512, 1024, 2048]]
    assert json.loads(output.read_text())["engine"] == "cached"


def test_batch_prompts_are_different_and_of_equal_length(tiny_tokenizer: Tokenizer) -> None:
    prompts = throughput.build_prompts(tiny_tokenizer, 40, 5)
    assert len(prompts) == 5
    assert all(len(prompt) == 40 for prompt in prompts)
    assert len({tuple(prompt) for prompt in prompts}) == 5
    # Each prompt starts one token further into the same text.
    assert prompts[1][:-1] == prompts[0][1:]


def test_batch_timing_derives_the_per_step_metrics() -> None:
    timing = throughput.BatchTiming(batch_size=4, output_tokens=11, total_s=2.0, ttft_s=0.5)
    assert timing.step_s == pytest.approx(0.15)
    assert timing.output_tokens_per_s == pytest.approx(22.0)


def test_time_batch_times_every_row_to_the_requested_length(
    generator: TextGenerator, tiny_tokenizer: Tokenizer
) -> None:
    prompts = throughput.build_prompts(tiny_tokenizer, 16, 3)
    timing = throughput.time_batch(generator, prompts, 8, CPU)
    assert (timing.batch_size, timing.output_tokens) == (3, 8)
    assert 0 < timing.ttft_s < timing.total_s


def test_time_batch_rejects_lengths_it_cannot_measure(
    generator: TextGenerator, tiny_tokenizer: Tokenizer
) -> None:
    prompts = throughput.build_prompts(tiny_tokenizer, 16, 2)
    with pytest.raises(ValueError, match="at least 2 output tokens"):
        throughput.time_batch(generator, prompts, 1, CPU)
    with pytest.raises(ValueError, match="context is too short"):
        throughput.time_batch(generator, prompts, 200, CPU)


def test_measure_warms_up_and_times_every_batch_size(
    generator: TextGenerator, tiny_tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_sizes: list[int] = []
    time_batch = throughput.time_batch

    def recording_time_batch(generator, prompts, output_tokens, device):
        batch_sizes.append(len(prompts))
        return time_batch(generator, prompts, output_tokens, device)

    monkeypatch.setattr(throughput, "time_batch", recording_time_batch)
    settings = throughput.Settings(
        prompt_tokens=16, output_tokens=4, batch_sizes=(1, 3), repeats=2, warmup=1
    )
    results = throughput.measure(generator, tiny_tokenizer, settings, CPU)
    assert batch_sizes == [1, 1, 1, 3, 3, 3]
    assert [entry["batch_size"] for entry in results] == [1, 3]
    for entry in results:
        runs, summary = entry["runs"], entry["median"]
        assert isinstance(runs, list)
        assert isinstance(summary, dict)
        assert len(runs) == 2
        assert summary["total_s"] == median([run["total_s"] for run in runs])
        assert set(summary) == {"total_s", "ttft_s", "step_s", "output_tokens_per_s"}


def test_throughput_main_saves_a_result_file(
    tiny_model_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "throughput.json"
    options = ["--device", "cpu", "--dtype", "fp32", "--prompt-tokens", "16", "--repeats", "2"]
    args = ["--model", str(tiny_model_dir), *options, "--output-tokens", "4"]
    assert throughput.main([*args, "--batch-sizes", "1", "2", "--output", str(output)]) == 0

    result = json.loads(output.read_text())
    assert (result["benchmark"], result["engine"]) == ("throughput", "static")
    assert (result["prompt_tokens"], result["output_tokens"]) == (16, 4)
    assert (result["repeats"], result["warmup"]) == (2, 1)
    assert [entry["batch_size"] for entry in result["results"]] == [1, 2]
    printed = capsys.readouterr().out
    assert "tokens/s" in printed
    assert f"saved {output}" in printed


def test_the_default_throughput_run_covers_batch_sizes_1_to_64(
    tiny_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: list[throughput.Settings] = []

    def fake_measure(generator, tokenizer, settings, device):
        requested.append(settings)
        return []

    monkeypatch.setattr(throughput, "measure", fake_measure)
    output = tmp_path / "throughput.json"
    assert throughput.main(["--model", str(tiny_model_dir), "--output", str(output)]) == 0
    assert requested == [
        throughput.Settings(
            prompt_tokens=128,
            output_tokens=128,
            batch_sizes=(1, 2, 4, 8, 16, 32, 64),
            repeats=3,
            warmup=1,
        )
    ]


@pytest.fixture(scope="module")
def tiny_model(tiny_model_dir: Path) -> LlamaModel:
    return load_model(tiny_model_dir, device=CPU, dtype=torch.float32)


@pytest.fixture(scope="module")
def workload(tiny_tokenizer: Tokenizer) -> batching.Workload:
    return batching.build_workload(tiny_tokenizer, 5, (4, 12), (2, 9), seed=0)


def test_the_workload_draws_lengths_from_the_ranges(tiny_tokenizer: Tokenizer) -> None:
    workload = batching.build_workload(tiny_tokenizer, 50, (4, 12), (2, 9), seed=0)
    assert all(4 <= len(prompt) <= 12 for prompt in workload.prompts)
    assert all(2 <= n <= 9 for n in workload.output_tokens)
    # Both ends of each range occur, so the ranges include their ends.
    assert {len(prompt) for prompt in workload.prompts} >= {4, 12}
    assert set(workload.output_tokens) >= {2, 9}
    assert len({tuple(prompt) for prompt in workload.prompts}) == 50
    assert workload == batching.build_workload(tiny_tokenizer, 50, (4, 12), (2, 9), seed=0)
    assert workload != batching.build_workload(tiny_tokenizer, 50, (4, 12), (2, 9), seed=1)


def test_the_workload_asks_for_exactly_its_output_lengths(workload: batching.Workload) -> None:
    params = workload.params
    assert [p.max_tokens for p in params] == workload.output_tokens
    assert all(p.temperature == 0 and p.ignore_eos for p in params)
    assert workload.max_len == max(
        len(prompt) + n for prompt, n in zip(workload.prompts, workload.output_tokens, strict=True)
    )


def test_static_decode_use_counts_the_steps_of_ended_outputs() -> None:
    workload = batching.Workload(prompts=[[1]] * 5, output_tokens=[2, 4, 1, 1, 3])
    # Batches [2, 4] and [1, 1] run 4 and 1 steps per row; [3] runs alone.
    assert batching.static_decode_utilization(workload, 2) == pytest.approx(11 / 13)
    assert batching.static_decode_utilization(workload, 1) == 1.0


def test_run_timing_summarizes_throughput_and_latency() -> None:
    timing = batching.RunTiming(
        total_s=4.0, ttft_s=[0.5, 1.0, 1.5], e2e_s=[2.0, 3.0, 4.0], output_tokens=20
    )
    assert timing.output_tokens_per_s == 5.0
    assert timing.requests_per_s == 0.75
    summary = timing.summary()
    assert (summary["ttft_p50_s"], summary["e2e_p50_s"]) == (1.0, 3.0)
    assert summary["e2e_p99_s"] == pytest.approx(3.98)


def check_timing(timing: batching.RunTiming, workload: batching.Workload) -> None:
    assert timing.output_tokens == sum(workload.output_tokens)
    assert len(timing.ttft_s) == len(timing.e2e_s) == len(workload.prompts)
    for first, last in zip(timing.ttft_s, timing.e2e_s, strict=True):
        assert 0 < first <= last <= timing.total_s


def test_static_batching_runs_every_request_to_its_length(
    tiny_model: LlamaModel, tiny_tokenizer: Tokenizer, workload: batching.Workload
) -> None:
    timing = batching.time_static(TextGenerator(tiny_model, tiny_tokenizer), workload, 2, CPU)
    check_timing(timing, workload)
    # The third batch starts after the first two have ended.
    assert timing.ttft_s[4] >= max(timing.e2e_s[:4])


def test_continuous_batching_runs_every_request_to_its_length(
    tiny_model: LlamaModel, tiny_tokenizer: Tokenizer, workload: batching.Workload
) -> None:
    engine = LLMEngine(tiny_model, tiny_tokenizer, EngineConfig(max_num_seqs=2))
    check_timing(batching.time_continuous(engine, workload, CPU), workload)
    assert not engine.has_unfinished()


def test_batching_measures_both_engines_at_each_batch_size(
    tiny_model: LlamaModel, tiny_tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int]] = []
    time_static, time_continuous = batching.time_static, batching.time_continuous

    def recording_static(generator, workload, batch_size, device):
        calls.append(("static", batch_size))
        return time_static(generator, workload, batch_size, device)

    def recording_continuous(engine, workload, device):
        calls.append(("continuous", engine._scheduler._config.max_num_seqs))
        return time_continuous(engine, workload, device)

    monkeypatch.setattr(batching, "time_static", recording_static)
    monkeypatch.setattr(batching, "time_continuous", recording_continuous)
    settings = batching.Settings(
        num_requests=4,
        prompt_tokens=(4, 8),
        output_tokens=(2, 6),
        seed=0,
        batch_sizes=(1, 3),
        max_num_batched_tokens=32,
        repeats=2,
        warmup=1,
    )
    results = batching.measure(tiny_model, tiny_tokenizer, settings, CPU)
    assert (
        calls
        == [("static", 1)] * 3
        + [("continuous", 1)] * 3
        + [("static", 3)] * 3
        + [("continuous", 3)] * 3
    )
    assert [entry["batch_size"] for entry in results] == [1, 3]
    for entry in results:
        engines = entry["engines"]
        assert isinstance(engines, dict)
        assert list(engines) == ["static", "continuous"]
        for summary in engines.values():
            assert len(summary["runs"]) == 2
            assert summary["median"]["total_s"] == median([r["total_s"] for r in summary["runs"]])
    assert results[0]["static_decode_utilization"] == 1.0


def test_batching_main_saves_a_result_file(
    tiny_model_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "batching.json"
    options = ["--device", "cpu", "--dtype", "fp32", "--num-requests", "4", "--repeats", "1"]
    lengths = ["--prompt-tokens", "4", "8", "--output-tokens", "2", "6", "--warmup", "0"]
    args = ["--model", str(tiny_model_dir), *options, *lengths, "--batch-sizes", "2"]
    assert batching.main([*args, "--output", str(output)]) == 0

    result = json.loads(output.read_text())
    assert result["benchmark"] == "batching"
    assert (result["prompt_tokens"], result["output_tokens"]) == ([4, 8], [2, 6])
    assert (result["num_requests"], result["seed"], result["repeats"]) == (4, 0, 1)
    assert result["max_num_batched_tokens"] == EngineConfig().max_num_batched_tokens
    assert [entry["batch_size"] for entry in result["results"]] == [2]
    printed = capsys.readouterr().out
    assert "continuous" in printed
    assert f"saved {output}" in printed


def test_continuous_timing_can_count_the_requests_after_each_step(
    tiny_model: LlamaModel, tiny_tokenizer: Tokenizer, workload: batching.Workload
) -> None:
    engine = LLMEngine(tiny_model, tiny_tokenizer, EngineConfig(max_num_seqs=2))
    running: list[int] = []
    batching.time_continuous(engine, workload, CPU, running)
    assert max(running) == 2
    assert min(running) >= 1


def test_the_contiguous_layout_reserves_the_longest_request_per_sequence(
    workload: batching.Workload,
) -> None:
    config = capacity.engine_config("contiguous", 3 * workload.max_len + 1, workload, 64)
    assert (config.max_num_seqs, config.num_kv_blocks) == (3, 3)
    assert config.block_size == config.max_model_len == workload.max_len
    assert not config.enable_prefix_caching


def test_the_paged_layout_is_limited_by_memory_only(workload: batching.Workload) -> None:
    config = capacity.engine_config("paged", 100, workload, 64)
    assert config.block_size == 16
    assert config.num_kv_blocks == 6
    assert config.max_num_seqs == len(workload.prompts)
    assert config.max_model_len == workload.max_len
    assert not config.enable_prefix_caching


@pytest.mark.parametrize(
    ("layout", "kv_tokens", "message"),
    [
        ("paged", 10, "a budget of 10 tokens cannot hold a request of"),
        ("rows", 1000, "unknown layout 'rows'"),
    ],
)
def test_invalid_capacity_settings_raise(
    workload: batching.Workload, layout: str, kv_tokens: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        capacity.engine_config(layout, kv_tokens, workload, 64)


def test_a_capacity_run_counts_requests_and_preemptions(
    tiny_model: LlamaModel, tiny_tokenizer: Tokenizer
) -> None:
    workload = batching.build_workload(tiny_tokenizer, 8, (4, 12), (20, 30), seed=0)
    # Room for two of the longest requests, in blocks: all eight prompts fit at the start,
    # and their outputs do not.
    config = capacity.engine_config("paged", 2 * workload.max_len, workload, 64)
    engine = LLMEngine(tiny_model, tiny_tokenizer, config)
    first = capacity.run_once(engine, workload, CPU)
    assert first["peak_running"] > 2
    assert 1 <= first["mean_running"] <= first["peak_running"]
    assert first["preemptions"] > 0
    # Preemptions are counted per run, not since the engine started.
    second = capacity.run_once(engine, workload, CPU)
    assert second["preemptions"] == first["preemptions"]


def test_capacity_measures_both_layouts_at_each_budget(
    tiny_model: LlamaModel, tiny_tokenizer: Tokenizer
) -> None:
    settings = capacity.Settings(
        num_requests=4,
        prompt_tokens=(4, 8),
        output_tokens=(2, 6),
        seed=0,
        kv_tokens=(32, 64),
        max_num_batched_tokens=32,
        repeats=2,
        warmup=1,
    )
    workload, results = capacity.measure(tiny_model, tiny_tokenizer, settings, CPU)
    assert len(workload.prompts) == 4
    assert [entry["kv_tokens"] for entry in results] == [32, 64]
    # 2 layers, keys and values, 2 KV heads of 16 dimensions, 4 bytes each.
    assert [entry["kv_bytes"] for entry in results] == [32 * 512, 64 * 512]
    for entry, kv_tokens in zip(results, settings.kv_tokens, strict=True):
        layouts = entry["layouts"]
        assert isinstance(layouts, dict)
        assert list(layouts) == ["contiguous", "paged"]
        assert layouts["contiguous"]["max_num_seqs"] == kv_tokens // workload.max_len
        assert layouts["paged"]["num_kv_blocks"] == kv_tokens // 16
        for summary in layouts.values():
            assert len(summary["runs"]) == 2
            assert summary["median"]["total_s"] == median([r["total_s"] for r in summary["runs"]])


def prefix_settings(**overrides: object) -> prefix.Settings:
    settings = prefix.Settings(
        system_tokens=(20, 40),
        num_requests=3,
        user_tokens=(2, 5),
        output_tokens=3,
        seed=0,
        max_num_batched_tokens=64,
        repeats=2,
        warmup=1,
    )
    return dataclasses.replace(settings, **overrides)


def test_prefix_requests_share_only_the_system_prompt(tiny_tokenizer: Tokenizer) -> None:
    workload = prefix.build_workload(tiny_tokenizer, 40, prefix_settings(num_requests=8))
    system = build_prompt(tiny_tokenizer, 40)
    assert all(prompt[:40] == system for prompt in workload.prompts)
    assert all(2 <= len(prompt) - 40 <= 5 for prompt in workload.prompts)
    assert len({tuple(prompt) for prompt in workload.prompts}) == 8
    assert workload.output_tokens == [3] * 8


def test_a_prefix_run_holds_every_request_without_sharing(tiny_tokenizer: Tokenizer) -> None:
    workload = prefix.build_workload(tiny_tokenizer, 40, prefix_settings())
    config = prefix.engine_config(workload, 64, cache=True)
    assert config.enable_prefix_caching
    assert config.max_num_seqs == 3
    assert config.max_model_len == workload.max_len
    assert config.num_kv_blocks == 3 * -(-workload.max_len // 16)
    assert not prefix.engine_config(workload, 64, cache=False).enable_prefix_caching
    with pytest.raises(ValueError, match="need a token budget of at least that, got 40"):
        prefix.engine_config(workload, 40, cache=True)


def test_requests_one_at_a_time_are_timed_from_when_each_is_added(
    tiny_model: LlamaModel, tiny_tokenizer: Tokenizer
) -> None:
    workload = prefix.build_workload(tiny_tokenizer, 40, prefix_settings())
    engine = LLMEngine(tiny_model, tiny_tokenizer, prefix.engine_config(workload, 64, cache=True))
    timing = prefix.time_one_at_a_time(engine, workload, CPU)
    assert len(timing.ttft_s) == len(timing.e2e_s) == 3
    assert all(0 < ttft < e2e for ttft, e2e in zip(timing.ttft_s, timing.e2e_s, strict=True))
    assert sum(timing.e2e_s) <= timing.total_s
    assert timing.output_tokens == 9
    # Two full blocks of the system prompt are shared by the second and third requests.
    assert engine.num_prefix_hit_tokens == 2 * 32


def test_a_prefix_run_starts_with_an_empty_cache(
    tiny_model: LlamaModel, tiny_tokenizer: Tokenizer
) -> None:
    workload = prefix.build_workload(tiny_tokenizer, 40, prefix_settings())
    prompt_tokens = sum(len(prompt) for prompt in workload.prompts)
    for cache, hit_rate in [(True, 2 * 32 / prompt_tokens), (False, 0.0)]:
        config = prefix.engine_config(workload, 64, cache)
        for _ in range(2):
            timer = partial(prefix.time_one_at_a_time, workload=workload, device=CPU)
            summary = prefix.run_once(tiny_model, tiny_tokenizer, workload, config, timer)
            assert summary["prefix_hit_rate"] == hit_rate
            assert summary["preemptions"] == 0
            assert summary["ttft_mean_s"] > 0


def test_prefix_measures_both_arrivals_with_the_cache_off_and_on(
    tiny_model: LlamaModel, tiny_tokenizer: Tokenizer
) -> None:
    results = prefix.measure(tiny_model, tiny_tokenizer, prefix_settings(), CPU)
    assert [entry["system_tokens"] for entry in results] == [20, 40]
    for entry in results:
        arrivals = entry["arrivals"]
        assert isinstance(arrivals, dict)
        assert list(arrivals) == ["one_at_a_time", "all_at_once"]
        for caches in arrivals.values():
            assert list(caches) == ["off", "on"]
            for name, summary in caches.items():
                assert len(summary["runs"]) == 2
                rate = summary["median"]["prefix_hit_rate"]
                assert rate > 0 if name == "on" else rate == 0


def test_prefix_main_saves_a_result_file(
    tiny_model_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "prefix.json"
    options = ["--device", "cpu", "--dtype", "fp32", "--num-requests", "3", "--repeats", "1"]
    lengths = ["--system-tokens", "20", "--user-tokens", "2", "5", "--output-tokens", "3"]
    args = ["--model", str(tiny_model_dir), *options, *lengths, "--warmup", "0"]
    assert prefix.main([*args, "--max-num-batched-tokens", "64", "--output", str(output)]) == 0

    result = json.loads(output.read_text())
    assert result["benchmark"] == "prefix"
    assert (result["user_tokens"], result["output_tokens"]) == ([2, 5], 3)
    assert (result["num_requests"], result["seed"], result["repeats"]) == (3, 0, 1)
    assert (result["block_size"], result["max_num_batched_tokens"]) == (16, 64)
    assert [entry["system_tokens"] for entry in result["results"]] == [20]
    printed = capsys.readouterr().out
    assert "one_at_a_time" in printed
    assert f"saved {output}" in printed


def test_capacity_main_saves_a_result_file(
    tiny_model_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "capacity.json"
    options = ["--device", "cpu", "--dtype", "fp32", "--num-requests", "4", "--repeats", "1"]
    lengths = ["--prompt-tokens", "4", "8", "--output-tokens", "2", "6", "--warmup", "0"]
    args = ["--model", str(tiny_model_dir), *options, *lengths, "--kv-tokens", "48"]
    assert capacity.main([*args, "--output", str(output)]) == 0

    result = json.loads(output.read_text())
    assert result["benchmark"] == "capacity"
    assert (result["prompt_tokens"], result["output_tokens"]) == ([4, 8], [2, 6])
    assert (result["num_requests"], result["seed"], result["repeats"]) == (4, 0, 1)
    assert 6 <= result["max_len"] <= 14
    assert [entry["kv_tokens"] for entry in result["results"]] == [48]
    printed = capsys.readouterr().out
    assert "paged" in printed
    assert f"saved {output}" in printed


def test_poisson_arrivals_are_seeded_and_scale_with_the_rate() -> None:
    slow = serving.poisson_arrivals(1000, 1.0, seed=3)
    fast = serving.poisson_arrivals(1000, 4.0, seed=3)
    assert slow[0] == 0.0
    assert slow == sorted(slow)
    assert slow == serving.poisson_arrivals(1000, 1.0, seed=3)
    # The same gaps, four times shorter.
    assert fast == pytest.approx([t / 4 for t in slow])
    # The mean gap of an exponential distribution is 1 / rate.
    assert slow[-1] / 999 == pytest.approx(1.0, rel=0.1)
    with pytest.raises(ValueError, match="rate must be above 0"):
        serving.poisson_arrivals(10, 0.0, seed=3)


def serving_settings(**overrides: object) -> serving.Settings:
    settings = serving.Settings(
        model="org/model",
        revision="abc",
        device="cpu",
        dtype="fp32",
        servers=serving.SERVERS,
        rates=(1.0,),
        repeats=1,
        warmup_requests=2,
        kv_tokens=8192,
        batch_size=16,
        max_num_seqs=64,
    )
    return dataclasses.replace(settings, **overrides)


def test_continuous_and_paged_servers_get_the_same_kv_memory() -> None:
    settings = serving_settings()
    continuous = serving.server_command("continuous", settings, 512, 9000)
    paged = serving.server_command("paged", settings, 512, 9000)
    assert continuous[1:2] == paged[1:2] == ["serve"]
    # 16 rows of 512 tokens against 512 blocks of 16 tokens.
    assert continuous[-7:] == [
        *["--block-size", "512", "--num-kv-blocks", "16", "--max-num-seqs", "16"],
        "--no-prefix-caching",
    ]
    assert paged[-7:] == [
        *["--block-size", "16", "--num-kv-blocks", "512", "--max-num-seqs", "64"],
        "--no-prefix-caching",
    ]
    for command in (continuous, paged):
        assert command[command.index("--revision") :][:2] == ["--revision", "abc"]
        assert command[command.index("--max-model-len") :][:2] == ["--max-model-len", "512"]


def test_baseline_servers_batch_one_request_or_batch_size_requests() -> None:
    settings = serving_settings(revision=None)
    hf = serving.server_command("hf", settings, 512, 9000)
    static = serving.server_command("static", settings, 512, 9000)
    assert hf[1:5] == ["-m", "benchmarks.hf_baseline", "--engine", "hf"]
    assert hf[5:7] == ["--batch-size", "1"]
    assert static[3:7] == ["--engine", "static", "--batch-size", "16"]
    assert "--revision" not in hf
    with pytest.raises(ValueError, match="unknown server 'other'"):
        serving.server_command("other", settings, 512, 9000)


def test_a_serving_run_is_summarized_from_the_chunk_times() -> None:
    records = [
        # Sent at 0, chunks at 0.5, 0.6, 0.7: TTFT 0.5, E2E 0.7, TPOT 0.2 / 2.
        serving.RequestRecord(0.0, [0.5, 0.6, 0.7], prompt_tokens=10, output_tokens=3),
        # Sent at 1, chunks at 1.2 and 2.0, the second holding 3 tokens.
        serving.RequestRecord(1.0, [1.2, 2.0], prompt_tokens=20, output_tokens=4),
    ]
    summary = serving.summarize(records)
    assert summary["duration_s"] == 2.0
    assert summary["requests_per_s"] == 1.0
    assert summary["output_tokens_per_s"] == 3.5
    assert summary["total_tokens_per_s"] == 18.5
    assert summary["ttft_p50_s"] == pytest.approx((0.5 + 0.2) / 2)
    assert summary["e2e_p99_s"] == pytest.approx(0.7 + 0.99 * (1.0 - 0.7))
    assert summary["tpot_p50_s"] == pytest.approx((0.1 + 0.8 / 3) / 2)
    # Gaps 0.1, 0.1 and 0.8.
    assert summary["itl_p50_s"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("num_tokens", "window", "stride"),
    [(10, 4, 2), (9, 4, 2), (40, 16, 8), (40, 16, 16), (40, 16, 1), (5, 16, 8), (2, 4, 4)],
)
def test_sliding_windows_score_every_token_but_the_first_once(
    num_tokens: int, window: int, stride: int
) -> None:
    windows = perplexity.sliding_windows(num_tokens, window, stride)
    scored = [t for w in windows for t in range(w.end - w.num_scored, w.end)]
    assert scored == list(range(1, num_tokens))
    for index, w in enumerate(windows):
        assert w.start == index * stride
        assert w.end - w.start <= window
        assert w.num_scored >= 1
        if index > 0:
            # Every scored token sees at least window - stride tokens of its window first.
            assert w.end - w.num_scored - w.start >= window - stride


def test_sliding_windows_reject_bad_settings() -> None:
    with pytest.raises(ValueError, match="at least 2 tokens"):
        perplexity.sliding_windows(1, 4, 2)
    with pytest.raises(ValueError, match=r"stride must be in \[1, window=4\]"):
        perplexity.sliding_windows(10, 4, 5)
    with pytest.raises(ValueError, match="stride must be"):
        perplexity.sliding_windows(10, 4, 0)


def score_token_by_token(
    model: LlamaModel, token_ids: list[int], windows: list[perplexity.Window]
) -> tuple[float, list[int]]:
    """Each scored token's NLL and the model's top token, from one forward per token."""
    nll_sum = 0.0
    top: list[int] = []
    with torch.inference_mode():
        for w in windows:
            for target in range(w.end - w.num_scored, w.end):
                context = torch.tensor([token_ids[w.start : target]])
                logits = model(context, torch.arange(context.shape[1]).unsqueeze(0))[0]
                nll_sum -= float(torch.log_softmax(logits, dim=-1)[token_ids[target]])
                top.append(int(logits.argmax()))
    return nll_sum, top


def test_compare_scores_each_token_given_its_window(tiny_model_dir: Path) -> None:
    reference = load_model(tiny_model_dir, device=CPU, dtype=torch.float32)
    candidate = load_model(tiny_model_dir, device=CPU, dtype=torch.float32, quantization="int8")
    generator = torch.Generator().manual_seed(0)
    token_ids = torch.randint(0, reference.config.vocab_size, (40,), generator=generator).tolist()
    windows = perplexity.sliding_windows(len(token_ids), 16, 8)
    comparison = perplexity.compare(reference, candidate, token_ids, windows)

    reference_nll, reference_top = score_token_by_token(reference, token_ids, windows)
    candidate_nll, candidate_top = score_token_by_token(candidate, token_ids, windows)
    assert comparison.num_scored == 39
    assert comparison.reference.nll_sum == pytest.approx(reference_nll, rel=1e-5)
    assert comparison.candidate.nll_sum == pytest.approx(candidate_nll, rel=1e-5)
    assert comparison.reference_perplexity == pytest.approx(math.exp(reference_nll / 39))
    agreed = sum(a == b for a, b in zip(reference_top, candidate_top, strict=True))
    assert comparison.num_agreed == agreed
    assert comparison.agreement == agreed / 39
    assert comparison.perplexity_increase == pytest.approx(
        comparison.candidate_perplexity / comparison.reference_perplexity - 1
    )


def test_a_model_compared_with_itself_agrees_everywhere(tiny_model_dir: Path) -> None:
    model = load_model(tiny_model_dir, device=CPU, dtype=torch.float32)
    token_ids = list(range(3, 43))
    comparison = perplexity.compare(
        model, model, token_ids, perplexity.sliding_windows(len(token_ids), 16, 8)
    )
    assert comparison.agreement == 1.0
    assert comparison.perplexity_increase == 0.0


def test_compare_rejects_windows_longer_than_the_context(tiny_model_dir: Path) -> None:
    model = load_model(tiny_model_dir, device=CPU, dtype=torch.float32)
    windows = perplexity.sliding_windows(300, 200, 100)
    with pytest.raises(ValueError, match="window of 200 tokens exceeds the model's context of 128"):
        perplexity.compare(model, model, list(range(300)), windows)


def test_perplexity_main_saves_a_result_file(
    tiny_model_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(perplexity, "load_wikitext2", lambda: "the cat sat on the mat. " * 20)
    output = tmp_path / "perplexity.json"
    options = ["--device", "cpu", "--dtype", "fp32", "--window", "16", "--stride", "8"]
    args = ["--model", str(tiny_model_dir), *options, "--max-tokens", "50"]
    assert perplexity.main([*args, "--output", str(output)]) == 0

    result = json.loads(output.read_text())
    assert result["benchmark"] == "perplexity"
    assert (result["window"], result["stride"]) == (16, 8)
    assert (result["tokens"], result["scored_tokens"]) == (50, 49)
    assert result["dataset"]["revision"] == perplexity.WIKITEXT_REVISION
    assert result["dataset"]["tokens"] > 50
    results = result["results"]
    assert results["reference"] == "float32"
    assert set(results["models"]) == {"float32", "int8"}
    assert results["models"]["int8"]["perplexity"] > 1
    assert 0 <= results["top1_agreement"] <= 1
    printed = capsys.readouterr().out
    assert "top-1 agreement" in printed
    assert f"saved {output}" in printed
