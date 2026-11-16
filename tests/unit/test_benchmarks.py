import json
import subprocess
from pathlib import Path

import pytest
import torch

from benchmarks import latency
from benchmarks.common import REPO_ROOT, capture_environment, median, save_result
from kiyal.engine.generator import TextGenerator
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
    json.dumps(environment)


def test_median_is_the_middle_run() -> None:
    assert median([3.0, 1.0, 10.0]) == 3.0


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
