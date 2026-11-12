import shutil
from pathlib import Path

import pytest
import torch

from kiyal import __version__
from kiyal.cli import main
from kiyal.engine.naive import NaiveGenerator
from kiyal.engine.sequence import SamplingParams
from kiyal.model.loader import load_model
from kiyal.tokenizer import Tokenizer


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    assert capsys.readouterr().out.startswith("usage: kiyal")


def test_version_prints_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"kiyal {__version__}"


def test_no_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert capsys.readouterr().out.startswith("usage: kiyal")


def test_unknown_option_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--no-such-option"])
    assert exc_info.value.code == 2
    assert "unrecognized arguments: --no-such-option" in capsys.readouterr().err


def run_generate(model_dir: Path, *args: str) -> list[str]:
    return ["generate", "--model", str(model_dir), "--device", "cpu", "--dtype", "fp32", *args]


def expected_text(model_dir: Path, prompt_token_ids: list[int], params: SamplingParams) -> str:
    """What the engine generates for a prompt, bypassing the command line."""
    model = load_model(model_dir, device=torch.device("cpu"), dtype=torch.float32)
    generator = NaiveGenerator(model, Tokenizer.from_dir(model_dir))
    return "".join(step.text for step in generator.generate(prompt_token_ids, params))


def test_generate_prints_the_reply_to_a_chat_prompt(
    tiny_model_dir: Path, tiny_tokenizer: Tokenizer, capsys: pytest.CaptureFixture[str]
) -> None:
    args = run_generate(tiny_model_dir, "--max-tokens", "12", "--temperature", "0", "the cat")
    assert main(args) == 0
    prompt = tiny_tokenizer.encode_chat([{"role": "user", "content": "the cat"}])
    params = SamplingParams(max_tokens=12, temperature=0)
    captured = capsys.readouterr()
    assert captured.out == expected_text(tiny_model_dir, prompt, params) + "\n"
    assert "tokens/s" in captured.err
    assert "finish reason: " in captured.err


def test_generate_places_the_system_message_first(
    tiny_model_dir: Path, tiny_tokenizer: Tokenizer, capsys: pytest.CaptureFixture[str]
) -> None:
    args = run_generate(
        tiny_model_dir, "--max-tokens", "12", "--temperature", "0", "--system", "sat", "the cat"
    )
    assert main(args) == 0
    messages = [{"role": "system", "content": "sat"}, {"role": "user", "content": "the cat"}]
    params = SamplingParams(max_tokens=12, temperature=0)
    expected = expected_text(tiny_model_dir, tiny_tokenizer.encode_chat(messages), params)
    assert capsys.readouterr().out == expected + "\n"


def test_generate_raw_continues_the_plain_text(
    tiny_model_dir: Path, tiny_tokenizer: Tokenizer, capsys: pytest.CaptureFixture[str]
) -> None:
    args = run_generate(tiny_model_dir, "--raw", "--max-tokens", "12", "--temperature", "0", "the")
    assert main(args) == 0
    params = SamplingParams(max_tokens=12, temperature=0)
    expected = expected_text(tiny_model_dir, tiny_tokenizer.encode("the"), params)
    assert capsys.readouterr().out == expected + "\n"


def test_generate_passes_sampling_options_to_the_engine(
    tiny_model_dir: Path, tiny_tokenizer: Tokenizer, capsys: pytest.CaptureFixture[str]
) -> None:
    params = SamplingParams(
        max_tokens=30,
        temperature=1.3,
        top_k=50,
        top_p=0.9,
        min_p=0.01,
        repetition_penalty=1.2,
        presence_penalty=0.3,
        frequency_penalty=0.2,
        seed=7,
    )
    prompt = tiny_tokenizer.encode("the cat")
    full_text = expected_text(tiny_model_dir, prompt, params)
    stop = full_text[len(full_text) // 2 :][:3]
    options = [
        *("--max-tokens", "30", "--temperature", "1.3", "--top-k", "50", "--top-p", "0.9"),
        *("--min-p", "0.01", "--repetition-penalty", "1.2", "--presence-penalty", "0.3"),
        *("--frequency-penalty", "0.2", "--seed", "7", "--raw"),
    ]
    assert main(run_generate(tiny_model_dir, *options, "the cat")) == 0
    assert capsys.readouterr().out == full_text + "\n"

    assert main(run_generate(tiny_model_dir, *options, "--stop", stop, "the cat")) == 0
    captured = capsys.readouterr()
    assert captured.out == full_text[: full_text.index(stop)] + "\n"
    assert "finish reason: stop" in captured.err


def test_generate_without_a_chat_template_continues_the_plain_text(
    tiny_model_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_dir = tmp_path / "no-template"
    shutil.copytree(tiny_model_dir, model_dir)
    (model_dir / "chat_template.jinja").unlink()
    options = ("--max-tokens", "12", "--temperature", "0", "the cat")
    assert main(run_generate(model_dir, *options)) == 0
    without_template = capsys.readouterr().out
    assert main(run_generate(tiny_model_dir, "--raw", *options)) == 0
    assert without_template == capsys.readouterr().out

    assert main(run_generate(model_dir, "--system", "sat", *options)) == 1
    assert "--system needs the model's chat template" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--top-p", "0"], "top_p must be in"),
        (["--max-tokens", "0"], "max_tokens must be"),
        (["--stop", ""], "stop must be"),
        (["--raw", "--system", "sat"], "--system needs the model's chat template"),
        (["--device", "cuda:99"], "cuda:99"),
        (["--dtype", "fp8"], "unsupported dtype"),
    ],
)
def test_generate_reports_invalid_options(
    tiny_model_dir: Path, capsys: pytest.CaptureFixture[str], options: list[str], message: str
) -> None:
    assert main(["generate", "--model", str(tiny_model_dir), *options, "the cat"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("kiyal: error: ")
    assert message in captured.err


def test_generate_reports_a_missing_model_directory(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["generate", "--model", "./no-such-model", "the cat"]) == 1
    assert "kiyal: error: model directory no-such-model does not exist" in capsys.readouterr().err


def test_generate_reports_a_directory_without_a_tokenizer(
    tiny_checkpoint: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["generate", "--model", str(tiny_checkpoint), "the cat"]) == 1
    assert "no tokenizer.json" in capsys.readouterr().err


def test_generate_requires_a_model(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["generate", "the cat"])
    assert exc_info.value.code == 2
    assert "--model" in capsys.readouterr().err
