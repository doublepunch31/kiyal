import asyncio
import sys
from pathlib import Path

import httpx
import pytest

from benchmarks import serving
from benchmarks.batching import Workload
from tests.server.conftest import MODEL_NAME, Server


def test_the_client_records_every_request_of_a_workload(
    server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(serving, "SERVED_MODEL_NAME", MODEL_NAME)
    workload = Workload([[5, 6, 7], [8] * 20, [9, 10]], [4, 12, 1])
    arrivals = [0.0, 0.05, 0.1]
    records = asyncio.run(serving.run_workload(server.base_url, workload, arrivals))
    assert [record.output_tokens for record in records] == [4, 12, 1]
    assert [record.prompt_tokens for record in records] == [3, 20, 2]
    for record, arrival in zip(records, arrivals, strict=True):
        assert record.sent >= arrival
        assert record.chunk_times == sorted(record.chunk_times)
        assert record.chunk_times[0] > record.sent
        # One chunk per token, or fewer where text was held back.
        assert 1 <= len(record.chunk_times) <= record.output_tokens
    assert serving.summarize(records)["output_tokens_per_s"] > 0


def test_an_error_response_fails_the_run(server: Server) -> None:
    # The served model is not called "model", so every request is a 404.
    workload = Workload([[5, 6]], [2])
    with pytest.raises(RuntimeError, match="the server answered 404"):
        asyncio.run(serving.run_workload(server.base_url, workload, [0.0]))


def test_a_server_process_is_started_and_stopped(tiny_model_dir: Path, tmp_path: Path) -> None:
    port = serving.free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        *[str(Path(sys.executable).with_name("kiyal")), "serve"],
        *["--model", str(tiny_model_dir), "--device", "cpu", "--port", str(port)],
    ]
    log_path = tmp_path / "server.log"
    with serving.running_server(command, base_url, log_path):
        assert httpx.get(f"{base_url}/health").status_code == 200
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"{base_url}/health")
    assert "Application shutdown complete" in log_path.read_text()


def test_a_server_that_exits_early_is_reported(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "raise SystemExit(3)"]
    with (
        pytest.raises(RuntimeError, match="the server exited with 3"),
        serving.running_server(command, "http://127.0.0.1:9", tmp_path / "server.log"),
    ):
        pass
