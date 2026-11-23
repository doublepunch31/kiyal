import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from openai import OpenAI

from kiyal.config import EngineConfig
from kiyal.engine.async_engine import AsyncEngine
from kiyal.engine.llm_engine import LLMEngine
from kiyal.engine.sequence import RequestOutput
from kiyal.model.llama import LlamaModel
from kiyal.server.app import build_app
from kiyal.tokenizer import Tokenizer
from tests.server.conftest import MODEL_NAME, Server
from tests.server.serving import serving

PROMPT = "the cat sat on"


def data_lines(lines: Iterator[str]) -> Iterator[str]:
    return (line for line in lines if line.startswith("data: "))


@pytest.mark.parametrize(
    "fields",
    [
        {"max_tokens": 30, "temperature": 0},
        {"max_tokens": 30, "seed": 4},
        {"max_tokens": 40, "temperature": 0, "stop": ["the"]},
    ],
)
def test_a_stream_carries_the_whole_completion(
    client: OpenAI, chat_prompt: list[int], fields: dict[str, Any]
) -> None:
    whole = client.completions.create(model=MODEL_NAME, prompt=chat_prompt, **fields)
    chunks = list(
        client.completions.create(model=MODEL_NAME, prompt=chat_prompt, stream=True, **fields)
    )
    assert "".join(chunk.choices[0].text for chunk in chunks) == whole.choices[0].text
    assert [chunk.choices[0].finish_reason for chunk in chunks] == [None] * (len(chunks) - 1) + [
        whole.choices[0].finish_reason
    ]
    [request_id] = {chunk.id for chunk in chunks}
    assert request_id.startswith("cmpl-")
    assert all(chunk.object == "text_completion" for chunk in chunks)
    assert all(chunk.usage is None for chunk in chunks)


def test_the_usage_comes_last_when_asked_for(client: OpenAI, tiny_tokenizer: Tokenizer) -> None:
    chunks = list(
        client.completions.create(
            model=MODEL_NAME,
            prompt=PROMPT,
            max_tokens=10,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"ignore_eos": True},
        )
    )
    *pieces, last = chunks
    assert last.choices == []
    assert last.usage is not None
    num_prompt_tokens = len(tiny_tokenizer.encode(PROMPT))
    assert (last.usage.prompt_tokens, last.usage.completion_tokens) == (num_prompt_tokens, 10)
    assert all(piece.usage is None for piece in pieces)


def test_events_follow_the_server_sent_events_format(server: Server) -> None:
    body = {"model": MODEL_NAME, "prompt": PROMPT, "max_tokens": 5, "stream": True}
    with httpx.stream("POST", f"{server.base_url}/v1/completions", json=body) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = response.read().decode().split("\n\n")
    assert events[-2:] == ["data: [DONE]", ""]
    for event in events[:-2]:
        assert event.startswith("data: ")
        assert json.loads(event.removeprefix("data: "))["object"] == "text_completion"


def test_pieces_arrive_while_the_request_runs(server: Server) -> None:
    body = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "max_tokens": 1500,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
    }
    start = time.perf_counter()
    with httpx.stream("POST", f"{server.base_url}/v1/completions", json=body) as response:
        lines = data_lines(response.iter_lines())
        next(lines)
        first = time.perf_counter() - start
        assert list(lines)[-1] == "data: [DONE]"
    total = time.perf_counter() - start
    # The first piece comes after a few of the 1500 steps, not after all of them.
    assert first < 0.5 * total


def test_a_rejected_request_gets_an_error_response_instead_of_a_stream(client: OpenAI) -> None:
    with pytest.raises(openai.BadRequestError, match="must also fit at least one output token"):
        client.completions.create(model=MODEL_NAME, prompt=[5] * 2048, stream=True)


def test_stream_options_need_streaming(server: Server) -> None:
    body = {"model": MODEL_NAME, "prompt": PROMPT, "stream_options": {"include_usage": True}}
    response = httpx.post(f"{server.base_url}/v1/completions", json=body)
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "stream_options"


def test_an_engine_that_fails_mid_stream_ends_it_with_an_error_event(
    model: LlamaModel, tiny_long_model_dir: Path
) -> None:
    engine = LLMEngine(model, Tokenizer.from_dir(tiny_long_model_dir), EngineConfig(max_num_seqs=2))
    step = engine.step
    num_steps = 0

    def failing_step() -> list[RequestOutput]:
        nonlocal num_steps
        num_steps += 1
        if num_steps > 5:
            raise RuntimeError("device lost")
        return step()

    engine.step = failing_step
    app = build_app(AsyncEngine(engine), Tokenizer.from_dir(tiny_long_model_dir), MODEL_NAME)
    body = {"model": MODEL_NAME, "prompt": PROMPT, "max_tokens": 20, "stream": True}
    with serving(app) as base_url:
        with httpx.stream("POST", f"{base_url}/v1/completions", json=body) as response:
            assert response.status_code == 200
            events = list(data_lines(response.iter_lines()))
        assert events[-1] == "data: [DONE]"
        error = json.loads(events[-2].removeprefix("data: "))["error"]
        assert (error["type"], error["message"]) == ("server_error", "the engine has stopped")
        assert httpx.get(f"{base_url}/health").status_code == 503
        response = httpx.post(f"{base_url}/v1/completions", json={**body, "stream": False})
        assert response.status_code == 503
