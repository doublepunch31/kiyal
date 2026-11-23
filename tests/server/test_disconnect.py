import asyncio
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.types import Message

from kiyal.config import EngineConfig
from kiyal.engine.async_engine import AsyncEngine
from kiyal.engine.llm_engine import LLMEngine
from kiyal.model.llama import LlamaModel
from kiyal.server.app import EventStreamResponse, build_app
from kiyal.tokenizer import Tokenizer
from tests.server.conftest import MODEL_NAME, Server
from tests.server.serving import is_idle, serving, wait_for

# Long enough that a client always leaves before the request ends: 2000 steps take most of
# a second on a CPU.
LONG: dict[str, Any] = {"max_tokens": 2000, "temperature": 0, "ignore_eos": True}
BODIES: dict[str, dict[str, Any]] = {
    "completions": {"model": MODEL_NAME, "prompt": "the cat sat on", **LONG},
    "chat/completions": {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "the cat"}],
        **LONG,
    },
}


@pytest.mark.parametrize("endpoint", BODIES)
def test_closing_a_stream_aborts_its_request(server: Server, endpoint: str) -> None:
    aborted = server.engine.stats().num_aborted
    url = f"{server.base_url}/v1/{endpoint}"
    with httpx.stream("POST", url, json={**BODIES[endpoint], "stream": True}) as response:
        lines = response.iter_lines()
        assert next(lines).startswith("data: ")
        # Leaving the block closes the connection with the rest of the stream unread.
    wait_for(server.engine, lambda stats: stats.num_aborted == aborted + 1)
    wait_for(server.engine, is_idle)


@pytest.mark.parametrize("endpoint", BODIES)
def test_a_client_that_leaves_before_the_response_aborts_its_request(
    server: Server, endpoint: str
) -> None:
    aborted = server.engine.stats().num_aborted
    with pytest.raises(httpx.ReadTimeout):
        httpx.post(f"{server.base_url}/v1/{endpoint}", json=BODIES[endpoint], timeout=0.2)
    wait_for(server.engine, lambda stats: stats.num_aborted == aborted + 1)
    wait_for(server.engine, is_idle)


@pytest.fixture(scope="module")
def one_seat_server(model: LlamaModel, tiny_long_model_dir: Path) -> Iterator[Server]:
    """A server whose batch holds one request, so the next one waits."""
    config = EngineConfig(max_num_seqs=1, num_kv_blocks=256)
    engine = AsyncEngine(LLMEngine(model, Tokenizer.from_dir(tiny_long_model_dir), config))
    with serving(build_app(engine, Tokenizer.from_dir(tiny_long_model_dir), MODEL_NAME)) as url:
        yield Server(url, engine)


def test_a_stream_still_waiting_for_its_first_token_is_aborted(
    one_seat_server: Server, tiny_tokenizer: Tokenizer
) -> None:
    engine = one_seat_server.engine
    url = f"{one_seat_server.base_url}/v1/completions"
    body = {**BODIES["completions"], "stream": True}
    with httpx.stream("POST", url, json=body) as running:
        # Kept until the end: httpx closes the connection when the line iterator goes away.
        # Blank lines separate the events.
        lines = (line for line in running.iter_lines() if line)
        next(lines)
        # The batch is full, so this request waits, and not even its status line is sent.
        with pytest.raises(httpx.ReadTimeout):
            httpx.post(url, json=body, timeout=0.2)
        stats = wait_for(engine, lambda stats: stats.num_aborted == 1)
        assert (stats.num_running, stats.num_waiting) == (1, 0)
        # Only the running request ever started.
        assert stats.num_prefix_query_tokens == len(tiny_tokenizer.encode(body["prompt"]))
        assert next(lines).startswith("data: ")
    wait_for(engine, lambda stats: stats.num_aborted == 2 and is_idle(stats))


def test_an_event_stream_closes_its_events_when_the_client_leaves_during_a_send() -> None:
    closed: list[bool] = []

    async def events() -> AsyncGenerator[str, None]:
        try:
            yield "data: 1\n\n"
            yield "data: 2\n\n"
        finally:
            closed.append(True)

    async def scenario() -> list[bool]:
        sending = asyncio.Event()

        async def receive() -> Message:
            await sending.wait()
            return {"type": "http.disconnect"}

        async def send(message: Message) -> None:
            if message["type"] == "http.response.body":
                # The client has stopped reading, so the send never completes.
                sending.set()
                await asyncio.sleep(3600)

        generator = events()
        scope = {"type": "http", "asgi": {"spec_version": "2.3"}}
        await EventStreamResponse(generator)(scope, receive, send)
        # Checked here: asyncio.run closes leftover generators itself when it ends.
        return list(closed)

    assert asyncio.run(scenario()) == [True]
