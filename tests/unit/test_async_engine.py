import asyncio
import contextlib
from collections.abc import AsyncGenerator, Callable
from pathlib import Path

import pytest
import torch

from kiyal.config import EngineConfig
from kiyal.engine.async_engine import AsyncEngine, EngineStats, EngineStoppedError
from kiyal.engine.generator import TextGenerator
from kiyal.engine.llm_engine import LLMEngine
from kiyal.engine.sequence import FinishReason, RequestOutput, SamplingParams
from kiyal.model.llama import LlamaModel
from kiyal.model.loader import load_model
from kiyal.tokenizer import Tokenizer

CPU = torch.device("cpu")
GREEDY = SamplingParams(max_tokens=12, temperature=0, ignore_eos=True)
# About half a second of steps on a CPU, so the request is still running when a test acts on
# it a few steps after it starts.
LONG = SamplingParams(max_tokens=1500, temperature=0, ignore_eos=True)
TEXTS = ["the cat sat on", "мама мыла раму", "кыял деген эмне", "你好世界"]


@pytest.fixture(scope="module")
def model(tiny_long_model_dir: Path) -> LlamaModel:
    return load_model(tiny_long_model_dir, device=CPU, dtype=torch.float32)


@pytest.fixture
def engine(model: LlamaModel, tiny_tokenizer: Tokenizer) -> LLMEngine:
    return LLMEngine(model, tiny_tokenizer, EngineConfig(max_num_seqs=4))


@pytest.fixture(scope="module")
def prompts(tiny_tokenizer: Tokenizer) -> list[list[int]]:
    return [tiny_tokenizer.encode(text) for text in TEXTS]


@contextlib.asynccontextmanager
async def started(engine: LLMEngine) -> AsyncGenerator[AsyncEngine, None]:
    async_engine = AsyncEngine(engine)
    async_engine.start()
    try:
        yield async_engine
    finally:
        async_engine.shutdown()


async def collect(
    engine: AsyncEngine, request_id: str, prompt: list[int], params: SamplingParams
) -> list[RequestOutput]:
    return [output async for output in engine.generate(request_id, prompt, params)]


async def wait_for(engine: AsyncEngine, condition: Callable[[EngineStats], bool]) -> EngineStats:
    """Polls the engine's stats until the condition holds, for up to 5 seconds."""
    stats = engine.stats()
    for _ in range(500):
        stats = engine.stats()
        if condition(stats):
            return stats
        await asyncio.sleep(0.01)
    raise AssertionError(f"the engine did not get there; its last stats were {stats}")


def is_idle(stats: EngineStats) -> bool:
    return (
        stats.num_running == 0
        and stats.num_waiting == 0
        and stats.num_free_kv_blocks == stats.num_kv_blocks
    )


def test_concurrent_requests_get_what_they_get_alone(
    engine: LLMEngine, model: LlamaModel, tiny_tokenizer: Tokenizer, prompts: list[list[int]]
) -> None:
    async def scenario() -> list[list[RequestOutput]]:
        async with started(engine) as async_engine:
            return await asyncio.gather(
                *(collect(async_engine, str(i), p, GREEDY) for i, p in enumerate(prompts))
            )

    static = TextGenerator(model, tiny_tokenizer)
    for outputs, prompt in zip(asyncio.run(scenario()), prompts, strict=True):
        expected = static.generate(prompt, GREEDY)
        assert [(o.token_id, o.text, o.finish_reason) for o in outputs] == [
            (s.token_id, s.text, s.finish_reason) for s in expected
        ]


def test_a_finished_request_leaves_the_engine_idle(
    engine: LLMEngine, prompts: list[list[int]]
) -> None:
    async def scenario() -> EngineStats:
        async with started(engine) as async_engine:
            await collect(async_engine, "a", prompts[0], GREEDY)
            return await wait_for(async_engine, is_idle)

    assert asyncio.run(scenario()).num_aborted == 0


def test_a_rejected_request_raises_and_the_engine_keeps_serving(
    engine: LLMEngine, prompts: list[list[int]]
) -> None:
    async def scenario() -> list[RequestOutput]:
        async with started(engine) as async_engine:
            with pytest.raises(ValueError, match="must also fit at least one output token"):
                await collect(async_engine, "a", [5] * engine.max_model_len, GREEDY)
            with pytest.raises(ValueError, match="token ids outside"):
                await collect(async_engine, "a", [-1], GREEDY)
            return await collect(async_engine, "a", prompts[0], GREEDY)

    outputs = asyncio.run(scenario())
    assert len(outputs) == GREEDY.max_tokens
    assert outputs[-1].finish_reason is FinishReason.LENGTH


def test_a_request_id_in_use_raises(engine: LLMEngine, prompts: list[list[int]]) -> None:
    async def scenario() -> None:
        async with started(engine) as async_engine:
            outputs = async_engine.generate("a", prompts[0], LONG)
            await anext(outputs)
            with pytest.raises(ValueError, match="request a is already running"):
                await anext(async_engine.generate("a", prompts[1], GREEDY))
            await outputs.aclose()

    asyncio.run(scenario())


def test_leaving_the_iteration_aborts_the_request(
    engine: LLMEngine, prompts: list[list[int]]
) -> None:
    async def scenario() -> EngineStats:
        async with started(engine) as async_engine:
            outputs = async_engine.generate("a", prompts[0], LONG)
            await anext(outputs)
            await outputs.aclose()
            return await wait_for(async_engine, lambda stats: stats.num_aborted == 1)

    assert is_idle(asyncio.run(scenario()))


def test_cancelling_the_task_aborts_the_request(
    engine: LLMEngine, prompts: list[list[int]]
) -> None:
    async def scenario() -> EngineStats:
        async with started(engine) as async_engine:
            task = asyncio.create_task(collect(async_engine, "a", prompts[0], LONG))
            await wait_for(async_engine, lambda stats: stats.num_running == 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return await wait_for(async_engine, lambda stats: stats.num_aborted == 1)

    assert is_idle(asyncio.run(scenario()))


def test_stats_count_waiting_requests(
    model: LlamaModel, tiny_tokenizer: Tokenizer, prompts: list[list[int]]
) -> None:
    # Room for two long requests, but only one runs at a time.
    engine = LLMEngine(model, tiny_tokenizer, EngineConfig(max_num_seqs=1, num_kv_blocks=256))

    async def scenario() -> None:
        async with started(engine) as async_engine:
            first = async_engine.generate("a", prompts[0], LONG)
            second = async_engine.generate("b", prompts[1], LONG)
            await anext(first)
            # The second request's command is queued at its first iteration.
            pending = asyncio.ensure_future(anext(second))
            await wait_for(async_engine, lambda s: (s.num_running, s.num_waiting) == (1, 1))
            pending.cancel()
            await first.aclose()
            await wait_for(async_engine, is_idle)

    asyncio.run(scenario())


def test_shutdown_ends_unfinished_requests(engine: LLMEngine, prompts: list[list[int]]) -> None:
    async def scenario() -> None:
        async_engine = AsyncEngine(engine)
        async_engine.start()
        outputs = async_engine.generate("a", prompts[0], LONG)
        await anext(outputs)
        async_engine.shutdown()
        assert not async_engine.is_running
        with pytest.raises(EngineStoppedError, match="the engine has stopped"):
            async for _ in outputs:
                pass
        with pytest.raises(EngineStoppedError, match="the engine is not running"):
            await anext(async_engine.generate("b", prompts[0], GREEDY))

    asyncio.run(scenario())


def test_a_failing_step_stops_the_engine(
    engine: LLMEngine, prompts: list[list[int]], monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_step() -> list[RequestOutput]:
        raise RuntimeError("out of memory")

    monkeypatch.setattr(engine, "step", failing_step)

    async def scenario() -> None:
        async with started(engine) as async_engine:
            with pytest.raises(EngineStoppedError) as exc_info:
                await collect(async_engine, "a", prompts[0], GREEDY)
            assert isinstance(exc_info.value.__cause__, RuntimeError)
            assert not async_engine.is_running
            with pytest.raises(EngineStoppedError, match="the engine is not running"):
                await collect(async_engine, "b", prompts[0], GREEDY)

    asyncio.run(scenario())


def test_a_request_before_start_raises(engine: LLMEngine, prompts: list[list[int]]) -> None:
    async def scenario() -> None:
        with pytest.raises(EngineStoppedError, match="the engine is not running"):
            await collect(AsyncEngine(engine), "a", prompts[0], GREEDY)

    asyncio.run(scenario())


def test_the_engine_starts_once(engine: LLMEngine) -> None:
    async def scenario() -> None:
        async with started(engine) as async_engine:
            with pytest.raises(RuntimeError, match="has been started before"):
                async_engine.start()

    asyncio.run(scenario())
    with pytest.raises(RuntimeError, match="no running event loop"):
        AsyncEngine(engine).start()
