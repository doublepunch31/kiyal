import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import torch
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionStreamOptionsParam

from kiyal.config import EngineConfig
from kiyal.engine.async_engine import AsyncEngine
from kiyal.engine.generator import TextGenerator
from kiyal.engine.llm_engine import LLMEngine
from kiyal.engine.sequence import SamplingParams
from kiyal.model.llama import LlamaModel
from kiyal.model.loader import load_model
from kiyal.server.app import build_app
from kiyal.tokenizer import Tokenizer
from tests.equivalence.greedy import near_tie_steps, single_greedy
from tests.equivalence.prompts import SMOLLM2_PROMPTS, TINY_TEXTS, random_prompts
from tests.server.conftest import MODEL_NAME, Server
from tests.server.serving import serving

# As in the equivalence ladder: the largest logit difference allowed between a batched run
# and a prompt decoded alone.
TINY_ATOL = 2e-5
SMOLLM2_ATOL = 1e-3


@dataclass(frozen=True)
class Request:
    """A greedy request that ignores EOS: a completion of a text or token-id prompt, or,
    with messages, a chat reply."""

    max_tokens: int
    stream: bool
    prompt: str | list[int] = ""
    messages: list[dict[str, str]] | None = None

    def token_ids(self, tokenizer: Tokenizer) -> list[int]:
        if self.messages is not None:
            return tokenizer.encode_chat(self.messages)
        return tokenizer.encode(self.prompt) if isinstance(self.prompt, str) else self.prompt


IGNORE_EOS = {"ignore_eos": True}
INCLUDE_USAGE: ChatCompletionStreamOptionsParam = {"include_usage": True}


async def send(client: AsyncOpenAI, request: Request) -> tuple[str, int | None]:
    """Sends a request and returns its text and its number of output tokens."""
    if request.messages is not None:
        # The dicts have the shape of OpenAI's message types.
        messages = cast(list[ChatCompletionMessageParam], request.messages)
        return await send_chat(client, request, messages)
    if not request.stream:
        completion = await client.completions.create(
            model=MODEL_NAME,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=0,
            extra_body=IGNORE_EOS,
        )
        usage = completion.usage
        return completion.choices[0].text, None if usage is None else usage.completion_tokens
    pieces: list[str] = []
    num_tokens = None
    async for chunk in await client.completions.create(
        model=MODEL_NAME,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=0,
        stream=True,
        stream_options=INCLUDE_USAGE,
        extra_body=IGNORE_EOS,
    ):
        if chunk.choices:
            pieces.append(chunk.choices[0].text)
        if chunk.usage is not None:
            num_tokens = chunk.usage.completion_tokens
    return "".join(pieces), num_tokens


async def send_chat(
    client: AsyncOpenAI, request: Request, messages: list[ChatCompletionMessageParam]
) -> tuple[str, int | None]:
    if not request.stream:
        reply = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=0,
            extra_body=IGNORE_EOS,
        )
        usage = reply.usage
        text = reply.choices[0].message.content or ""
        return text, None if usage is None else usage.completion_tokens
    pieces: list[str] = []
    num_tokens = None
    async for chunk in await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=0,
        stream=True,
        stream_options=INCLUDE_USAGE,
        extra_body=IGNORE_EOS,
    ):
        if chunk.choices:
            pieces.append(chunk.choices[0].delta.content or "")
        if chunk.usage is not None:
            num_tokens = chunk.usage.completion_tokens
    return "".join(pieces), num_tokens


def send_all(base_url: str, requests: list[Request]) -> list[tuple[str, int | None]]:
    """Sends all requests at once, so the engine batches them."""

    async def scenario() -> list[tuple[str, int | None]]:
        async with AsyncOpenAI(
            base_url=f"{base_url}/v1", api_key="unused", max_retries=0
        ) as client:
            return await asyncio.gather(*(send(client, request) for request in requests))

    return asyncio.run(scenario())


def assert_each_gets_its_single_run(
    model: LlamaModel,
    tokenizer: Tokenizer,
    requests: list[Request],
    results: list[tuple[str, int | None]],
    atol: float,
) -> None:
    generator = TextGenerator(model, tokenizer)
    for request, (text, num_tokens) in zip(requests, results, strict=True):
        prompt = request.token_ids(tokenizer)
        logits, _ = single_greedy(model, prompt, request.max_tokens)
        # Batching changes the logits in the last bits, which can flip a near tie. None of
        # these prompts has one, so every token must match.
        assert not near_tie_steps(logits, atol).any(), "pick prompts without near ties"
        params = SamplingParams(max_tokens=request.max_tokens, temperature=0, ignore_eos=True)
        expected = "".join(step.text for step in generator.generate(prompt, params))
        assert (text, num_tokens) == (expected, request.max_tokens)


def test_concurrent_requests_get_what_they_get_alone(
    server: Server, model: LlamaModel, tiny_tokenizer: Tokenizer
) -> None:
    prompts: list[str | list[int]] = [*TINY_TEXTS.values()]
    prompts += random_prompts(tiny_tokenizer.vocab_size, [3, 30])
    conversations = [
        [{"role": "user", "content": "the cat"}],
        [{"role": "system", "content": "кыял"}, {"role": "user", "content": "你好"}],
    ]
    completion_lengths = [24, 40, 16, 32, 48, 20]
    chat_lengths = [36, 28]
    requests = [
        Request(n, stream=i % 2 == 0, prompt=prompt)
        for i, (n, prompt) in enumerate(zip(completion_lengths, prompts, strict=True))
    ]
    requests += [
        Request(n, stream=i % 2 == 0, messages=messages)
        for i, (n, messages) in enumerate(zip(chat_lengths, conversations, strict=True))
    ]
    # Eight requests in a batch of four, so half of them wait and start as others end.
    assert len(requests) == 8
    results = send_all(server.base_url, requests)
    assert_each_gets_its_single_run(model, tiny_tokenizer, requests, results, TINY_ATOL)


@pytest.fixture(scope="module")
def smollm2_server(smollm2_checkpoint: Path) -> Iterator[tuple[str, LlamaModel, Tokenizer]]:
    model = load_model(smollm2_checkpoint, device=torch.device("cpu"), dtype=torch.float32)
    config = EngineConfig(max_num_seqs=4, max_model_len=256)
    engine = LLMEngine(model, Tokenizer.from_dir(smollm2_checkpoint), config)
    app = build_app(AsyncEngine(engine), Tokenizer.from_dir(smollm2_checkpoint), MODEL_NAME)
    with serving(app) as base_url:
        yield base_url, model, Tokenizer.from_dir(smollm2_checkpoint)


@pytest.mark.slow
def test_smollm2_requests_through_the_server_get_what_they_get_alone(
    smollm2_server: tuple[str, LlamaModel, Tokenizer],
) -> None:
    base_url, model, tokenizer = smollm2_server
    requests = [
        Request(64, stream=i % 2 == 0, prompt=prompt)
        if isinstance(prompt, str)
        else Request(64, stream=i % 2 == 0, messages=prompt)
        for i, prompt in enumerate(SMOLLM2_PROMPTS.values())
    ]
    results = send_all(base_url, requests)
    assert_each_gets_its_single_run(model, tokenizer, requests, results, SMOLLM2_ATOL)
