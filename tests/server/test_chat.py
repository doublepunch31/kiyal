import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from kiyal.config import EngineConfig
from kiyal.engine.async_engine import AsyncEngine
from kiyal.engine.llm_engine import LLMEngine
from kiyal.engine.sequence import SamplingParams
from kiyal.model.llama import LlamaModel
from kiyal.server.app import build_app
from kiyal.tokenizer import Tokenizer
from tests.server.conftest import MODEL_NAME, Server
from tests.server.serving import serving

MESSAGES: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": "you are a cat"},
    {"role": "user", "content": "the cat"},
]
# The same conversation in the form the tokenizer takes.
PLAIN_MESSAGES = [
    {"role": "system", "content": "you are a cat"},
    {"role": "user", "content": "the cat"},
]
# Rejects conversations that do not start with a user message, as some models' templates do.
STRICT_TEMPLATE = (
    "{% if messages[0]['role'] != 'user' %}"
    "{{ raise_exception('the conversation must start with a user message') }}"
    "{% endif %}"
    "{% for message in messages %}{{ message['content'] }}{% endfor %}"
)


def test_a_chat_completion_answers_the_rendered_conversation(
    client: OpenAI, tiny_tokenizer: Tokenizer, reference: Any
) -> None:
    completion = client.chat.completions.create(
        model=MODEL_NAME, messages=MESSAGES, max_tokens=30, temperature=0
    )
    prompt = tiny_tokenizer.encode_chat(PLAIN_MESSAGES)
    expected = reference(prompt, SamplingParams(max_tokens=30, temperature=0))
    assert completion.id.startswith("chatcmpl-")
    assert (completion.object, completion.model) == ("chat.completion", MODEL_NAME)
    [choice] = completion.choices
    assert choice.message.role == "assistant"
    assert (choice.message.content, choice.finish_reason) == (expected.text, expected.finish_reason)
    assert completion.usage is not None
    assert (completion.usage.prompt_tokens, completion.usage.completion_tokens) == (
        len(prompt),
        expected.num_tokens,
    )


def test_a_reply_may_fill_the_rest_of_the_context_by_default(
    client: OpenAI, tiny_tokenizer: Tokenizer
) -> None:
    completion = client.chat.completions.create(
        model=MODEL_NAME, messages=MESSAGES, temperature=0, extra_body={"ignore_eos": True}
    )
    assert completion.usage is not None
    num_prompt_tokens = len(tiny_tokenizer.encode_chat(PLAIN_MESSAGES))
    assert completion.usage.completion_tokens == 2048 - num_prompt_tokens
    assert completion.choices[0].finish_reason == "length"


def test_max_completion_tokens_wins_over_max_tokens(client: OpenAI) -> None:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        max_completion_tokens=5,
        max_tokens=9,
        extra_body={"ignore_eos": True},
    )
    assert completion.usage is not None
    assert completion.usage.completion_tokens == 5


@pytest.mark.parametrize(
    "fields",
    [
        {"max_tokens": 30, "temperature": 0},
        {"max_tokens": 40, "temperature": 0, "stop": "the"},
        {"max_tokens": 30, "seed": 11, "top_p": 0.9},
    ],
)
def test_a_stream_carries_the_whole_reply(client: OpenAI, fields: dict[str, Any]) -> None:
    whole = client.chat.completions.create(model=MODEL_NAME, messages=MESSAGES, **fields)
    first, *chunks = client.chat.completions.create(
        model=MODEL_NAME, messages=MESSAGES, stream=True, **fields
    )
    assert first.choices[0].delta.role == "assistant"
    assert first.choices[0].delta.content == ""
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == (
        whole.choices[0].message.content
    )
    assert chunks[-1].choices[0].finish_reason == whole.choices[0].finish_reason
    assert all(chunk.choices[0].finish_reason is None for chunk in [first, *chunks[:-1]])
    [request_id] = {chunk.id for chunk in [first, *chunks]}
    assert request_id.startswith("chatcmpl-")
    assert all(chunk.object == "chat.completion.chunk" for chunk in [first, *chunks])


def test_a_chat_stream_can_end_with_the_usage(client: OpenAI) -> None:
    *_, last = client.chat.completions.create(
        model=MODEL_NAME,
        messages=MESSAGES,
        max_tokens=7,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"ignore_eos": True},
    )
    assert last.choices == []
    assert last.usage is not None
    assert last.usage.completion_tokens == 7


@pytest.mark.parametrize(
    ("fields", "message", "param"),
    [
        ({"messages": []}, "messages: List should have at least 1 item", "messages"),
        ({"messages": [{"role": "tool", "content": "x"}]}, "messages.0.role", "messages"),
        ({"messages": [{"role": "user"}]}, "messages.0.content: Field required", "messages"),
        ({"logprobs": True}, "logprobs are not supported", "logprobs"),
        ({"n": 3}, "only n=1 is supported", "n"),
        ({"temperature": -0.5}, "temperature must be 0 or greater", None),
    ],
)
def test_invalid_chat_requests_are_bad_requests(
    server: Server, fields: dict[str, Any], message: str, param: str | None
) -> None:
    body = {"model": MODEL_NAME, "messages": MESSAGES, **fields}
    response = httpx.post(f"{server.base_url}/v1/chat/completions", json=body)
    assert response.status_code == 400
    error = response.json()["error"]
    assert message in error["message"]
    assert (error["type"], error["param"]) == ("invalid_request_error", param)


@pytest.fixture(scope="module")
def strict_server(
    model: LlamaModel, tiny_long_model_dir: Path, tmp_path_factory: Any
) -> Iterator[str]:
    """A server for the tiny model whose chat template rejects some conversations."""
    model_dir = tmp_path_factory.mktemp("strict")
    shutil.copytree(tiny_long_model_dir, model_dir, dirs_exist_ok=True)
    (model_dir / "chat_template.jinja").write_text(STRICT_TEMPLATE)
    engine = LLMEngine(model, Tokenizer.from_dir(model_dir), EngineConfig(max_num_seqs=2))
    app = build_app(AsyncEngine(engine), Tokenizer.from_dir(model_dir), MODEL_NAME)
    with serving(app) as base_url:
        yield base_url


def test_a_conversation_the_template_rejects_is_a_bad_request(strict_server: str) -> None:
    body = {"model": MODEL_NAME, "messages": MESSAGES, "max_tokens": 3}
    response = httpx.post(f"{strict_server}/v1/chat/completions", json=body)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["message"] == (
        "the chat template failed: the conversation must start with a user message"
    )
    assert error["param"] == "messages"
    response = httpx.post(
        f"{strict_server}/v1/chat/completions", json={**body, "messages": MESSAGES[1:]}
    )
    assert response.status_code == 200


def test_a_model_without_a_chat_template_rejects_chat_requests(
    model: LlamaModel, tiny_long_model_dir: Path, tmp_path: Path
) -> None:
    model_dir = tmp_path / "no-template"
    shutil.copytree(tiny_long_model_dir, model_dir)
    (model_dir / "chat_template.jinja").unlink()
    engine = LLMEngine(model, Tokenizer.from_dir(model_dir), EngineConfig(max_num_seqs=2))
    app = build_app(AsyncEngine(engine), Tokenizer.from_dir(model_dir), MODEL_NAME)
    with serving(app) as base_url:
        body = {"model": MODEL_NAME, "messages": MESSAGES}
        response = httpx.post(f"{base_url}/v1/chat/completions", json=body)
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "the tokenizer has no chat template"
