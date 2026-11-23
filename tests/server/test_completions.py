import re
from typing import Any

import httpx
import openai
import pytest
from openai import OpenAI

from kiyal.engine.sequence import SamplingParams
from kiyal.tokenizer import Tokenizer
from tests.server.conftest import MODEL_NAME, Server

PROMPT = "the cat sat on"


@pytest.fixture(scope="module")
def chat_prompt(tiny_tokenizer: Tokenizer) -> list[int]:
    """Token ids whose greedy continuation contains plain words, for stop strings."""
    return tiny_tokenizer.encode_chat([{"role": "user", "content": "the cat"}])


def test_a_completion_is_what_the_engine_generates(
    client: OpenAI, tiny_tokenizer: Tokenizer, reference: Any
) -> None:
    completion = client.completions.create(
        model=MODEL_NAME, prompt=PROMPT, max_tokens=20, temperature=0
    )
    prompt = tiny_tokenizer.encode(PROMPT)
    expected = reference(prompt, SamplingParams(max_tokens=20, temperature=0))
    assert completion.id.startswith("cmpl-")
    assert (completion.object, completion.model) == ("text_completion", MODEL_NAME)
    [choice] = completion.choices
    assert (choice.index, choice.text, choice.finish_reason) == (
        0,
        expected.text,
        expected.finish_reason,
    )
    assert completion.usage is not None
    assert (
        completion.usage.prompt_tokens,
        completion.usage.completion_tokens,
        completion.usage.total_tokens,
    ) == (len(prompt), expected.num_tokens, len(prompt) + expected.num_tokens)


def test_a_prompt_can_be_token_ids(client: OpenAI, tiny_tokenizer: Tokenizer) -> None:
    by_text = client.completions.create(
        model=MODEL_NAME, prompt=PROMPT, max_tokens=20, temperature=0
    )
    by_ids = client.completions.create(
        model=MODEL_NAME, prompt=tiny_tokenizer.encode(PROMPT), max_tokens=20, temperature=0
    )
    assert by_ids.choices[0].text == by_text.choices[0].text
    assert by_ids.usage == by_text.usage


def test_max_tokens_defaults_to_16(client: OpenAI) -> None:
    completion = client.completions.create(
        model=MODEL_NAME, prompt=PROMPT, temperature=0, extra_body={"ignore_eos": True}
    )
    assert completion.usage is not None
    assert completion.usage.completion_tokens == 16
    assert completion.choices[0].finish_reason == "length"


def test_the_output_ends_before_a_stop_string(
    client: OpenAI, chat_prompt: list[int], reference: Any
) -> None:
    completion = client.completions.create(
        model=MODEL_NAME, prompt=chat_prompt, max_tokens=40, temperature=0, stop=["the"]
    )
    expected = reference(chat_prompt, SamplingParams(max_tokens=40, temperature=0, stop=("the",)))
    # The greedy output without the stop string contains it, so the request stopped early.
    unstopped = reference(chat_prompt, SamplingParams(max_tokens=40, temperature=0))
    assert "the" in unstopped.text
    assert (completion.choices[0].text, completion.choices[0].finish_reason) == (
        expected.text,
        "stop",
    )
    assert "the" not in expected.text


def test_a_seed_makes_sampling_reproducible(
    client: OpenAI, tiny_tokenizer: Tokenizer, reference: Any
) -> None:
    texts = [
        client.completions.create(model=MODEL_NAME, prompt=PROMPT, max_tokens=20, seed=7)
        .choices[0]
        .text
        for _ in range(2)
    ]
    expected = reference(tiny_tokenizer.encode(PROMPT), SamplingParams(max_tokens=20, seed=7))
    assert texts == [expected.text, expected.text]


def test_engine_settings_outside_the_openai_api_apply(
    client: OpenAI, tiny_tokenizer: Tokenizer, reference: Any
) -> None:
    extra = {"top_k": 5, "min_p": 0.1, "repetition_penalty": 1.3}
    completion = client.completions.create(
        model=MODEL_NAME, prompt=PROMPT, max_tokens=20, seed=1, extra_body=extra
    )
    params = SamplingParams(max_tokens=20, seed=1, **extra)
    expected = reference(tiny_tokenizer.encode(PROMPT), params)
    assert completion.choices[0].text == expected.text


def test_the_model_list_names_the_served_model(client: OpenAI) -> None:
    [card] = client.models.list().data
    assert (card.id, card.object, card.owned_by) == (MODEL_NAME, "model", "kiyal")


def test_health_is_ok_while_the_engine_runs(server: Server) -> None:
    assert httpx.get(f"{server.base_url}/health").status_code == 200


@pytest.mark.parametrize(
    ("fields", "status", "message", "param"),
    [
        ({"model": "other"}, 404, "the model 'other' does not exist", "model"),
        ({"prompt": None}, 400, "prompt", "prompt"),
        ({"best_of": 2}, 400, "best_of: Extra inputs are not permitted", "best_of"),
        ({"max_tokens": "many"}, 400, "max_tokens: Input should be a valid integer", "max_tokens"),
        ({"n": 2}, 400, "only n=1 is supported", "n"),
        ({"echo": True}, 400, "echo is not supported", "echo"),
        ({"logprobs": 1}, 400, "logprobs are not supported", "logprobs"),
        ({"max_tokens": 0}, 400, "max_tokens must be 1 or greater", None),
        ({"temperature": -1}, 400, "temperature must be 0 or greater", None),
        ({"top_p": 0}, 400, r"top_p must be in \(0, 1\]", None),
        ({"seed": -1}, 400, "seed must be in", None),
        ({"stop": [""]}, 400, "stop must be a tuple of non-empty strings", None),
        ({"prompt": []}, 400, "the prompt has no tokens", None),
        ({"prompt": [5, 100000]}, 400, "the prompt has token ids outside", None),
        ({"prompt": [5] * 2048}, 400, "must also fit at least one output token", None),
    ],
)
def test_invalid_requests_get_openai_shaped_errors(
    server: Server, fields: dict[str, Any], status: int, message: str, param: str | None
) -> None:
    body = {"model": MODEL_NAME, "prompt": PROMPT, **fields}
    response = httpx.post(f"{server.base_url}/v1/completions", json=body)
    assert response.status_code == status
    error = response.json()["error"]
    assert set(error) == {"message", "type", "param", "code"}
    assert error["type"] == "invalid_request_error"
    assert error["param"] == param
    assert re.search(message, error["message"]), error["message"]


def test_the_openai_client_raises_its_error_classes(client: OpenAI) -> None:
    with pytest.raises(openai.NotFoundError) as not_found:
        client.completions.create(model="other", prompt=PROMPT)
    assert not_found.value.code == "model_not_found"
    with pytest.raises(openai.BadRequestError, match="temperature must be 0 or greater"):
        client.completions.create(model=MODEL_NAME, prompt=PROMPT, temperature=-1)


def test_a_body_that_is_not_json_is_a_bad_request(server: Server) -> None:
    response = httpx.post(
        f"{server.base_url}/v1/completions",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_unknown_paths_get_openai_shaped_errors(server: Server) -> None:
    response = httpx.get(f"{server.base_url}/v1/unknown")
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Not Found"
    response = httpx.get(f"{server.base_url}/v1/completions")
    assert response.status_code == 405
    assert response.headers["allow"] == "POST"
