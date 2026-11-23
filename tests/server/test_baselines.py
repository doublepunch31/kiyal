import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
import openai
import pytest
import torch
from openai import OpenAI

from benchmarks.hf_baseline import build_engine, build_runner
from kiyal.engine.generator import TextGenerator
from kiyal.engine.sequence import SamplingParams
from kiyal.model.llama import LlamaModel
from kiyal.server.app import build_app
from kiyal.tokenizer import Tokenizer
from tests.equivalence.greedy import near_tie_steps, single_greedy
from tests.equivalence.prompts import TINY_TEXTS
from tests.server.conftest import MODEL_NAME, Server
from tests.server.serving import serving, wait_for

TINY_ATOL = 2e-5


@dataclass(frozen=True)
class Baseline:
    engine_name: str
    server: Server


@pytest.fixture(scope="module", params=["hf", "static"])
def baseline(request: pytest.FixtureRequest, tiny_long_model_dir: Path) -> Iterator[Baseline]:
    """A baseline engine with batches of two, served by Kiyal's app."""
    name = request.param
    runner = build_runner(name, tiny_long_model_dir, torch.device("cpu"), torch.float32)
    engine = build_engine(runner, tiny_long_model_dir, batch_size=2, max_model_len=2048)
    app = build_app(engine, Tokenizer.from_dir(tiny_long_model_dir), MODEL_NAME)
    with serving(app) as base_url:
        yield Baseline(name, Server(base_url, engine))


def complete(base_url: str, prompt: str, max_tokens: int) -> str:
    with OpenAI(base_url=f"{base_url}/v1", api_key="unused", max_retries=0) as client:
        completion = client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0,
            extra_body={"ignore_eos": True},
        )
    return completion.choices[0].text


def test_greedy_requests_get_what_kiyal_generates_alone(
    baseline: Baseline, model: LlamaModel, tiny_tokenizer: Tokenizer
) -> None:
    prompts = list(TINY_TEXTS.values())
    lengths = [12, 30, 20, 8]
    base_url = baseline.server.base_url
    # Four requests at once: two batches of two.
    with ThreadPoolExecutor(len(prompts)) as pool:
        requests = zip(prompts, lengths, strict=True)
        texts = list(pool.map(lambda args: complete(base_url, *args), requests))
    generator = TextGenerator(model, tiny_tokenizer)
    for prompt, max_tokens, text in zip(prompts, lengths, texts, strict=True):
        token_ids = tiny_tokenizer.encode(prompt)
        logits, _ = single_greedy(model, token_ids, max_tokens)
        assert not near_tie_steps(logits, TINY_ATOL).any(), "pick prompts without near ties"
        params = SamplingParams(max_tokens=max_tokens, temperature=0, ignore_eos=True)
        assert text == "".join(step.text for step in generator.generate(token_ids, params))


def test_a_batch_runs_until_its_longest_output_ends(baseline: Baseline) -> None:
    engine = baseline.server.engine
    aborted = engine.stats().num_aborted
    url = f"{baseline.server.base_url}/v1/completions"
    body = {"model": MODEL_NAME, "prompt": "the cat", "temperature": 0, "ignore_eos": True}

    def post(max_tokens: int) -> threading.Thread:
        request = {"json": {**body, "max_tokens": max_tokens}, "timeout": 60}
        thread = threading.Thread(target=httpx.post, args=(url,), kwargs=request)
        thread.start()
        return thread

    # The worker takes whatever is waiting when a batch starts. Keeping it busy first makes
    # the next two requests wait, so that they start together.
    blocker = post(300)
    wait_for(engine, lambda stats: stats.num_running == 1)
    short = post(4)
    long = post(2000)
    wait_for(engine, lambda stats: stats.num_waiting == 2)
    blocker.join()
    wait_for(engine, lambda stats: stats.num_running == 2)
    short.join()
    # The short request has ended, but its row stays in the batch: a third request waits
    # for the long one. Its client gives up first.
    with pytest.raises(httpx.ReadTimeout):
        httpx.post(url, json={**body, "max_tokens": 4}, timeout=0.2)
    stats = wait_for(engine, lambda stats: stats.num_aborted == aborted + 1)
    assert (stats.num_running, stats.num_waiting) == (1, 0)
    long.join()
    wait_for(engine, lambda stats: stats.num_running == 0)


def test_only_the_transformers_baseline_needs_greedy_requests(baseline: Baseline) -> None:
    base_url = baseline.server.base_url
    with OpenAI(base_url=f"{base_url}/v1", api_key="unused", max_retries=0) as client:
        if baseline.engine_name == "hf":
            with pytest.raises(openai.BadRequestError, match="only decodes greedily"):
                client.completions.create(model=MODEL_NAME, prompt="the cat", seed=1)
        else:
            completion = client.completions.create(model=MODEL_NAME, prompt="the cat", seed=1)
            assert completion.choices[0].finish_reason is not None
