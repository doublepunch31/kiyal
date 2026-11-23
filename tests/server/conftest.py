from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from openai import OpenAI

from kiyal.config import EngineConfig
from kiyal.engine.async_engine import AsyncEngine
from kiyal.engine.generator import TextGenerator
from kiyal.engine.llm_engine import LLMEngine
from kiyal.engine.sequence import SamplingParams
from kiyal.model.llama import LlamaModel
from kiyal.model.loader import load_model
from kiyal.server.app import build_app
from kiyal.tokenizer import Tokenizer
from tests.server.serving import serving

MODEL_NAME = "tiny"


@dataclass(frozen=True)
class Server:
    base_url: str
    engine: AsyncEngine


@dataclass(frozen=True)
class Reference:
    """What the engine generates for a prompt offline, without the server."""

    text: str
    num_tokens: int
    finish_reason: str


@pytest.fixture(scope="module")
def model(tiny_long_model_dir: Path) -> LlamaModel:
    return load_model(tiny_long_model_dir, device=torch.device("cpu"), dtype=torch.float32)


@pytest.fixture(scope="module")
def server(model: LlamaModel, tiny_long_model_dir: Path) -> Iterator[Server]:
    """The app serving the tiny model with a context of 2048 tokens, one per test module."""
    engine = LLMEngine(model, Tokenizer.from_dir(tiny_long_model_dir), EngineConfig(max_num_seqs=4))
    async_engine = AsyncEngine(engine)
    app = build_app(async_engine, Tokenizer.from_dir(tiny_long_model_dir), MODEL_NAME)
    with serving(app) as base_url:
        yield Server(base_url, async_engine)


@pytest.fixture(scope="module")
def client(server: Server) -> Iterator[OpenAI]:
    with OpenAI(base_url=f"{server.base_url}/v1", api_key="unused", max_retries=0) as client:
        yield client


@pytest.fixture(scope="module")
def reference(model: LlamaModel, tiny_tokenizer: Tokenizer):
    """Returns a function that generates a prompt's output offline."""
    generator = TextGenerator(model, tiny_tokenizer)

    def generate(prompt_token_ids: list[int], params: SamplingParams) -> Reference:
        steps = list(generator.generate(prompt_token_ids, params))
        reason = steps[-1].finish_reason
        assert reason is not None
        return Reference("".join(step.text for step in steps), len(steps), reason.value)

    return generate
