from pathlib import Path

import pytest
import torch
from transformers import LlamaForCausalLM

from kiyal.engine.naive import NaiveGenerator
from kiyal.engine.sequence import FinishReason, SamplingParams
from kiyal.model.loader import load_model
from kiyal.tokenizer import Tokenizer

CPU = torch.device("cpu")
QUESTION = [{"role": "user", "content": "What is the capital of France?"}]
END_OF_TURN = 2

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tokenizer(smollm2_checkpoint: Path) -> Tokenizer:
    return Tokenizer.from_dir(smollm2_checkpoint)


@pytest.fixture(scope="module")
def generator(smollm2_checkpoint: Path, tokenizer: Tokenizer) -> NaiveGenerator:
    model = load_model(smollm2_checkpoint, device=CPU, dtype=torch.float32)
    return NaiveGenerator(model, tokenizer)


def test_smollm2_continues_a_sentence(generator: NaiveGenerator, tokenizer: Tokenizer) -> None:
    params = SamplingParams(max_tokens=16, temperature=0)
    steps = list(generator.generate(tokenizer.encode("The capital of France is"), params))
    assert len(steps) == 16
    assert steps[-1].finish_reason is FinishReason.LENGTH
    assert "".join(step.text for step in steps).startswith(" Paris")


def test_smollm2_ends_a_chat_reply_at_the_end_of_turn_token(
    generator: NaiveGenerator, tokenizer: Tokenizer
) -> None:
    params = SamplingParams(max_tokens=64, temperature=0)
    steps = list(generator.generate(tokenizer.encode_chat(QUESTION), params))
    assert len(steps) < 64
    assert steps[-1].finish_reason is FinishReason.STOP
    assert steps[-1].token_id == END_OF_TURN
    assert "Paris" in "".join(step.text for step in steps)


@torch.inference_mode()
def test_smollm2_greedy_output_matches_transformers_generate(
    generator: NaiveGenerator, tokenizer: Tokenizer, smollm2_checkpoint: Path
) -> None:
    prompt = tokenizer.encode_chat(QUESTION)
    reference = LlamaForCausalLM.from_pretrained(smollm2_checkpoint, dtype=torch.float32).eval()
    # transformers types generate()'s self as a protocol that its own model classes do not
    # satisfy, so the attribute access is reported.
    expected = reference.generate(  # pyright: ignore[reportAttributeAccessIssue]
        torch.tensor([prompt]),
        do_sample=False,
        max_new_tokens=64,
        eos_token_id=END_OF_TURN,
        pad_token_id=END_OF_TURN,
    )
    steps = generator.generate(prompt, SamplingParams(max_tokens=64, temperature=0))
    # generate() returns the prompt followed by the new tokens, ending with the EOS token.
    assert [step.token_id for step in steps] == expected[0, len(prompt) :].tolist()
