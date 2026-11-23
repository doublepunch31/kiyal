from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from kiyal.config import EngineConfig
from kiyal.engine.sequence import SamplingParams
from kiyal.model.llama import LlamaModel
from kiyal.model.loader import load_model
from kiyal.tokenizer import Tokenizer
from tests.equivalence.continuous import (
    CPU,
    Setup,
    assert_continuous_logits_match,
    assert_same_generation,
    tiny_model,
)
from tests.equivalence.prompts import TINY_TEXTS, random_prompts

# The tolerances of the earlier rungs. Largest differences from the prompts decoded alone,
# measured on an Apple M4: 2.3e-6 on the tiny model at every block size, with and without
# preemption; 1.0e-4 on SmolLM2. The reference is the prompt decoded alone rather than the
# engine without the cache, as in the paged rung, which is the stronger check.
TINY_ATOL = 2e-5
SMOLLM2_ATOL = 1e-3
TINY_OUTPUT_LENGTHS = [20, 30, 12, 25, 40, 18, 22]
TINY_SYSTEM = "the dog sat on the log and the cat sat on the mat"
SMOLLM2_SYSTEM = (
    "You are a careful assistant for a physics class. Answer in plain language, give numbers "
    "with their units, and say so when a question has no single answer. Keep each answer "
    "under five sentences."
)
SMOLLM2_QUESTIONS = [
    "What is the boiling point of water at sea level?",
    "Why is the sky blue?",
    "How fast does sound travel in air?",
    "What is the capital of France?",
]
SMOLLM2_OUTPUT_LENGTHS = [64, 24, 48, 40]
SMOLLM2_CONFIG = EngineConfig(max_num_seqs=2, max_num_batched_tokens=128)


def shared_prefix_prompts(vocab_size: int) -> list[list[int]]:
    """Prompts that share their first tokens in the ways prefix caching has to handle.

    Most start with the same 29 tokens, which are not a whole number of blocks at block
    sizes above 1. One prompt is exactly those 29 tokens, so every full block before its last
    token is cached; one is their first 16, a whole number of blocks at sizes 1, 4 and 16;
    one extends another prompt past the common start; two are equal; one shares nothing.
    """
    system, first, second, third, other = random_prompts(vocab_size, [29, 6, 11, 9, 17])
    return [
        system + first,
        system + second,
        system,
        system + first + third,
        other,
        system[:16],
        system + second,
    ]


def chat_prompts(tokenizer: Tokenizer, system: str, questions: list[str]) -> list[list[int]]:
    """Chat prompts with the same system message, which the chat template puts first."""
    return [
        tokenizer.encode_chat(
            [{"role": "system", "content": system}, {"role": "user", "content": question}]
        )
        for question in questions
    ]


# Blocks of 1 share every token before the last, blocks of 16 at most one block of the
# common start.
@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_cached_prefix_logits_match_single_prompt_logits_on_the_tiny_model(
    make_tiny_checkpoint: Callable[..., Path], tiny_tokenizer: Tokenizer, block_size: int
) -> None:
    model = tiny_model(make_tiny_checkpoint, tiny_tokenizer)
    prompts = shared_prefix_prompts(model.config.vocab_size)
    config = EngineConfig(max_num_seqs=3, max_num_batched_tokens=48, block_size=block_size)
    setup = Setup(model, tiny_tokenizer, config, TINY_ATOL)
    counts = assert_continuous_logits_match(setup, prompts, TINY_OUTPUT_LENGTHS)
    assert all(c.prefix_hit_tokens > 0 for c in counts)


def test_cached_prefix_logits_match_single_prompt_logits_when_blocks_run_out(
    make_tiny_checkpoint: Callable[..., Path], tiny_tokenizer: Tokenizer
) -> None:
    model = tiny_model(make_tiny_checkpoint, tiny_tokenizer)
    prompts = shared_prefix_prompts(model.config.vocab_size)
    # 32 blocks of 4 hold one sequence of the whole context, less than four of these
    # requests grow to, so requests are preempted, cached blocks are evicted, and preempted
    # requests restart after whatever is left of their own blocks.
    config = EngineConfig(max_num_seqs=4, max_num_batched_tokens=48, block_size=4, num_kv_blocks=32)
    setup = Setup(model, tiny_tokenizer, config, TINY_ATOL)
    counts = assert_continuous_logits_match(setup, prompts, TINY_OUTPUT_LENGTHS)
    assert all(c.preemptions > 0 and c.prefix_hit_tokens > 0 for c in counts)


def test_cached_prefix_generation_matches_single_generation_on_the_tiny_model(
    tiny_model_dir: Path, tiny_tokenizer: Tokenizer
) -> None:
    model = load_model(tiny_model_dir, device=CPU, dtype=torch.float32)
    prompts = chat_prompts(tiny_tokenizer, TINY_SYSTEM, list(TINY_TEXTS.values()))
    params = [SamplingParams(max_tokens=n, temperature=0) for n in (64, 40, 64, 20)]
    config = EngineConfig(max_num_seqs=2, max_num_batched_tokens=64, block_size=4)
    counts = assert_same_generation(
        Setup(model, tiny_tokenizer, config, TINY_ATOL), prompts, params
    )
    assert all(c.prefix_hit_tokens > 0 for c in counts)


@pytest.mark.slow
def test_cached_prefix_logits_match_single_prompt_logits_on_smollm2(
    smollm2: LlamaModel, smollm2_kiyal_tokenizer: Tokenizer
) -> None:
    prompts = chat_prompts(smollm2_kiyal_tokenizer, SMOLLM2_SYSTEM, SMOLLM2_QUESTIONS)
    setup = Setup(smollm2, smollm2_kiyal_tokenizer, SMOLLM2_CONFIG, SMOLLM2_ATOL)
    counts = assert_continuous_logits_match(setup, prompts, SMOLLM2_OUTPUT_LENGTHS)
    assert all(c.prefix_hit_tokens > 0 for c in counts)


@pytest.mark.slow
def test_cached_prefix_generation_matches_single_generation_on_smollm2(
    smollm2: LlamaModel, smollm2_kiyal_tokenizer: Tokenizer
) -> None:
    prompts = chat_prompts(smollm2_kiyal_tokenizer, SMOLLM2_SYSTEM, SMOLLM2_QUESTIONS)
    params = [SamplingParams(max_tokens=n, temperature=0) for n in SMOLLM2_OUTPUT_LENGTHS]
    setup = Setup(smollm2, smollm2_kiyal_tokenizer, SMOLLM2_CONFIG, SMOLLM2_ATOL)
    counts = assert_same_generation(setup, prompts, params)
    assert all(c.prefix_hit_tokens > 0 for c in counts)
