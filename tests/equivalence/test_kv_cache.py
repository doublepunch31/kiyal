from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import Tensor

from kiyal.cache.contiguous import ContiguousKVCache
from kiyal.engine.generator import TextGenerator
from kiyal.engine.sequence import SamplingParams
from kiyal.model.llama import LlamaModel
from kiyal.model.loader import load_model
from kiyal.tokenizer import Tokenizer
from tests.equivalence.greedy import assert_greedy_equivalent, near_tie_steps
from tests.equivalence.prompts import SMOLLM2_PROMPTS, TINY_TEXTS, TINY_VARIANTS, encode_prompt

CPU = torch.device("cpu")
# Largest differences between cached and uncached logits measured on an Apple M4: 2.7e-6 on
# the tiny model and 1.4e-4 on SmolLM2, whose hidden states are much larger. The tolerances
# leave about 7x room for other BLAS libraries, which round differently.
TINY_ATOL = 2e-5
SMOLLM2_ATOL = 1e-3


@torch.inference_mode()
def uncached_greedy(model: LlamaModel, prompt: list[int], num_steps: int) -> tuple[Tensor, Tensor]:
    """Decodes greedily by running the model on the whole sequence at every step.

    Returns:
        The [num_steps, V] logits of every step and the [num_steps] tokens chosen from them.
    """
    token_ids = torch.tensor([prompt])  # [1, T]
    logits: list[Tensor] = []
    for _ in range(num_steps):
        positions = torch.arange(token_ids.shape[1]).unsqueeze(0)  # [1, T]
        logits.append(model(token_ids, positions))  # [1, V]
        token_ids = torch.cat([token_ids, logits[-1].argmax(dim=-1, keepdim=True)], dim=1)
    return torch.cat(logits), token_ids[0, len(prompt) :]


@torch.inference_mode()
def cached_logits(model: LlamaModel, prompt: list[int], tokens: Tensor) -> Tensor:
    """Computes the logits of every step with a KV cache: one prefill, then decode steps.

    The cached path is fed the tokens that the uncached path chose, so both compute logits
    for the same sequences and every step can be compared.

    Returns:
        [len(tokens), V] logits.
    """
    num_steps = len(tokens)
    cache = ContiguousKVCache(model.config, 1, len(prompt) + num_steps - 1)
    positions = torch.arange(len(prompt)).unsqueeze(0)  # [1, T]
    logits = [model(torch.tensor([prompt]), positions, cache=cache)]  # prefill
    for token in tokens[:-1]:
        position = torch.tensor([[cache.length]])  # [1, 1]
        logits.append(model(token.view(1, 1), position, cache=cache))  # decode
    return torch.cat(logits)


def random_prompt(vocab_size: int, num_tokens: int) -> list[int]:
    generator = torch.Generator().manual_seed(num_tokens)
    return torch.randint(0, vocab_size, (num_tokens,), generator=generator).tolist()


@pytest.mark.parametrize("prompt_len", [1, 7, 40])
@pytest.mark.parametrize("variant", TINY_VARIANTS)
def test_cached_logits_match_uncached_logits_on_the_tiny_model(
    make_tiny_checkpoint: Callable[..., Path], variant: str, prompt_len: int
) -> None:
    checkpoint = make_tiny_checkpoint(**TINY_VARIANTS[variant])
    model = load_model(checkpoint, device=CPU, dtype=torch.float32)
    prompt = random_prompt(model.config.vocab_size, prompt_len)
    # Decoding runs until the context is full, so the last cache position is used too.
    num_steps = model.config.max_position_embeddings - prompt_len
    expected, tokens = uncached_greedy(model, prompt, num_steps)
    assert_greedy_equivalent(cached_logits(model, prompt, tokens), expected, TINY_ATOL)


@pytest.mark.parametrize("text", TINY_TEXTS.values(), ids=TINY_TEXTS.keys())
def test_cached_generation_matches_uncached_generation_on_the_tiny_model(
    tiny_model_dir: Path, tiny_tokenizer: Tokenizer, text: str
) -> None:
    model = load_model(tiny_model_dir, device=CPU, dtype=torch.float32)
    assert_same_generation(model, tiny_tokenizer, tiny_tokenizer.encode(text), 64, TINY_ATOL)


def assert_same_generation(
    model: LlamaModel, tokenizer: Tokenizer, prompt: list[int], max_tokens: int, atol: float
) -> None:
    """Checks that generation gives the same tokens, text and finish reason in both modes."""
    params = SamplingParams(max_tokens=max_tokens, temperature=0)
    expected = list(TextGenerator(model, tokenizer, use_kv_cache=False).generate(prompt, params))
    actual = list(TextGenerator(model, tokenizer, use_kv_cache=True).generate(prompt, params))
    # After a near tie the two modes may pick different tokens and then continue different
    # sequences, so only the steps before the first one can be required to match.
    logits, _ = uncached_greedy(model, prompt, len(expected))
    near_ties = near_tie_steps(logits, atol).nonzero().flatten().tolist()
    comparable = near_ties[0] if near_ties else len(expected)
    assert actual[:comparable] == expected[:comparable]
    if not near_ties:
        assert len(actual) == len(expected)


@pytest.mark.slow
@pytest.mark.parametrize("name", SMOLLM2_PROMPTS)
def test_cached_logits_match_uncached_logits_on_smollm2(
    smollm2: LlamaModel, smollm2_kiyal_tokenizer: Tokenizer, name: str
) -> None:
    prompt = encode_prompt(smollm2_kiyal_tokenizer, SMOLLM2_PROMPTS[name])
    expected, tokens = uncached_greedy(smollm2, prompt, 64)
    assert_greedy_equivalent(cached_logits(smollm2, prompt, tokens), expected, SMOLLM2_ATOL)


@pytest.mark.slow
@pytest.mark.parametrize("name", SMOLLM2_PROMPTS)
def test_cached_generation_matches_uncached_generation_on_smollm2(
    smollm2: LlamaModel, smollm2_kiyal_tokenizer: Tokenizer, name: str
) -> None:
    prompt = encode_prompt(smollm2_kiyal_tokenizer, SMOLLM2_PROMPTS[name])
    assert_same_generation(smollm2, smollm2_kiyal_tokenizer, prompt, 64, SMOLLM2_ATOL)
