from collections.abc import Callable, Sequence
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
# Largest differences between batched and single-prompt logits measured on an Apple M4: 2.5e-6
# on the tiny model and 1.3e-4 on SmolLM2, about the same as between cached and uncached
# decoding. The tolerances are the same as for the cache, which leaves about 8x room.
TINY_ATOL = 2e-5
SMOLLM2_ATOL = 1e-3
# Prompts of different lengths need padding; prompts of equal length run without it.
TINY_PROMPT_LENGTHS = {"padded": [1, 7, 40, 19], "unpadded": [16, 16, 16]}


@torch.inference_mode()
def single_greedy(model: LlamaModel, prompt: list[int], num_steps: int) -> tuple[Tensor, Tensor]:
    """Decodes one prompt greedily with a KV cache, the step before batching on the ladder.

    Returns:
        The [num_steps, V] logits of every step and the [num_steps] tokens chosen from them.
    """
    cache = ContiguousKVCache(model.config, 1, len(prompt) + num_steps - 1)
    positions = torch.arange(len(prompt)).unsqueeze(0)  # [1, T]
    logits = [model(torch.tensor([prompt]), positions, cache=cache)]  # prefill, [1, V]
    tokens = [logits[-1].argmax(dim=-1)]  # [1] each
    for _ in range(num_steps - 1):
        position = torch.tensor([[cache.length]])  # [1, 1]
        logits.append(model(tokens[-1].view(1, 1), position, cache=cache))  # decode
        tokens.append(logits[-1].argmax(dim=-1))
    return torch.cat(logits), torch.cat(tokens)


@torch.inference_mode()
def batched_logits(model: LlamaModel, prompts: Sequence[list[int]], tokens: Tensor) -> Tensor:
    """Computes the logits of every step for all prompts at once, in one left-padded batch.

    Each row is fed the tokens that its single run chose (teacher forcing), so every row
    computes logits for the same sequence as its single run and every step can be compared.

    Args:
        model: The model.
        prompts: Token ids of each prompt.
        tokens: [B, steps] tokens chosen by the single runs.

    Returns:
        [B, steps, V] logits.
    """
    num_steps = tokens.shape[1]
    longest = max(len(prompt) for prompt in prompts)
    padding = torch.tensor([longest - len(prompt) for prompt in prompts])  # [B]
    left_padding = padding if padding.any() else None
    input_ids = torch.tensor([[0] * (longest - len(prompt)) + prompt for prompt in prompts])
    positions = (torch.arange(longest) - padding.unsqueeze(-1)).clamp(min=0)  # [B, T]
    cache = ContiguousKVCache(model.config, len(prompts), longest + num_steps - 1)
    logits = [model(input_ids, positions, cache=cache, left_padding=left_padding)]  # [B, V]
    for step in range(num_steps - 1):
        positions = positions[:, -1:] + 1  # [B, 1]
        step_tokens = tokens[:, step : step + 1]  # [B, 1]
        logits.append(model(step_tokens, positions, cache=cache, left_padding=left_padding))
    return torch.stack(logits, dim=1)


def random_prompts(vocab_size: int, lengths: Sequence[int]) -> list[list[int]]:
    generator = torch.Generator().manual_seed(0)
    return [torch.randint(0, vocab_size, (n,), generator=generator).tolist() for n in lengths]


def assert_batched_logits_match(
    model: LlamaModel, prompts: Sequence[list[int]], num_steps: int, atol: float
) -> None:
    expected = [single_greedy(model, prompt, num_steps) for prompt in prompts]
    actual = batched_logits(model, prompts, torch.stack([tokens for _, tokens in expected]))
    for row, (logits, _) in enumerate(expected):
        assert_greedy_equivalent(actual[row], logits, atol)


def assert_same_generation(
    model: LlamaModel,
    tokenizer: Tokenizer,
    prompts: Sequence[list[int]],
    params: Sequence[SamplingParams],
    atol: float,
) -> None:
    """Checks that each row of a batch gives the tokens, text and finish reason of its prompt
    generated alone."""
    generator = TextGenerator(model, tokenizer)
    batch = list(generator.generate_batch(prompts, params))
    for row, (prompt, row_params) in enumerate(zip(prompts, params, strict=True)):
        expected = list(generator.generate(prompt, row_params))
        actual = [steps[row] for steps in batch if steps[row] is not None]
        # After a near tie the row may pick another token than its single run and then
        # continue another sequence, so only the steps before the first one must match.
        # Rows do not interact, so a near tie in one row does not end the check of another.
        logits, _ = single_greedy(model, prompt, len(expected))
        near_ties = near_tie_steps(logits, atol).nonzero().flatten().tolist()
        comparable = near_ties[0] if near_ties else len(expected)
        assert actual[:comparable] == expected[:comparable]
        if not near_ties:
            assert len(actual) == len(expected)


@pytest.mark.parametrize("lengths", TINY_PROMPT_LENGTHS.values(), ids=TINY_PROMPT_LENGTHS.keys())
@pytest.mark.parametrize("variant", TINY_VARIANTS)
def test_batched_logits_match_single_prompt_logits_on_the_tiny_model(
    make_tiny_checkpoint: Callable[..., Path], variant: str, lengths: list[int]
) -> None:
    checkpoint = make_tiny_checkpoint(**TINY_VARIANTS[variant])
    model = load_model(checkpoint, device=CPU, dtype=torch.float32)
    prompts = random_prompts(model.config.vocab_size, lengths)
    # The longest prompt decodes until its context is full.
    num_steps = model.config.max_position_embeddings - max(lengths)
    assert_batched_logits_match(model, prompts, num_steps, TINY_ATOL)


def test_batched_generation_matches_single_generation_on_the_tiny_model(
    tiny_model_dir: Path, tiny_tokenizer: Tokenizer
) -> None:
    model = load_model(tiny_model_dir, device=CPU, dtype=torch.float32)
    prompts = [tiny_tokenizer.encode(text) for text in TINY_TEXTS.values()]
    # Rows end at different steps, and the batch runs on after each of them.
    params = [SamplingParams(max_tokens=n, temperature=0) for n in (64, 40, 64, 20)]
    assert_same_generation(model, tiny_tokenizer, prompts, params, TINY_ATOL)


@pytest.mark.slow
def test_batched_logits_match_single_prompt_logits_on_smollm2(
    smollm2: LlamaModel, smollm2_kiyal_tokenizer: Tokenizer
) -> None:
    prompts = [encode_prompt(smollm2_kiyal_tokenizer, p) for p in SMOLLM2_PROMPTS.values()]
    assert_batched_logits_match(smollm2, prompts, 64, SMOLLM2_ATOL)


@pytest.mark.slow
def test_batched_generation_matches_single_generation_on_smollm2(
    smollm2: LlamaModel, smollm2_kiyal_tokenizer: Tokenizer
) -> None:
    prompts = [encode_prompt(smollm2_kiyal_tokenizer, p) for p in SMOLLM2_PROMPTS.values()]
    params = [SamplingParams(max_tokens=64, temperature=0)] * len(prompts)
    assert_same_generation(smollm2, smollm2_kiyal_tokenizer, prompts, params, SMOLLM2_ATOL)
