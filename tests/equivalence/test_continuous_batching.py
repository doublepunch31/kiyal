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
from tests.equivalence.prompts import (
    SMOLLM2_PROMPTS,
    TINY_TEXTS,
    TINY_VARIANTS,
    encode_prompt,
    random_prompts,
)

# Largest differences between continuously batched and single-prompt logits measured on an
# Apple M4: 1.9e-6 on the tiny model and 1.4e-4 on SmolLM2, about the same as for static
# batching. The tolerances are those of the earlier rungs, which leaves about 7x room.
TINY_ATOL = 2e-5
SMOLLM2_ATOL = 1e-3
# More prompts than places in the batch, so blocks are reused, and a token budget smaller
# than all prompts together, so prompts that arrive together start in different steps.
# Prefix caching is off in this rung and the paged one; its own rung turns it on.
TINY_CONFIG = EngineConfig(max_num_seqs=3, max_num_batched_tokens=48, enable_prefix_caching=False)
TINY_PROMPT_LENGTHS = [1, 7, 40, 19, 25, 3]
TINY_OUTPUT_LENGTHS = [30, 60, 12, 45, 20, 50]
SMOLLM2_CONFIG = EngineConfig(
    max_num_seqs=2, max_num_batched_tokens=64, enable_prefix_caching=False
)
SMOLLM2_OUTPUT_LENGTHS = [64, 24, 48, 40]


@pytest.mark.parametrize("variant", TINY_VARIANTS)
def test_continuous_logits_match_single_prompt_logits_on_the_tiny_model(
    make_tiny_checkpoint: Callable[..., Path], tiny_tokenizer: Tokenizer, variant: str
) -> None:
    model = tiny_model(make_tiny_checkpoint, tiny_tokenizer, variant)
    prompts = random_prompts(model.config.vocab_size, TINY_PROMPT_LENGTHS)
    setup = Setup(model, tiny_tokenizer, TINY_CONFIG, TINY_ATOL)
    assert_continuous_logits_match(setup, prompts, TINY_OUTPUT_LENGTHS)


def test_continuous_generation_matches_single_generation_on_the_tiny_model(
    tiny_model_dir: Path, tiny_tokenizer: Tokenizer
) -> None:
    model = load_model(tiny_model_dir, device=CPU, dtype=torch.float32)
    prompts = [tiny_tokenizer.encode(text) for text in TINY_TEXTS.values()]
    # Outputs of different lengths, so requests leave the batch at different steps.
    params = [SamplingParams(max_tokens=n, temperature=0) for n in (64, 40, 64, 20)]
    config = EngineConfig(max_num_seqs=2, max_num_batched_tokens=32, enable_prefix_caching=False)
    assert_same_generation(Setup(model, tiny_tokenizer, config, TINY_ATOL), prompts, params)


@pytest.mark.slow
def test_continuous_logits_match_single_prompt_logits_on_smollm2(
    smollm2: LlamaModel, smollm2_kiyal_tokenizer: Tokenizer
) -> None:
    prompts = [encode_prompt(smollm2_kiyal_tokenizer, p) for p in SMOLLM2_PROMPTS.values()]
    setup = Setup(smollm2, smollm2_kiyal_tokenizer, SMOLLM2_CONFIG, SMOLLM2_ATOL)
    assert_continuous_logits_match(setup, prompts, SMOLLM2_OUTPUT_LENGTHS)


@pytest.mark.slow
def test_continuous_generation_matches_single_generation_on_smollm2(
    smollm2: LlamaModel, smollm2_kiyal_tokenizer: Tokenizer
) -> None:
    prompts = [encode_prompt(smollm2_kiyal_tokenizer, p) for p in SMOLLM2_PROMPTS.values()]
    params = [SamplingParams(max_tokens=n, temperature=0) for n in SMOLLM2_OUTPUT_LENGTHS]
    setup = Setup(smollm2, smollm2_kiyal_tokenizer, SMOLLM2_CONFIG, SMOLLM2_ATOL)
    assert_same_generation(setup, prompts, params)
