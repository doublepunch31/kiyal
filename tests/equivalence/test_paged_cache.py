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
from tests.equivalence.prompts import SMOLLM2_PROMPTS, TINY_TEXTS, encode_prompt, random_prompts

# The tolerances of the earlier rungs. Largest differences from the prompts decoded alone,
# measured on an Apple M4: 1.9e-6 on the tiny model at every block size and 2.4e-6 with
# preemption; 1.3e-4 on SmolLM2 with preemption.
TINY_ATOL = 2e-5
SMOLLM2_ATOL = 1e-3
TINY_PROMPT_LENGTHS = [1, 7, 40, 19, 25, 3]
TINY_OUTPUT_LENGTHS = [30, 60, 12, 45, 20, 50]
# 32 blocks of 4 hold one sequence of the tiny model's whole context of 128 tokens, less
# than any four of these requests grow to. The token budget is below the longest requests,
# so one that is preempted late restarts alone.
TINY_PREEMPT_CONFIG = EngineConfig(
    max_num_seqs=4, max_num_batched_tokens=48, block_size=4, num_kv_blocks=32
)
SMOLLM2_OUTPUT_LENGTHS = [64, 24, 48, 40]
# 8 blocks of 16 hold one sequence of 128 tokens, less than any three of these requests
# grow to.
SMOLLM2_PREEMPT_CONFIG = EngineConfig(
    max_num_seqs=3, max_num_batched_tokens=64, max_model_len=128, num_kv_blocks=8
)


# Blocks of 1 and 3 split each sequence over many blocks, with 3 dividing few of the
# lengths. 16 is the default. 128, the tiny model's context, gives each sequence a single
# block as long as the longest sequence, which is the contiguous pool: decode steps read it
# in place instead of gathering blocks.
@pytest.mark.parametrize("block_size", [1, 3, 16, 128])
def test_paged_logits_match_single_prompt_logits_on_the_tiny_model(
    make_tiny_checkpoint: Callable[..., Path], tiny_tokenizer: Tokenizer, block_size: int
) -> None:
    model = tiny_model(make_tiny_checkpoint, tiny_tokenizer)
    prompts = random_prompts(model.config.vocab_size, TINY_PROMPT_LENGTHS)
    config = EngineConfig(max_num_seqs=3, max_num_batched_tokens=48, block_size=block_size)
    setup = Setup(model, tiny_tokenizer, config, TINY_ATOL)
    preemptions = assert_continuous_logits_match(setup, prompts, TINY_OUTPUT_LENGTHS)
    assert preemptions == [0, 0, 0]


def test_preempted_logits_match_uninterrupted_logits_on_the_tiny_model(
    make_tiny_checkpoint: Callable[..., Path], tiny_tokenizer: Tokenizer
) -> None:
    model = tiny_model(make_tiny_checkpoint, tiny_tokenizer)
    prompts = random_prompts(model.config.vocab_size, TINY_PROMPT_LENGTHS)
    setup = Setup(model, tiny_tokenizer, TINY_PREEMPT_CONFIG, TINY_ATOL)
    preemptions = assert_continuous_logits_match(setup, prompts, TINY_OUTPUT_LENGTHS)
    assert all(n > 0 for n in preemptions)


def test_preempted_generation_matches_uninterrupted_generation_on_the_tiny_model(
    tiny_model_dir: Path, tiny_tokenizer: Tokenizer
) -> None:
    model = load_model(tiny_model_dir, device=CPU, dtype=torch.float32)
    prompts = [tiny_tokenizer.encode(text) for text in TINY_TEXTS.values()]
    params = [SamplingParams(max_tokens=n, temperature=0) for n in (64, 40, 64, 20)]
    # 20 blocks of 4 hold one sequence of 80 tokens, less than two of these requests use.
    config = EngineConfig(
        max_num_seqs=2, max_num_batched_tokens=32, max_model_len=80, block_size=4, num_kv_blocks=20
    )
    preemptions = assert_same_generation(
        Setup(model, tiny_tokenizer, config, TINY_ATOL), prompts, params
    )
    assert all(n > 0 for n in preemptions)


@pytest.mark.slow
def test_preempted_logits_match_uninterrupted_logits_on_smollm2(
    smollm2: LlamaModel, smollm2_kiyal_tokenizer: Tokenizer
) -> None:
    prompts = [encode_prompt(smollm2_kiyal_tokenizer, p) for p in SMOLLM2_PROMPTS.values()]
    setup = Setup(smollm2, smollm2_kiyal_tokenizer, SMOLLM2_PREEMPT_CONFIG, SMOLLM2_ATOL)
    preemptions = assert_continuous_logits_match(setup, prompts, SMOLLM2_OUTPUT_LENGTHS)
    assert all(n > 0 for n in preemptions)


@pytest.mark.slow
def test_preempted_generation_matches_uninterrupted_generation_on_smollm2(
    smollm2: LlamaModel, smollm2_kiyal_tokenizer: Tokenizer
) -> None:
    prompts = [encode_prompt(smollm2_kiyal_tokenizer, p) for p in SMOLLM2_PROMPTS.values()]
    params = [SamplingParams(max_tokens=n, temperature=0) for n in SMOLLM2_OUTPUT_LENGTHS]
    setup = Setup(smollm2, smollm2_kiyal_tokenizer, SMOLLM2_PREEMPT_CONFIG, SMOLLM2_ATOL)
    preemptions = assert_same_generation(setup, prompts, params)
    assert all(n > 0 for n in preemptions)
