import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import Tensor

from kiyal.config import EngineConfig
from kiyal.engine.generator import TextGenerator
from kiyal.engine.llm_engine import LLMEngine
from kiyal.engine.model_runner import ModelRunner
from kiyal.engine.scheduler import Scheduler
from kiyal.engine.sequence import RequestOutput, SamplingParams, SequenceFactory
from kiyal.model.llama import LlamaModel
from kiyal.model.loader import load_model
from kiyal.tokenizer import Tokenizer
from tests.equivalence.greedy import assert_greedy_equivalent, near_tie_steps, single_greedy
from tests.equivalence.prompts import (
    SMOLLM2_PROMPTS,
    TINY_TEXTS,
    TINY_VARIANTS,
    encode_prompt,
    random_prompts,
)

CPU = torch.device("cpu")
# Largest differences between continuously batched and single-prompt logits measured on an
# Apple M4: 1.9e-6 on the tiny model and 1.4e-4 on SmolLM2, about the same as for static
# batching. The tolerances are those of the earlier rungs, which leaves about 7x room.
TINY_ATOL = 2e-5
SMOLLM2_ATOL = 1e-3
# More prompts than rows, so rows are reused, and a token budget smaller than all prompts
# together, so prompts that arrive together start in different steps.
TINY_CONFIG = EngineConfig(max_num_seqs=3, max_num_batched_tokens=48)
TINY_PROMPT_LENGTHS = [1, 7, 40, 19, 25, 3]
TINY_OUTPUT_LENGTHS = [30, 60, 12, 45, 20, 50]
SMOLLM2_CONFIG = EngineConfig(max_num_seqs=2, max_num_batched_tokens=64)
SMOLLM2_OUTPUT_LENGTHS = [64, 24, 48, 40]
# (seed, spread): requests arrive at random steps in [0, spread). With a spread of 1 they all
# arrive at once, so the token budget splits their prompts across steps.
ARRIVAL_SCHEDULES = [(0, 24), (1, 24), (2, 1)]


@dataclass(frozen=True)
class Setup:
    """The model under test, how the engine batches it, and the tolerance for its logits."""

    model: LlamaModel
    tokenizer: Tokenizer
    config: EngineConfig
    atol: float


def random_arrivals(num_requests: int, seed: int, spread: int) -> list[tuple[int, int]]:
    """Returns (step, request index) pairs by step, in random order within a step."""
    generator = random.Random(seed)
    order = list(range(num_requests))
    generator.shuffle(order)
    # The sort is stable, so requests that arrive at the same step keep the shuffled order.
    return sorted(((generator.randrange(spread), index) for index in order), key=lambda a: a[0])


@torch.inference_mode()
def continuous_logits(
    setup: Setup,
    prompts: Sequence[list[int]],
    forced_tokens: Sequence[list[int]],
    arrivals: Sequence[tuple[int, int]],
) -> list[Tensor]:
    """Runs the prompts through the scheduler and the model runner as they arrive.

    Each request is fed the tokens its single run chose (teacher forcing) instead of
    sampling, so it computes logits for the same sequence as its single run at every step,
    whatever the batch around it.

    Returns:
        The [steps, V] logits of each prompt.
    """
    model, config = setup.model, setup.config
    max_len = model.config.max_position_embeddings
    factory = SequenceFactory(model.config, setup.tokenizer, max_len)
    scheduler = Scheduler(config)
    runner = ModelRunner(model, config.max_num_seqs, max_len)
    sequences = [
        factory.create(str(index), prompt, SamplingParams(max_tokens=len(tokens), ignore_eos=True))
        for index, (prompt, tokens) in enumerate(zip(prompts, forced_tokens, strict=True))
    ]
    logits: list[list[Tensor]] = [[] for _ in prompts]
    step = 0
    while step <= arrivals[-1][0] or scheduler.has_unfinished():
        for arrival, index in arrivals:
            if arrival == step:
                scheduler.add(sequences[index])
        scheduled = scheduler.schedule()
        if not scheduled.is_empty:
            step_logits = runner.execute(scheduled.sequences)  # [B, V]
            for seq, row_logits in zip(scheduled.sequences, step_logits, strict=True):
                index = int(seq.request_id)
                logits[index].append(row_logits)
                seq.num_computed_tokens = seq.num_tokens
                seq.append(forced_tokens[index][len(seq.output_token_ids)])
        step += 1
    return [torch.stack(rows) for rows in logits]


def assert_continuous_logits_match(
    setup: Setup, prompts: Sequence[list[int]], num_steps: Sequence[int]
) -> None:
    expected = [
        single_greedy(setup.model, prompt, n) for prompt, n in zip(prompts, num_steps, strict=True)
    ]
    forced = [tokens.tolist() for _, tokens in expected]
    for seed, spread in ARRIVAL_SCHEDULES:
        arrivals = random_arrivals(len(prompts), seed, spread)
        actual = continuous_logits(setup, prompts, forced, arrivals)
        for (logits, _), row_logits in zip(expected, actual, strict=True):
            assert_greedy_equivalent(row_logits, logits, setup.atol)


def engine_outputs(
    engine: LLMEngine,
    prompts: Sequence[list[int]],
    params: Sequence[SamplingParams],
    arrivals: Sequence[tuple[int, int]],
) -> list[list[RequestOutput]]:
    """Adds each request at its arrival step and returns every request's outputs."""
    request_index: dict[str, int] = {}
    outputs: list[list[RequestOutput]] = [[] for _ in prompts]
    step = 0
    while step <= arrivals[-1][0] or engine.has_unfinished():
        for arrival, index in arrivals:
            if arrival == step:
                request_index[engine.add_request(prompts[index], params[index])] = index
        for output in engine.step():
            outputs[request_index[output.request_id]].append(output)
        step += 1
    return outputs


def assert_same_generation(
    setup: Setup, prompts: Sequence[list[int]], params: Sequence[SamplingParams]
) -> None:
    """Checks that each request gives the tokens, text and finish reason of its prompt
    generated alone, for requests arriving in several random orders."""
    model, tokenizer = setup.model, setup.tokenizer
    generator = TextGenerator(model, tokenizer)
    expected = [list(generator.generate(p, s)) for p, s in zip(prompts, params, strict=True)]
    # After a near tie a request may pick another token than its single run and then
    # continue another sequence, so only the steps before the first one must match.
    comparable: list[int] = []
    for prompt, steps in zip(prompts, expected, strict=True):
        logits, _ = single_greedy(model, prompt, len(steps))
        near_ties = near_tie_steps(logits, setup.atol).nonzero().flatten().tolist()
        comparable.append(near_ties[0] if near_ties else len(steps))

    for seed, spread in ARRIVAL_SCHEDULES:
        engine = LLMEngine(model, tokenizer, setup.config)
        arrivals = random_arrivals(len(prompts), seed, spread)
        actual = engine_outputs(engine, prompts, params, arrivals)
        for outputs, steps, num_comparable in zip(actual, expected, comparable, strict=True):
            got = [(o.token_id, o.text, o.finish_reason) for o in outputs]
            want = [(s.token_id, s.text, s.finish_reason) for s in steps]
            assert got[:num_comparable] == want[:num_comparable]
            if num_comparable == len(steps):
                assert len(got) == len(want)


def tiny_model(
    make_checkpoint: Callable[..., Path], tokenizer: Tokenizer, variant: str
) -> LlamaModel:
    # Sized for the tiny tokenizer, which the engine needs for its sequences.
    checkpoint = make_checkpoint(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **TINY_VARIANTS[variant],
    )
    return load_model(checkpoint, device=CPU, dtype=torch.float32)


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
    config = EngineConfig(max_num_seqs=2, max_num_batched_tokens=32)
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
