import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from kiyal.cache.block_manager import BlockManager
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
from tests.equivalence.prompts import TINY_VARIANTS

CPU = torch.device("cpu")
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


@dataclass(frozen=True)
class RunCounts:
    """What happened in one run, so a test can check that it covered what it is meant to.

    Attributes:
        preemptions: Times a request gave up its KV blocks.
        prefix_hit_tokens: Tokens whose keys and values came from the prefix cache.
    """

    preemptions: int
    prefix_hit_tokens: int


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
) -> tuple[list[Tensor], RunCounts]:
    """Runs the prompts through the scheduler and the model runner as they arrive.

    Each request is fed the tokens its single run chose (teacher forcing) instead of
    sampling, so it computes logits for the same sequence as its single run at every step,
    whatever the batch around it. A preempted request computes the logits of its next token
    when it starts again, so it still has one row per forced token.

    Returns:
        The [steps, V] logits of each prompt, and the counts of the run.
    """
    model, config = setup.model, setup.config
    # The pool as LLMEngine sizes it.
    max_len = min(config.max_model_len, model.config.max_position_embeddings)
    num_blocks = config.num_kv_blocks
    if num_blocks is None:
        num_blocks = config.max_num_seqs * -(-max_len // config.block_size)
    watermark = int(config.watermark * num_blocks)
    factory = SequenceFactory(model.config, setup.tokenizer, max_len)
    scheduler = Scheduler(config, BlockManager(num_blocks, config.block_size, watermark))
    runner = ModelRunner(model, num_blocks, config.block_size)
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
    counts = RunCounts(scheduler.num_preemptions, scheduler.num_prefix_hit_tokens)
    return [torch.stack(rows) for rows in logits], counts


def assert_continuous_logits_match(
    setup: Setup, prompts: Sequence[list[int]], num_steps: Sequence[int]
) -> list[RunCounts]:
    """Checks every request's logits against its prompt decoded alone, for requests
    arriving in several random orders.

    Returns:
        The counts of the run in each order.
    """
    expected = [
        single_greedy(setup.model, prompt, n) for prompt, n in zip(prompts, num_steps, strict=True)
    ]
    forced = [tokens.tolist() for _, tokens in expected]
    counts: list[RunCounts] = []
    for seed, spread in ARRIVAL_SCHEDULES:
        arrivals = random_arrivals(len(prompts), seed, spread)
        actual, run_counts = continuous_logits(setup, prompts, forced, arrivals)
        for (logits, _), row_logits in zip(expected, actual, strict=True):
            assert_greedy_equivalent(row_logits, logits, setup.atol)
        counts.append(run_counts)
    return counts


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
) -> list[RunCounts]:
    """Checks that each request gives the tokens, text and finish reason of its prompt
    generated alone, for requests arriving in several random orders.

    Returns:
        The counts of the run in each order.
    """
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

    counts: list[RunCounts] = []
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
        counts.append(RunCounts(engine.num_preemptions, engine.num_prefix_hit_tokens))
    return counts


def tiny_model(
    make_checkpoint: Callable[..., Path], tokenizer: Tokenizer, variant: str = "gqa"
) -> LlamaModel:
    # Sized for the tiny tokenizer, which the engine needs for its sequences.
    checkpoint = make_checkpoint(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **TINY_VARIANTS[variant],
    )
    return load_model(checkpoint, device=CPU, dtype=torch.float32)
