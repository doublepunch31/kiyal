import dataclasses

import pytest

from kiyal.config import EngineConfig
from kiyal.engine.scheduler import Scheduler, SchedulerOutput
from kiyal.engine.sequence import (
    FinishReason,
    SamplingParams,
    Sequence,
    SequenceFactory,
    SequenceStatus,
)
from kiyal.model.config import ModelConfig
from kiyal.tokenizer import Tokenizer

# Any token that is not a stop token.
TOKEN = 5


@pytest.fixture
def factory(tiny_config: ModelConfig, tiny_tokenizer: Tokenizer) -> SequenceFactory:
    config = dataclasses.replace(tiny_config, vocab_size=tiny_tokenizer.vocab_size)
    return SequenceFactory(config, tiny_tokenizer, config.max_position_embeddings)


def make_scheduler(max_num_seqs: int = 4, max_num_batched_tokens: int = 64) -> Scheduler:
    return Scheduler(
        EngineConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens)
    )


def add(
    scheduler: Scheduler,
    factory: SequenceFactory,
    name: str,
    prompt_len: int,
    max_tokens: int = 8,
) -> Sequence:
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    seq = factory.create(name, [TOKEN] * prompt_len, params)
    scheduler.add(seq)
    return seq


def run(output: SchedulerOutput) -> None:
    """Stands in for the model runner and sampler: every scheduled sequence gets a token."""
    for seq in output.sequences:
        seq.num_computed_tokens = seq.num_tokens
        seq.append(TOKEN)


def names(sequences: list[Sequence]) -> list[str]:
    return [seq.request_id for seq in sequences]


def test_waiting_sequences_start_in_arrival_order(factory: SequenceFactory) -> None:
    scheduler = make_scheduler()
    for name in "cab":
        add(scheduler, factory, name, 3)
    output = scheduler.schedule()
    assert names(output.prefills) == ["c", "a", "b"]
    assert output.decodes == []
    assert all(seq.status is SequenceStatus.RUNNING for seq in output.prefills)
    assert all(seq.first_scheduled_time is not None for seq in output.prefills)
    assert scheduler.num_running == 3
    assert scheduler.num_waiting == 0


def test_running_sequences_get_distinct_rows_in_order(factory: SequenceFactory) -> None:
    scheduler = make_scheduler()
    for name in "abc":
        add(scheduler, factory, name, 3)
    assert [seq.cache_row for seq in scheduler.schedule().prefills] == [0, 1, 2]


def test_a_step_starts_prompts_within_the_token_budget(factory: SequenceFactory) -> None:
    scheduler = make_scheduler(max_num_batched_tokens=10)
    for name in "abc":
        add(scheduler, factory, name, 4)
    assert names(scheduler.schedule().prefills) == ["a", "b"]
    assert names(scheduler.schedule().prefills) == ["c"]


def test_a_prompt_that_does_not_fit_blocks_the_ones_behind_it(factory: SequenceFactory) -> None:
    scheduler = make_scheduler(max_num_batched_tokens=10)
    add(scheduler, factory, "a", 4)
    add(scheduler, factory, "b", 8)
    # "c" would fit next to "a", but starting it first could pass over "b" forever.
    add(scheduler, factory, "c", 2)
    assert names(scheduler.schedule().prefills) == ["a"]
    assert names(scheduler.schedule().prefills) == ["b", "c"]


def test_a_prompt_as_long_as_the_budget_starts_alone(factory: SequenceFactory) -> None:
    scheduler = make_scheduler(max_num_batched_tokens=10)
    add(scheduler, factory, "a", 10)
    add(scheduler, factory, "b", 1)
    assert names(scheduler.schedule().prefills) == ["a"]
    assert names(scheduler.schedule().prefills) == ["b"]


def test_the_batch_holds_at_most_max_num_seqs(factory: SequenceFactory) -> None:
    scheduler = make_scheduler(max_num_seqs=2)
    for name in "abc":
        add(scheduler, factory, name, 3)
    output = scheduler.schedule()
    assert names(output.prefills) == ["a", "b"]
    run(output)
    # "c" waits for a row, and the running sequences decode.
    output = scheduler.schedule()
    assert names(output.decodes) == ["a", "b"]
    assert output.prefills == []
    assert scheduler.num_waiting == 1


def test_prefills_go_before_decodes(factory: SequenceFactory) -> None:
    scheduler = make_scheduler()
    add(scheduler, factory, "a", 3)
    run(scheduler.schedule())
    add(scheduler, factory, "b", 3)
    # "a" pauses for one step while "b" starts, then both decode together.
    output = scheduler.schedule()
    assert names(output.prefills) == ["b"]
    assert output.decodes == []
    run(output)
    output = scheduler.schedule()
    assert names(output.decodes) == ["a", "b"]


def test_a_decode_step_runs_every_running_sequence_once(factory: SequenceFactory) -> None:
    scheduler = make_scheduler()
    for name in "ab":
        add(scheduler, factory, name, 3)
    run(scheduler.schedule())
    for _ in range(3):
        output = scheduler.schedule()
        assert names(output.sequences) == ["a", "b"]
        run(output)
    assert [len(seq.output_token_ids) for seq in output.decodes] == [4, 4]


def test_a_finished_sequence_frees_its_row_for_a_waiting_one(factory: SequenceFactory) -> None:
    scheduler = make_scheduler(max_num_seqs=2)
    short = add(scheduler, factory, "short", 3, max_tokens=2)
    add(scheduler, factory, "long", 3, max_tokens=8)
    waiting = add(scheduler, factory, "waiting", 3)
    run(scheduler.schedule())
    run(scheduler.schedule())
    assert short.is_finished
    assert short.cache_row == 0
    # The next step drops the finished sequence and starts the waiting one in its row,
    # while "long" is still decoding.
    output = scheduler.schedule()
    assert names(output.prefills) == ["waiting"]
    assert short.cache_row is None
    assert waiting.cache_row == 0
    assert scheduler.num_running == 2


def test_the_scheduler_is_done_when_every_output_has_ended(factory: SequenceFactory) -> None:
    scheduler = make_scheduler()
    assert not scheduler.has_unfinished()
    assert scheduler.schedule().is_empty
    add(scheduler, factory, "a", 3, max_tokens=1)
    assert scheduler.has_unfinished()
    run(scheduler.schedule())
    # Finished, but still holding its row until the next step.
    assert not scheduler.has_unfinished()
    assert scheduler.schedule().is_empty
    assert scheduler.num_running == 0


def test_aborting_a_waiting_request_removes_it(factory: SequenceFactory) -> None:
    scheduler = make_scheduler(max_num_seqs=1)
    add(scheduler, factory, "a", 3)
    waiting = add(scheduler, factory, "b", 3)
    assert scheduler.abort("b") is waiting
    assert waiting.finish_reason is FinishReason.ABORT
    assert scheduler.num_waiting == 1
    assert names(scheduler.schedule().prefills) == ["a"]


def test_aborting_a_running_request_frees_its_row_at_once(factory: SequenceFactory) -> None:
    scheduler = make_scheduler(max_num_seqs=1)
    running = add(scheduler, factory, "a", 3)
    run(scheduler.schedule())
    add(scheduler, factory, "b", 3)
    assert scheduler.abort("a") is running
    assert running.status is SequenceStatus.FINISHED
    assert running.finish_reason is FinishReason.ABORT
    assert running.cache_row is None
    assert scheduler.num_running == 0
    assert names(scheduler.schedule().prefills) == ["b"]


def test_aborting_an_unknown_or_finished_request_does_nothing(factory: SequenceFactory) -> None:
    scheduler = make_scheduler()
    add(scheduler, factory, "a", 3, max_tokens=1)
    run(scheduler.schedule())
    assert scheduler.abort("a") is None
    assert scheduler.abort("unknown") is None


def test_a_prompt_longer_than_a_step_is_rejected(factory: SequenceFactory) -> None:
    scheduler = make_scheduler(max_num_seqs=2, max_num_batched_tokens=8)
    with pytest.raises(ValueError, match="has 9 tokens, more than the 8 that one step can run"):
        add(scheduler, factory, "a", 9)
    assert not scheduler.has_unfinished()


def test_a_sequence_that_has_started_cannot_be_added(factory: SequenceFactory) -> None:
    scheduler = make_scheduler()
    seq = add(scheduler, factory, "a", 3)
    scheduler.schedule()
    with pytest.raises(ValueError, match="request a has already started"):
        scheduler.add(seq)
