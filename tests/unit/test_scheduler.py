import dataclasses

import pytest

from kiyal.cache.block_manager import BlockManager
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
BLOCK_SIZE = 4


@pytest.fixture
def factory(tiny_config: ModelConfig, tiny_tokenizer: Tokenizer) -> SequenceFactory:
    config = dataclasses.replace(tiny_config, vocab_size=tiny_tokenizer.vocab_size)
    return SequenceFactory(config, tiny_tokenizer, config.max_position_embeddings)


def make_scheduler(
    max_num_seqs: int = 4,
    max_num_batched_tokens: int = 64,
    num_blocks: int = 64,
    watermark: int = 0,
    prefix_caching: bool = False,
) -> tuple[Scheduler, BlockManager]:
    """Prefix caching is off by default: most tests use prompts of one repeated token, whose
    blocks would all be shared."""
    config = EngineConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=prefix_caching,
    )
    blocks = BlockManager(num_blocks, BLOCK_SIZE, watermark)
    return Scheduler(config, blocks), blocks


def add(
    scheduler: Scheduler,
    factory: SequenceFactory,
    name: str,
    prompt: int | list[int],
    max_tokens: int = 8,
) -> Sequence:
    """Queues a prompt of the given tokens, or of `prompt` copies of TOKEN."""
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    prompt_ids = [TOKEN] * prompt if isinstance(prompt, int) else prompt
    seq = factory.create(name, prompt_ids, params)
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
    scheduler, _ = make_scheduler()
    for name in "cab":
        add(scheduler, factory, name, 3)
    output = scheduler.schedule()
    assert names(output.prefills) == ["c", "a", "b"]
    assert output.decodes == []
    assert all(seq.status is SequenceStatus.RUNNING for seq in output.prefills)
    assert all(seq.first_scheduled_time is not None for seq in output.prefills)
    assert scheduler.num_running == 3
    assert scheduler.num_waiting == 0


def test_a_starting_sequence_gets_blocks_for_its_prompt(factory: SequenceFactory) -> None:
    scheduler, blocks = make_scheduler()
    for name, prompt_len in zip("abc", [3, 9, 4], strict=True):
        add(scheduler, factory, name, prompt_len)
    assert [seq.block_table for seq in scheduler.schedule().prefills] == [[0], [1, 2, 3], [4]]
    assert blocks.num_free_blocks == 64 - 5


def test_a_step_starts_prompts_within_the_token_budget(factory: SequenceFactory) -> None:
    scheduler, _ = make_scheduler(max_num_batched_tokens=10)
    for name in "abc":
        add(scheduler, factory, name, 4)
    assert names(scheduler.schedule().prefills) == ["a", "b"]
    assert names(scheduler.schedule().prefills) == ["c"]


def test_a_prompt_that_does_not_fit_blocks_the_ones_behind_it(factory: SequenceFactory) -> None:
    scheduler, _ = make_scheduler(max_num_batched_tokens=10)
    add(scheduler, factory, "a", 4)
    add(scheduler, factory, "b", 8)
    # "c" would fit next to "a", but starting it first could pass over "b" forever.
    add(scheduler, factory, "c", 2)
    assert names(scheduler.schedule().prefills) == ["a"]
    assert names(scheduler.schedule().prefills) == ["b", "c"]


def test_a_prompt_as_long_as_the_budget_starts_alone(factory: SequenceFactory) -> None:
    scheduler, _ = make_scheduler(max_num_batched_tokens=10)
    add(scheduler, factory, "a", 10)
    add(scheduler, factory, "b", 1)
    assert names(scheduler.schedule().prefills) == ["a"]
    assert names(scheduler.schedule().prefills) == ["b"]


def test_the_batch_holds_at_most_max_num_seqs(factory: SequenceFactory) -> None:
    scheduler, _ = make_scheduler(max_num_seqs=2)
    for name in "abc":
        add(scheduler, factory, name, 3)
    output = scheduler.schedule()
    assert names(output.prefills) == ["a", "b"]
    run(output)
    # "c" waits for a place, and the running sequences decode.
    output = scheduler.schedule()
    assert names(output.decodes) == ["a", "b"]
    assert output.prefills == []
    assert scheduler.num_waiting == 1


def test_a_prompt_waits_for_free_blocks(factory: SequenceFactory) -> None:
    scheduler, _ = make_scheduler(num_blocks=4)
    add(scheduler, factory, "a", 9, max_tokens=3)
    add(scheduler, factory, "b", 5)
    # "a" takes 3 of the 4 blocks and "b" needs 2.
    output = scheduler.schedule()
    assert names(output.prefills) == ["a"]
    run(output)
    assert names(scheduler.schedule().decodes) == ["a"]
    assert scheduler.num_waiting == 1


def test_a_prompt_must_leave_the_watermark_free(factory: SequenceFactory) -> None:
    scheduler, blocks = make_scheduler(num_blocks=4, watermark=1)
    add(scheduler, factory, "a", 9)
    add(scheduler, factory, "b", 1)
    # "b" would take the last free block.
    assert names(scheduler.schedule().prefills) == ["a"]
    assert blocks.num_free_blocks == 1
    assert scheduler.num_waiting == 1


def test_prefills_go_before_decodes(factory: SequenceFactory) -> None:
    scheduler, _ = make_scheduler()
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
    scheduler, _ = make_scheduler()
    for name in "ab":
        add(scheduler, factory, name, 3)
    run(scheduler.schedule())
    for _ in range(3):
        output = scheduler.schedule()
        assert names(output.sequences) == ["a", "b"]
        run(output)
    assert [len(seq.output_token_ids) for seq in output.decodes] == [4, 4]


def test_a_decode_step_takes_a_block_when_a_table_is_full(factory: SequenceFactory) -> None:
    scheduler, blocks = make_scheduler()
    seq = add(scheduler, factory, "a", 3)
    run(scheduler.schedule())
    # The first output token takes the last slot of the prompt's block.
    run(scheduler.schedule())
    assert seq.block_table == [0]
    # The second one starts a new block.
    run(scheduler.schedule())
    assert seq.block_table == [0, 1]
    assert blocks.num_free_blocks == 62


def test_the_newest_sequence_is_preempted_when_blocks_run_out(factory: SequenceFactory) -> None:
    scheduler, blocks = make_scheduler(num_blocks=3)
    old = add(scheduler, factory, "old", 4)
    middle = add(scheduler, factory, "middle", 1)
    new = add(scheduler, factory, "new", 1)
    run(scheduler.schedule())
    # "old" needs a second block for its first output token, and none is free. "new"
    # gives its block up, not "middle", which arrived before it.
    output = scheduler.schedule()
    assert names(output.decodes) == ["old", "middle"]
    assert old.block_table == [0, 2]
    assert middle.block_table == [1]
    assert new.status is SequenceStatus.PREEMPTED
    assert (new.block_table, new.num_computed_tokens) == ([], 0)
    assert new.output_token_ids == [TOKEN]
    assert scheduler.num_preemptions == 1
    assert (scheduler.num_running, scheduler.num_waiting) == (2, 1)
    assert blocks.num_free_blocks == 0


def test_a_sequence_that_cannot_grow_is_preempted_itself(factory: SequenceFactory) -> None:
    scheduler, _ = make_scheduler(num_blocks=2)
    old = add(scheduler, factory, "old", 3)
    new = add(scheduler, factory, "new", 4)
    run(scheduler.schedule())
    # "old" still fits in its block, so only "new", the newest, gives up its block.
    output = scheduler.schedule()
    assert names(output.decodes) == ["old"]
    assert old.block_table == [0]
    assert new.status is SequenceStatus.PREEMPTED
    assert scheduler.num_preemptions == 1


def test_a_preempted_sequence_restarts_first_and_reruns_its_output(
    factory: SequenceFactory,
) -> None:
    scheduler, _ = make_scheduler(max_num_seqs=2, num_blocks=3)
    add(scheduler, factory, "old", 4, max_tokens=2)
    new = add(scheduler, factory, "new", 4)
    # "late" waits for a place in the batch.
    late = add(scheduler, factory, "late", 1)
    run(scheduler.schedule())
    first_scheduled = new.first_scheduled_time
    # "old" takes the last free block, and "new" gives up its own.
    run(scheduler.schedule())
    assert new.status is SequenceStatus.PREEMPTED
    assert scheduler.num_waiting == 2
    # "old" has ended. "new" goes back ahead of "late", which waited before "new" was
    # preempted, and reruns its prompt and its output token.
    output = scheduler.schedule()
    assert names(output.prefills) == ["new", "late"]
    assert new.status is SequenceStatus.RUNNING
    assert new.uncomputed_token_ids() == [TOKEN] * 5
    assert len(new.block_table) == 2
    assert new.first_scheduled_time == first_scheduled
    assert len(late.block_table) == 1


def test_a_preempted_sequence_longer_than_the_budget_restarts_alone(
    factory: SequenceFactory,
) -> None:
    scheduler, _ = make_scheduler(max_num_batched_tokens=8, num_blocks=5)
    old = add(scheduler, factory, "old", 4, max_tokens=12)
    new = add(scheduler, factory, "new", 4, max_tokens=12)
    while new.status is not SequenceStatus.PREEMPTED:
        run(scheduler.schedule())
    # Both had two blocks, and "old" took the last free one for its ninth token.
    assert new.num_tokens == 9
    add(scheduler, factory, "late", 1)
    while not old.is_finished:
        output = scheduler.schedule()
        # "late" would fit, but waits behind "new".
        assert names(output.sequences) == ["old"]
        run(output)
    output = scheduler.schedule()
    assert names(output.prefills) == ["new"]
    run(output)
    assert names(scheduler.schedule().prefills) == ["late"]


def test_a_finished_sequence_frees_its_blocks_for_a_waiting_one(
    factory: SequenceFactory,
) -> None:
    scheduler, blocks = make_scheduler(max_num_seqs=2, num_blocks=3)
    short = add(scheduler, factory, "short", 3, max_tokens=2)
    add(scheduler, factory, "long", 3, max_tokens=8)
    waiting = add(scheduler, factory, "waiting", 3)
    run(scheduler.schedule())
    run(scheduler.schedule())
    assert short.is_finished
    assert short.block_table == [0]
    # The next step drops the finished sequence and starts the waiting one in its block,
    # while "long" is still decoding.
    output = scheduler.schedule()
    assert names(output.prefills) == ["waiting"]
    assert short.block_table == []
    assert waiting.block_table == [0]
    assert scheduler.num_running == 2
    assert blocks.num_free_blocks == 1


def test_the_scheduler_is_done_when_every_output_has_ended(factory: SequenceFactory) -> None:
    scheduler, blocks = make_scheduler()
    assert not scheduler.has_unfinished()
    assert scheduler.schedule().is_empty
    add(scheduler, factory, "a", 3, max_tokens=1)
    assert scheduler.has_unfinished()
    run(scheduler.schedule())
    # Finished, but still holding its blocks until the next step.
    assert not scheduler.has_unfinished()
    assert scheduler.schedule().is_empty
    assert scheduler.num_running == 0
    assert blocks.num_free_blocks == blocks.num_blocks


def test_aborting_a_waiting_request_removes_it(factory: SequenceFactory) -> None:
    scheduler, _ = make_scheduler(max_num_seqs=1)
    add(scheduler, factory, "a", 3)
    waiting = add(scheduler, factory, "b", 3)
    assert scheduler.abort("b") is waiting
    assert waiting.finish_reason is FinishReason.ABORT
    assert scheduler.num_waiting == 1
    assert names(scheduler.schedule().prefills) == ["a"]


def test_aborting_a_running_request_frees_its_blocks_at_once(factory: SequenceFactory) -> None:
    scheduler, blocks = make_scheduler(max_num_seqs=1)
    running = add(scheduler, factory, "a", 3)
    run(scheduler.schedule())
    add(scheduler, factory, "b", 3)
    assert scheduler.abort("a") is running
    assert running.status is SequenceStatus.FINISHED
    assert running.finish_reason is FinishReason.ABORT
    assert running.block_table == []
    assert blocks.num_free_blocks == blocks.num_blocks
    assert scheduler.num_running == 0
    assert names(scheduler.schedule().prefills) == ["b"]


def test_aborting_a_preempted_request_removes_it(factory: SequenceFactory) -> None:
    scheduler, _ = make_scheduler(num_blocks=2)
    add(scheduler, factory, "old", 4)
    new = add(scheduler, factory, "new", 1)
    run(scheduler.schedule())
    scheduler.schedule()
    assert new.status is SequenceStatus.PREEMPTED
    assert scheduler.abort("new") is new
    assert new.finish_reason is FinishReason.ABORT
    assert scheduler.num_waiting == 0


def test_aborting_an_unknown_or_finished_request_does_nothing(factory: SequenceFactory) -> None:
    scheduler, _ = make_scheduler()
    add(scheduler, factory, "a", 3, max_tokens=1)
    run(scheduler.schedule())
    assert scheduler.abort("a") is None
    assert scheduler.abort("unknown") is None


def test_a_prompt_longer_than_a_step_is_rejected(factory: SequenceFactory) -> None:
    scheduler, _ = make_scheduler(max_num_seqs=2, max_num_batched_tokens=8)
    with pytest.raises(ValueError, match="has 9 tokens, more than the 8 that one step can run"):
        add(scheduler, factory, "a", 9)
    assert not scheduler.has_unfinished()


def test_a_sequence_that_has_started_cannot_be_added(factory: SequenceFactory) -> None:
    scheduler, _ = make_scheduler()
    seq = add(scheduler, factory, "a", 3)
    scheduler.schedule()
    with pytest.raises(ValueError, match="request a has already started"):
        scheduler.add(seq)


# Ten distinct tokens: two full blocks of 4 and two tokens in a third block.
PROMPT = list(range(100, 110))


def test_a_sequence_starts_after_the_cached_prefix_of_a_running_one(
    factory: SequenceFactory,
) -> None:
    scheduler, blocks = make_scheduler(prefix_caching=True)
    first = add(scheduler, factory, "first", PROMPT)
    run(scheduler.schedule())
    # The same two full blocks, then other tokens.
    second = add(scheduler, factory, "second", [*PROMPT[:8], 7, 8, 9])
    output = scheduler.schedule()
    assert names(output.prefills) == ["second"]
    assert second.block_table[:2] == first.block_table[:2]
    assert second.block_table[2] not in first.block_table
    assert second.num_computed_tokens == 8
    assert second.uncomputed_token_ids() == [7, 8, 9]
    assert [blocks.ref_count(block) for block in second.block_table] == [2, 2, 1]
    assert (scheduler.num_prefix_query_tokens, scheduler.num_prefix_hit_tokens) == (21, 8)


def test_the_last_token_runs_even_when_every_block_before_it_is_cached(
    factory: SequenceFactory,
) -> None:
    scheduler, _ = make_scheduler(prefix_caching=True)
    first = add(scheduler, factory, "first", PROMPT[:8])
    run(scheduler.schedule())
    # Both blocks are cached, but the last token's logits are needed, so its block is
    # computed again in a block of the second sequence's own.
    second = add(scheduler, factory, "second", PROMPT[:8])
    scheduler.schedule()
    assert second.block_table[0] == first.block_table[0]
    assert second.block_table[1] != first.block_table[1]
    assert second.uncomputed_token_ids() == PROMPT[4:8]


def test_blocks_are_cached_once_their_keys_and_values_are_computed(
    factory: SequenceFactory,
) -> None:
    scheduler, _ = make_scheduler(prefix_caching=True)
    first = add(scheduler, factory, "first", PROMPT)
    # Starts in the same step as the first, so nothing is computed yet that it could share.
    second = add(scheduler, factory, "second", PROMPT)
    output = scheduler.schedule()
    assert names(output.prefills) == ["first", "second"]
    assert set(first.block_table).isdisjoint(second.block_table)
    run(output)
    # The cache holds the first sequence's blocks, not the second's copy of them.
    third = add(scheduler, factory, "third", PROMPT)
    scheduler.schedule()
    assert third.block_table[:2] == first.block_table[:2]
    assert scheduler.num_prefix_hit_tokens == 8


def test_blocks_filled_by_generated_tokens_are_cached(factory: SequenceFactory) -> None:
    scheduler, _ = make_scheduler(prefix_caching=True)
    first = add(scheduler, factory, "first", PROMPT[:3], max_tokens=10)
    # Six steps compute the keys and values of the prompt and of the first five generated
    # tokens, which fill two blocks.
    for _ in range(6):
        run(scheduler.schedule())
    assert first.num_computed_tokens == 8
    second = add(scheduler, factory, "second", [*PROMPT[:3], *[TOKEN] * 5, 1])
    scheduler.schedule()
    assert second.block_table[:2] == first.block_table[:2]
    assert second.num_computed_tokens == 8


def test_cached_tokens_do_not_count_against_the_token_budget(factory: SequenceFactory) -> None:
    scheduler, _ = make_scheduler(max_num_seqs=5, max_num_batched_tokens=12, prefix_caching=True)
    add(scheduler, factory, "first", [*PROMPT, 1, 2])
    run(scheduler.schedule())
    # 11 tokens each, of which 3 run. Without the cache, one would start per step.
    for name, suffix in zip("abcd", [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], strict=True):
        add(scheduler, factory, name, PROMPT[:8] + suffix)
    assert names(scheduler.schedule().prefills) == ["a", "b", "c", "d"]


def test_a_sequence_shares_the_blocks_of_one_that_has_finished(factory: SequenceFactory) -> None:
    scheduler, blocks = make_scheduler(prefix_caching=True)
    first = add(scheduler, factory, "first", PROMPT, max_tokens=1)
    run(scheduler.schedule())
    first_blocks = list(first.block_table)
    assert scheduler.schedule().is_empty
    assert blocks.num_free_blocks == blocks.num_blocks
    second = add(scheduler, factory, "second", [*PROMPT[:8], 7])
    scheduler.schedule()
    assert second.block_table[:2] == first_blocks[:2]
    assert second.num_computed_tokens == 8


def test_a_preempted_sequence_restarts_after_its_own_cached_blocks(
    factory: SequenceFactory,
) -> None:
    scheduler, _ = make_scheduler(num_blocks=5, prefix_caching=True)
    old = add(scheduler, factory, "old", [1, 2, 3, 4], max_tokens=6)
    new = add(scheduler, factory, "new", PROMPT[:8], max_tokens=12)
    run(scheduler.schedule())
    new_blocks = list(new.block_table)
    # "old" needs a third block for its ninth token and preempts "new", whose three full
    # blocks stay cached. "old" takes the one freed last, which holds generated tokens.
    while new.status is not SequenceStatus.PREEMPTED:
        run(scheduler.schedule())
    assert old.is_finished
    assert new.num_tokens == 13
    output = scheduler.schedule()
    assert names(output.prefills) == ["new"]
    assert new.block_table[:2] == new_blocks
    assert new.num_computed_tokens == 8
    assert new.uncomputed_token_ids() == [TOKEN] * 5
    assert scheduler.num_preemptions == 1


def test_without_prefix_caching_nothing_is_shared(factory: SequenceFactory) -> None:
    scheduler, _ = make_scheduler()
    first = add(scheduler, factory, "first", PROMPT)
    run(scheduler.schedule())
    second = add(scheduler, factory, "second", PROMPT)
    scheduler.schedule()
    assert set(first.block_table).isdisjoint(second.block_table)
    assert second.num_computed_tokens == 0
    assert (scheduler.num_prefix_query_tokens, scheduler.num_prefix_hit_tokens) == (0, 0)
    assert first.block_hashes == second.block_hashes == []
