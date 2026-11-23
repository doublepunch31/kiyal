import pytest

from kiyal.cache.block_manager import BlockManager, NoFreeBlocksError, block_hash

BLOCK_SIZE = 4
TOKENS = [11, 12, 13, 14, 21, 22, 23, 24, 31, 32]
# Hashes of the first two blocks of TOKENS.
FIRST = block_hash(None, TOKENS[:4])
SECOND = block_hash(FIRST, TOKENS[4:8])


def test_a_new_manager_has_every_block_free() -> None:
    manager = BlockManager(num_blocks=6, block_size=BLOCK_SIZE)
    assert (manager.num_blocks, manager.block_size, manager.num_free_blocks) == (6, 4, 6)
    assert all(manager.ref_count(block) == 0 for block in range(6))


@pytest.mark.parametrize(("num_tokens", "num_blocks"), [(1, 1), (4, 1), (5, 2), (12, 3)])
def test_allocation_takes_as_many_blocks_as_the_tokens_fill(
    num_tokens: int, num_blocks: int
) -> None:
    manager = BlockManager(num_blocks=6, block_size=BLOCK_SIZE)
    assert manager.blocks_needed(num_tokens) == num_blocks
    block_table = manager.allocate(num_tokens)
    assert block_table == list(range(num_blocks))
    assert manager.num_free_blocks == 6 - num_blocks
    assert all(manager.ref_count(block) == 1 for block in block_table)


def test_tables_get_distinct_blocks() -> None:
    manager = BlockManager(num_blocks=6, block_size=BLOCK_SIZE)
    first = manager.allocate(5)
    second = manager.allocate(9)
    assert first == [0, 1]
    assert second == [2, 3, 4]


def test_allocation_must_leave_the_watermark_free() -> None:
    manager = BlockManager(num_blocks=6, block_size=BLOCK_SIZE, watermark=2)
    assert manager.can_allocate(16)
    assert not manager.can_allocate(17)
    manager.allocate(16)
    assert not manager.can_allocate(1)


def test_allocating_more_than_is_free_raises_and_takes_nothing() -> None:
    manager = BlockManager(num_blocks=2, block_size=BLOCK_SIZE)
    with pytest.raises(NoFreeBlocksError, match="9 tokens need 3 more blocks, but 2 are free"):
        manager.allocate(9)
    assert manager.num_free_blocks == 2


def test_a_table_grows_by_a_block_when_its_blocks_are_full() -> None:
    manager = BlockManager(num_blocks=3, block_size=BLOCK_SIZE)
    block_table = manager.allocate(3)
    # The fourth token takes the last slot of the first block.
    manager.append_slot(block_table, 4)
    assert block_table == [0]
    manager.append_slot(block_table, 5)
    assert block_table == [0, 1]
    assert manager.num_free_blocks == 1


def test_growing_may_use_the_watermark() -> None:
    manager = BlockManager(num_blocks=3, block_size=BLOCK_SIZE, watermark=1)
    block_table = manager.allocate(8)
    assert not manager.can_allocate(1)
    assert manager.can_append(block_table, 9)
    manager.append_slot(block_table, 9)
    assert block_table == [0, 1, 2]


def test_a_full_pool_can_still_fill_the_last_block() -> None:
    manager = BlockManager(num_blocks=2, block_size=BLOCK_SIZE)
    block_table = manager.allocate(7)
    assert manager.num_free_blocks == 0
    assert manager.can_append(block_table, 8)
    assert not manager.can_append(block_table, 9)
    with pytest.raises(NoFreeBlocksError):
        manager.append_slot(block_table, 9)
    assert block_table == [0, 1]


def test_freed_blocks_are_handed_out_again() -> None:
    manager = BlockManager(num_blocks=4, block_size=BLOCK_SIZE)
    first = manager.allocate(8)
    second = manager.allocate(8)
    manager.free(first)
    assert first == []
    assert manager.num_free_blocks == 2
    assert manager.ref_count(0) == manager.ref_count(1) == 0
    # The lowest free blocks come first, in order.
    assert manager.allocate(8) == [0, 1]
    assert second == [2, 3]


def test_freeing_a_free_block_raises_and_frees_nothing() -> None:
    manager = BlockManager(num_blocks=4, block_size=BLOCK_SIZE)
    block_table = manager.allocate(8)
    copy = list(block_table)
    manager.free(block_table)
    other = manager.allocate(4)
    with pytest.raises(ValueError, match="block 1 is free already"):
        manager.free(copy)
    assert manager.ref_count(other[0]) == 1
    assert manager.num_free_blocks == 3


@pytest.mark.parametrize(
    ("num_blocks", "block_size", "watermark", "message"),
    [
        (0, 4, 0, "num_blocks must be 1 or greater"),
        (4, 0, 0, "block_size must be 1 or greater"),
        (4, 4, 4, r"watermark must be in \[0, 4\)"),
        (4, 4, -1, r"watermark must be in \[0, 4\)"),
    ],
)
def test_invalid_sizes_raise(
    num_blocks: int, block_size: int, watermark: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BlockManager(num_blocks, block_size, watermark)


def test_an_empty_allocation_raises() -> None:
    with pytest.raises(ValueError, match="num_tokens must be 1 or greater"):
        BlockManager(num_blocks=2, block_size=BLOCK_SIZE).allocate(0)


def test_a_block_hash_stands_for_its_tokens_and_every_token_before_them() -> None:
    assert block_hash(None, TOKENS[:4]) == FIRST
    assert len(FIRST) == 32
    assert block_hash(None, [11, 12, 13, 15]) != FIRST
    # The same tokens after another prefix, or at the start, are another block.
    assert block_hash(block_hash(None, [1, 2, 3, 4]), TOKENS[4:8]) != SECOND
    assert block_hash(None, TOKENS[4:8]) != SECOND


def cached_manager(num_blocks: int = 8) -> tuple[BlockManager, list[int]]:
    """Returns a manager with one sequence of TOKENS whose two full blocks are cached."""
    manager = BlockManager(num_blocks=num_blocks, block_size=BLOCK_SIZE)
    block_table = manager.allocate(len(TOKENS))
    manager.cache_block(block_table[0], FIRST)
    manager.cache_block(block_table[1], SECOND)
    return manager, block_table


def test_a_new_table_shares_the_cached_blocks_of_its_prefix() -> None:
    manager, first = cached_manager()
    assert first == [0, 1, 2]
    shared = manager.cached_prefix([FIRST, SECOND])
    assert shared == [0, 1]
    second = manager.allocate(9, shared)
    assert second == [0, 1, 3]
    assert [manager.ref_count(block) for block in range(4)] == [2, 2, 1, 1]
    assert manager.num_free_blocks == 4


def test_the_cached_prefix_ends_at_the_first_block_not_in_the_cache() -> None:
    manager, _ = cached_manager()
    other = block_hash(None, [1, 2, 3, 4])
    assert manager.cached_prefix([FIRST, other, SECOND]) == [0]
    assert manager.cached_prefix([other, SECOND]) == []
    assert manager.cached_prefix([]) == []


def test_shared_blocks_take_no_free_blocks() -> None:
    manager, _ = cached_manager(num_blocks=4)
    assert manager.num_free_blocks == 1
    assert not manager.can_allocate(12)
    assert manager.can_allocate(12, [0, 1])
    manager.allocate(12, [0, 1])
    assert manager.num_free_blocks == 0


def test_a_shared_block_is_free_once_no_table_holds_it() -> None:
    manager, first = cached_manager()
    second = manager.allocate(5, manager.cached_prefix([FIRST]))
    manager.free(first)
    # Block 0 is still held by the second table, blocks 1 and 2 by none.
    assert [manager.ref_count(block) for block in range(4)] == [1, 0, 0, 1]
    assert manager.num_free_blocks == 8 - 2
    manager.free(second)
    assert manager.num_free_blocks == 8


def test_a_freed_cached_block_stays_cached() -> None:
    manager, first = cached_manager()
    manager.free(first)
    assert manager.num_free_blocks == 8
    assert manager.cached_prefix([FIRST, SECOND]) == [0, 1]
    second = manager.allocate(9, [0, 1])
    assert second == [0, 1, 2]
    assert [manager.ref_count(block) for block in range(3)] == [1, 1, 1]
    assert manager.num_free_blocks == 8 - 3


def test_free_blocks_outside_the_cache_are_taken_before_cached_ones() -> None:
    manager, first = cached_manager(num_blocks=4)
    manager.free(first)
    # Blocks 2 and 3 are free and not cached.
    assert manager.allocate(8) == [2, 3]
    assert manager.cached_prefix([FIRST, SECOND]) == [0, 1]
    # Then the table's last cached block goes first, since the first one is still useful
    # without it, but not the other way around.
    assert manager.allocate(4) == [1]
    assert manager.cached_prefix([FIRST, SECOND]) == [0]
    assert manager.allocate(4) == [0]
    assert manager.cached_prefix([FIRST]) == []
    assert manager.num_free_blocks == 0


def test_the_cached_block_freed_longest_ago_is_evicted_first() -> None:
    manager = BlockManager(num_blocks=4, block_size=BLOCK_SIZE)
    other = block_hash(None, [1, 2, 3, 4])
    first, second = manager.allocate(4), manager.allocate(4)
    manager.cache_block(first[0], FIRST)
    manager.cache_block(second[0], other)
    manager.free(first)
    manager.free(second)
    # Sharing block 0 again and freeing it makes it the most recently freed.
    manager.free(manager.allocate(5, [0]))
    # The blocks outside the cache go first, then block 1 before block 0.
    assert manager.allocate(8) == [2, 3]
    assert manager.allocate(4) == [1]
    assert manager.cached_prefix([other]) == []
    assert manager.cached_prefix([FIRST]) == [0]


def test_sharing_a_freed_cached_block_takes_a_free_block() -> None:
    manager = BlockManager(num_blocks=4, block_size=BLOCK_SIZE, watermark=1)
    block_table = manager.allocate(8)
    manager.cache_block(block_table[0], FIRST)
    manager.cache_block(block_table[1], SECOND)
    manager.free(block_table)
    # Sharing both blocks and taking one more leaves the watermark's one block free.
    assert manager.can_allocate(12, [0, 1])
    assert not manager.can_allocate(16, [0, 1])
    with pytest.raises(NoFreeBlocksError, match="20 tokens need 5 more blocks, but 4 are free"):
        manager.allocate(20, [0, 1])
    assert manager.cached_prefix([FIRST, SECOND]) == [0, 1]


def test_a_new_table_never_evicts_a_block_it_shares() -> None:
    manager = BlockManager(num_blocks=2, block_size=BLOCK_SIZE)
    shared, other = manager.allocate(4), manager.allocate(4)
    manager.cache_block(shared[0], FIRST)
    manager.cache_block(other[0], block_hash(None, [1, 2, 3, 4]))
    manager.free(shared)
    manager.free(other)
    # Block 0 was freed first, so it would be evicted first if it were not shared.
    assert manager.allocate(8, [0]) == [0, 1]
    assert manager.cached_prefix([FIRST]) == [0]


def test_the_cache_keeps_the_first_block_cached_under_a_hash() -> None:
    manager, _ = cached_manager()
    # A second table that computed a block with the same tokens itself.
    duplicate = manager.allocate(4)
    manager.cache_block(duplicate[0], FIRST)
    assert manager.cached_prefix([FIRST]) == [0]
    # It is still its table's own, so freeing it does not touch the cache.
    manager.free(duplicate)
    assert manager.cached_prefix([FIRST, SECOND]) == [0, 1]
    # Caching a block again under its own hash changes nothing.
    manager.cache_block(0, FIRST)
    assert manager.cached_prefix([FIRST]) == [0]


def test_invalid_caching_and_sharing_raise() -> None:
    manager, _ = cached_manager(num_blocks=4)
    with pytest.raises(ValueError, match="block 3 is not held by a table and cannot be cached"):
        manager.cache_block(3, FIRST)
    with pytest.raises(ValueError, match="block 0 is cached already, under another hash"):
        manager.cache_block(0, SECOND)
    with pytest.raises(ValueError, match="block 2 is not cached and cannot be shared"):
        manager.allocate(12, [0, 2])
    with pytest.raises(ValueError, match="2 shared blocks hold more than 7 tokens"):
        manager.allocate(7, [0, 1])
    with pytest.raises(NoFreeBlocksError, match="16 tokens need 2 more blocks, but 1 are free"):
        manager.allocate(16, [0, 1])
    assert [manager.ref_count(block) for block in range(4)] == [1, 1, 1, 0]
