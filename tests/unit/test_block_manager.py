import pytest

from kiyal.cache.block_manager import BlockManager, NoFreeBlocksError

BLOCK_SIZE = 4


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
