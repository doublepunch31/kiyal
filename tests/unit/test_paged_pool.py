import pytest
import torch
from torch import Tensor

from kiyal.cache.paged import PagedKVPool
from kiyal.model.config import ModelConfig

NUM_BLOCKS = 6
BLOCK_SIZE = 4


@pytest.fixture
def pool(tiny_config: ModelConfig) -> PagedKVPool:
    return PagedKVPool(tiny_config, NUM_BLOCKS, BLOCK_SIZE)


def random_token_kv(config: ModelConfig, num_tokens: int, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    shape = (num_tokens, config.num_key_value_heads, config.head_dim)
    return torch.randn(shape, generator=generator), torch.randn(shape, generator=generator)


def slots(block_table: list[int], positions: range) -> list[int]:
    """Pool slots of a sequence's tokens at `positions`."""
    return [block_table[p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE for p in positions]


def test_a_new_pool_holds_zeros(pool: PagedKVPool) -> None:
    assert pool.num_blocks == NUM_BLOCKS
    assert pool.block_size == BLOCK_SIZE
    keys, values = pool.gather(0, torch.arange(NUM_BLOCKS).view(1, -1), NUM_BLOCKS * BLOCK_SIZE)
    assert not keys.any()
    assert not values.any()


def test_a_sequence_reads_back_in_order_from_scattered_blocks(
    pool: PagedKVPool, tiny_config: ModelConfig
) -> None:
    # Ten tokens over three blocks, the last one in part, out of order in the pool.
    block_table = [4, 0, 3]
    new_keys, new_values = random_token_kv(tiny_config, 10, seed=0)
    pool.write(0, torch.tensor(slots(block_table, range(10))), new_keys, new_values)
    keys, values = pool.gather(0, torch.tensor([block_table]), 10)
    assert keys.shape == (1, tiny_config.num_key_value_heads, 10, tiny_config.head_dim)
    torch.testing.assert_close(keys[0], new_keys.transpose(0, 1))
    torch.testing.assert_close(values[0], new_values.transpose(0, 1))


def test_the_pool_stores_tokens_of_several_sequences_in_one_write(
    pool: PagedKVPool, tiny_config: ModelConfig
) -> None:
    # Six tokens of a sequence in blocks 2 and 5, and one of a sequence in block 1.
    tables = [[2, 5], [1]]
    new_keys, new_values = random_token_kv(tiny_config, 7, seed=0)
    slot_mapping = slots(tables[0], range(6)) + slots(tables[1], range(1))
    pool.write(0, torch.tensor(slot_mapping), new_keys, new_values)
    # The shorter sequence's table is padded with block 0, as the attention metadata does.
    keys, values = pool.gather(0, torch.tensor([[2, 5], [1, 0]]), 6)
    assert keys.shape == (2, tiny_config.num_key_value_heads, 6, tiny_config.head_dim)
    torch.testing.assert_close(keys[0], new_keys[:6].transpose(0, 1))
    torch.testing.assert_close(values[0], new_values[:6].transpose(0, 1))
    torch.testing.assert_close(keys[1, :, 0], new_keys[6])
    torch.testing.assert_close(values[1, :, 0], new_values[6])
    # Positions nobody wrote are still zero.
    assert not keys[1, :, 1:].any()


def test_a_sequence_grows_into_a_new_block(pool: PagedKVPool, tiny_config: ModelConfig) -> None:
    block_table = [3]
    prompt = random_token_kv(tiny_config, BLOCK_SIZE, seed=0)
    pool.write(0, torch.tensor(slots(block_table, range(BLOCK_SIZE))), *prompt)
    block_table.append(1)
    step = random_token_kv(tiny_config, 1, seed=1)
    pool.write(0, torch.tensor(slots(block_table, range(BLOCK_SIZE, BLOCK_SIZE + 1))), *step)
    keys, _ = pool.gather(0, torch.tensor([block_table]), BLOCK_SIZE + 1)
    torch.testing.assert_close(keys[0], torch.cat([prompt[0], step[0]]).transpose(0, 1))


def test_pool_layers_are_stored_separately(pool: PagedKVPool, tiny_config: ModelConfig) -> None:
    first = random_token_kv(tiny_config, 2, seed=0)
    second = random_token_kv(tiny_config, 2, seed=1)
    pool.write(0, torch.tensor(slots([0], range(2))), *first)
    pool.write(1, torch.tensor(slots([0], range(2))), *second)
    keys, values = pool.gather(0, torch.tensor([[0]]), 2)
    torch.testing.assert_close(keys[0], first[0].transpose(0, 1))
    torch.testing.assert_close(values[0], first[1].transpose(0, 1))
    keys, values = pool.gather(1, torch.tensor([[0]]), 2)
    torch.testing.assert_close(keys[0], second[0].transpose(0, 1))
    torch.testing.assert_close(values[0], second[1].transpose(0, 1))


def test_gathering_copies_the_blocks(pool: PagedKVPool, tiny_config: ModelConfig) -> None:
    keys, _ = pool.gather(0, torch.tensor([[0]]), 1)
    pool.write(0, torch.tensor([0]), *random_token_kv(tiny_config, 1, seed=0))
    assert not keys.any()


def test_viewing_returns_the_first_blocks_without_copying(
    pool: PagedKVPool, tiny_config: ModelConfig
) -> None:
    keys, values = pool.view(0, 2, 3)
    assert (
        keys.shape == values.shape == (2, tiny_config.num_key_value_heads, 3, tiny_config.head_dim)
    )
    new_keys, new_values = random_token_kv(tiny_config, 1, seed=0)
    pool.write(0, torch.tensor(slots([1], range(2, 3))), new_keys, new_values)
    # The views show the write, and match what a copying gather returns.
    torch.testing.assert_close(keys[1, :, 2], new_keys[0])
    torch.testing.assert_close(values[1, :, 2], new_values[0])
    torch.testing.assert_close(keys, pool.gather(0, torch.tensor([[0], [1]]), 3)[0])


def test_viewing_more_than_a_block_raises(pool: PagedKVPool) -> None:
    with pytest.raises(ValueError, match="cannot view 5 positions of blocks of 4"):
        pool.view(0, 2, BLOCK_SIZE + 1)


def test_a_pool_write_of_the_wrong_size_raises_instead_of_broadcasting(
    pool: PagedKVPool, tiny_config: ModelConfig
) -> None:
    keys, values = random_token_kv(tiny_config, 1, seed=0)
    with pytest.raises(IndexError, match="Number of indices"):
        pool.write(0, torch.tensor(slots([0], range(2))), keys, values)


@pytest.mark.parametrize(("num_blocks", "block_size"), [(0, 4), (1, 0)])
def test_empty_pool_sizes_raise(tiny_config: ModelConfig, num_blocks: int, block_size: int) -> None:
    with pytest.raises(ValueError, match="must be 1 or greater"):
        PagedKVPool(tiny_config, num_blocks, block_size)


def test_the_pool_uses_the_requested_dtype(tiny_config: ModelConfig) -> None:
    pool = PagedKVPool(tiny_config, 1, 4, device=torch.device("cpu"), dtype=torch.float16)
    keys, values = pool.gather(0, torch.tensor([[0]]), 2)
    assert keys.dtype == values.dtype == torch.float16
