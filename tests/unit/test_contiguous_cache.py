import pytest
import torch
from torch import Tensor

from kiyal.cache.contiguous import ContiguousKVCache, ContiguousKVPool
from kiyal.model.config import ModelConfig

BATCH_SIZE = 2
MAX_TOKENS = 10


@pytest.fixture
def cache(tiny_config: ModelConfig) -> ContiguousKVCache:
    return ContiguousKVCache(tiny_config, BATCH_SIZE, MAX_TOKENS)


def random_kv(config: ModelConfig, num_tokens: int, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    shape = (BATCH_SIZE, config.num_key_value_heads, num_tokens, config.head_dim)
    return torch.randn(shape, generator=generator), torch.randn(shape, generator=generator)


def test_a_new_cache_is_empty(cache: ContiguousKVCache) -> None:
    assert cache.length == 0
    assert cache.max_tokens == MAX_TOKENS


def test_write_returns_cached_then_new_tokens(
    cache: ContiguousKVCache, tiny_config: ModelConfig
) -> None:
    prompt_keys, prompt_values = random_kv(tiny_config, 4, seed=0)
    keys, values = cache.write(0, prompt_keys, prompt_values)
    torch.testing.assert_close(keys, prompt_keys)
    torch.testing.assert_close(values, prompt_values)
    cache.advance(4)

    new_keys, new_values = random_kv(tiny_config, 1, seed=1)
    keys, values = cache.write(0, new_keys, new_values)
    torch.testing.assert_close(keys, torch.cat([prompt_keys, new_keys], dim=2))
    torch.testing.assert_close(values, torch.cat([prompt_values, new_values], dim=2))


def test_layers_are_stored_separately(cache: ContiguousKVCache, tiny_config: ModelConfig) -> None:
    first = random_kv(tiny_config, 3, seed=0)
    second = random_kv(tiny_config, 3, seed=1)
    cache.write(0, *first)
    cache.write(1, *second)
    cache.advance(3)

    step = random_kv(tiny_config, 1, seed=2)
    keys, values = cache.write(0, *step)
    torch.testing.assert_close(keys[:, :, :3], first[0])
    torch.testing.assert_close(values[:, :, :3], first[1])
    keys, values = cache.write(1, *step)
    torch.testing.assert_close(keys[:, :, :3], second[0])
    torch.testing.assert_close(values[:, :, :3], second[1])


def test_length_changes_only_on_advance(cache: ContiguousKVCache, tiny_config: ModelConfig) -> None:
    for layer in range(tiny_config.num_hidden_layers):
        keys, _ = cache.write(layer, *random_kv(tiny_config, 4, seed=layer))
        # Every layer writes at the same positions.
        assert keys.shape[2] == 4
        assert cache.length == 0
    cache.advance(4)
    assert cache.length == 4


def test_write_returns_views_without_copying(
    cache: ContiguousKVCache, tiny_config: ModelConfig
) -> None:
    first_keys, _ = cache.write(0, *random_kv(tiny_config, 2, seed=0))
    cache.advance(2)
    second_keys, _ = cache.write(0, *random_kv(tiny_config, 1, seed=1))
    assert first_keys.data_ptr() == second_keys.data_ptr()


def test_the_cache_can_be_filled_completely(
    cache: ContiguousKVCache, tiny_config: ModelConfig
) -> None:
    cache.write(0, *random_kv(tiny_config, MAX_TOKENS - 1, seed=0))
    cache.advance(MAX_TOKENS - 1)
    keys, _ = cache.write(0, *random_kv(tiny_config, 1, seed=1))
    cache.advance(1)
    assert keys.shape[2] == cache.length == MAX_TOKENS


def test_writing_past_the_end_raises(cache: ContiguousKVCache, tiny_config: ModelConfig) -> None:
    cache.write(0, *random_kv(tiny_config, 8, seed=0))
    cache.advance(8)
    with pytest.raises(ValueError, match="holds 8 of 10 tokens and cannot take 3 more"):
        cache.write(0, *random_kv(tiny_config, 3, seed=1))


@pytest.mark.parametrize("num_tokens", [0, -1, MAX_TOKENS + 1])
def test_advancing_by_an_invalid_count_raises(cache: ContiguousKVCache, num_tokens: int) -> None:
    with pytest.raises(ValueError, match="cannot advance"):
        cache.advance(num_tokens)
    assert cache.length == 0


def test_a_wrong_shape_raises_instead_of_broadcasting(
    cache: ContiguousKVCache, tiny_config: ModelConfig
) -> None:
    keys, values = random_kv(tiny_config, 3, seed=0)
    with pytest.raises(ValueError, match=r"must be \[2, 2, T, 16\]"):
        cache.write(0, keys[:1], values[:1])
    with pytest.raises(ValueError, match=r"must be \[2, 2, T, 16\]"):
        cache.write(0, keys, values[:, :, :2])
    with pytest.raises(ValueError, match=r"must be \[2, 2, T, 16\]"):
        cache.write(0, keys[..., :8], values[..., :8])


@pytest.mark.parametrize(("batch_size", "max_tokens"), [(0, 4), (1, 0)])
def test_empty_sizes_raise(tiny_config: ModelConfig, batch_size: int, max_tokens: int) -> None:
    with pytest.raises(ValueError, match="must be 1 or greater"):
        ContiguousKVCache(tiny_config, batch_size, max_tokens)


def test_the_cache_uses_the_requested_dtype(tiny_config: ModelConfig) -> None:
    cache = ContiguousKVCache(tiny_config, 1, 4, device=torch.device("cpu"), dtype=torch.float16)
    shape = (1, tiny_config.num_key_value_heads, 2, tiny_config.head_dim)
    keys, values = cache.write(0, torch.ones(shape), torch.ones(shape))
    assert keys.dtype == values.dtype == torch.float16


POOL_ROWS = 3
POOL_LEN = 8


@pytest.fixture
def pool(tiny_config: ModelConfig) -> ContiguousKVPool:
    return ContiguousKVPool(tiny_config, POOL_ROWS, POOL_LEN)


def random_token_kv(config: ModelConfig, num_tokens: int, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    shape = (num_tokens, config.num_key_value_heads, config.head_dim)
    return torch.randn(shape, generator=generator), torch.randn(shape, generator=generator)


def slots(row: int, positions: range) -> list[int]:
    return [row * POOL_LEN + position for position in positions]


def test_a_new_pool_holds_zeros(pool: ContiguousKVPool) -> None:
    assert pool.num_rows == POOL_ROWS
    assert pool.max_len == POOL_LEN
    keys, values = pool.read(0, torch.arange(POOL_ROWS), POOL_LEN)
    assert not keys.any()
    assert not values.any()


def test_the_pool_stores_tokens_of_several_rows_in_one_write(
    pool: ContiguousKVPool, tiny_config: ModelConfig
) -> None:
    # Three tokens of row 2 and one of row 0.
    new_keys, new_values = random_token_kv(tiny_config, 4, seed=0)
    pool.write(0, torch.tensor(slots(2, range(3)) + slots(0, range(1))), new_keys, new_values)
    keys, values = pool.read(0, torch.tensor([2, 0]), 3)
    assert keys.shape == (2, tiny_config.num_key_value_heads, 3, tiny_config.head_dim)
    torch.testing.assert_close(keys[0], new_keys[:3].transpose(0, 1))
    torch.testing.assert_close(values[0], new_values[:3].transpose(0, 1))
    torch.testing.assert_close(keys[1, :, 0], new_keys[3])
    torch.testing.assert_close(values[1, :, 0], new_values[3])
    # Positions nobody wrote are still zero.
    assert not keys[1, :, 1:].any()


def test_a_row_grows_one_position_at_a_time(
    pool: ContiguousKVPool, tiny_config: ModelConfig
) -> None:
    prompt = random_token_kv(tiny_config, 2, seed=0)
    pool.write(0, torch.tensor(slots(1, range(2))), *prompt)
    step = random_token_kv(tiny_config, 1, seed=1)
    pool.write(0, torch.tensor(slots(1, range(2, 3))), *step)
    keys, _ = pool.read(0, torch.tensor([1]), 3)
    torch.testing.assert_close(keys[0], torch.cat([prompt[0], step[0]]).transpose(0, 1))


def test_pool_layers_are_stored_separately(
    pool: ContiguousKVPool, tiny_config: ModelConfig
) -> None:
    first = random_token_kv(tiny_config, 2, seed=0)
    second = random_token_kv(tiny_config, 2, seed=1)
    pool.write(0, torch.tensor(slots(0, range(2))), *first)
    pool.write(1, torch.tensor(slots(0, range(2))), *second)
    keys, values = pool.read(0, torch.tensor([0]), 2)
    torch.testing.assert_close(keys[0], first[0].transpose(0, 1))
    torch.testing.assert_close(values[0], first[1].transpose(0, 1))
    keys, values = pool.read(1, torch.tensor([0]), 2)
    torch.testing.assert_close(keys[0], second[0].transpose(0, 1))
    torch.testing.assert_close(values[0], second[1].transpose(0, 1))


def test_reading_copies_the_rows(pool: ContiguousKVPool, tiny_config: ModelConfig) -> None:
    keys, _ = pool.read(0, torch.tensor([0]), 1)
    pool.write(0, torch.tensor([0]), *random_token_kv(tiny_config, 1, seed=0))
    assert not keys.any()


def test_viewing_returns_the_first_rows_without_copying(
    pool: ContiguousKVPool, tiny_config: ModelConfig
) -> None:
    keys, values = pool.view(0, 2, 3)
    assert (
        keys.shape == values.shape == (2, tiny_config.num_key_value_heads, 3, tiny_config.head_dim)
    )
    new_keys, new_values = random_token_kv(tiny_config, 1, seed=0)
    pool.write(0, torch.tensor(slots(1, range(2, 3))), new_keys, new_values)
    # The views show the write, and match what a copying read returns.
    torch.testing.assert_close(keys[1, :, 2], new_keys[0])
    torch.testing.assert_close(values[1, :, 2], new_values[0])
    torch.testing.assert_close(keys, pool.read(0, torch.tensor([0, 1]), 3)[0])


def test_a_pool_write_of_the_wrong_size_raises_instead_of_broadcasting(
    pool: ContiguousKVPool, tiny_config: ModelConfig
) -> None:
    keys, values = random_token_kv(tiny_config, 1, seed=0)
    with pytest.raises(IndexError, match="Number of indices"):
        pool.write(0, torch.tensor(slots(0, range(2))), keys, values)


@pytest.mark.parametrize(("num_rows", "max_len"), [(0, 4), (1, 0)])
def test_empty_pool_sizes_raise(tiny_config: ModelConfig, num_rows: int, max_len: int) -> None:
    with pytest.raises(ValueError, match="must be 1 or greater"):
        ContiguousKVPool(tiny_config, num_rows, max_len)


def test_the_pool_uses_the_requested_dtype(tiny_config: ModelConfig) -> None:
    pool = ContiguousKVPool(tiny_config, 1, 4, device=torch.device("cpu"), dtype=torch.float16)
    keys, values = pool.read(0, torch.tensor([0]), 2)
    assert keys.dtype == values.dtype == torch.float16
