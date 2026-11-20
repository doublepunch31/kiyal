import pytest

from kiyal.config import EngineConfig


def test_the_defaults_are_valid() -> None:
    config = EngineConfig()
    assert config.max_num_batched_tokens >= config.max_num_seqs >= 1
    assert config.max_model_len >= 1


@pytest.mark.parametrize(
    "name",
    ["max_num_seqs", "max_num_batched_tokens", "max_model_len", "block_size", "num_kv_blocks"],
)
def test_limits_below_1_raise(name: str) -> None:
    with pytest.raises(ValueError, match=f"{name} must be 1 or greater, got 0"):
        EngineConfig(**{name: 0})


def test_a_decode_step_must_fit_in_the_token_budget() -> None:
    EngineConfig(max_num_seqs=8, max_num_batched_tokens=8)
    with pytest.raises(ValueError, match=r"max_num_batched_tokens \(7\) must be at least"):
        EngineConfig(max_num_seqs=8, max_num_batched_tokens=7)


def test_the_default_pool_size_is_left_to_the_engine() -> None:
    assert EngineConfig().num_kv_blocks is None


@pytest.mark.parametrize("watermark", [-0.1, 1.0])
def test_a_watermark_outside_0_to_1_raises(watermark: float) -> None:
    with pytest.raises(ValueError, match=r"watermark must be in \[0, 1\)"):
        EngineConfig(watermark=watermark)
