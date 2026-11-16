import dataclasses

import pytest
import torch
from torch import Tensor

from kiyal.cache.contiguous import ContiguousKVCache
from kiyal.model.config import ModelConfig
from kiyal.model.llama import LlamaModel

BATCH_SIZE = 2
SEQ_LEN = 6
# A cached forward multiplies matrices of other shapes than an uncached one, so its results
# round differently in the last bits.
CACHED_RTOL = 1e-5
CACHED_ATOL = 1e-5


@pytest.fixture
def model(tiny_config: ModelConfig) -> LlamaModel:
    torch.manual_seed(0)
    return LlamaModel(tiny_config)


def token_batch(config: ModelConfig) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN), generator=generator)
    positions = torch.arange(SEQ_LEN).expand(BATCH_SIZE, SEQ_LEN)
    return input_ids, positions


def all_logits(model: LlamaModel, input_ids: Tensor, positions: Tensor) -> Tensor:
    batch_size, seq_len = input_ids.shape
    indices = torch.arange(batch_size * seq_len)
    return model(input_ids, positions, indices).view(batch_size, seq_len, -1)


@torch.inference_mode()
def test_default_returns_last_token_logits(model: LlamaModel) -> None:
    input_ids, positions = token_batch(model.config)
    logits = model(input_ids, positions)
    assert logits.shape == (BATCH_SIZE, model.config.vocab_size)
    torch.testing.assert_close(logits, all_logits(model, input_ids, positions)[:, -1])


@torch.inference_mode()
def test_logits_indices_select_flattened_tokens(model: LlamaModel) -> None:
    input_ids, positions = token_batch(model.config)
    # Row 0 tokens 0 and 5, row 1 tokens 1 and 5.
    indices = torch.tensor([0, 5, 7, 11])
    expected = all_logits(model, input_ids, positions).reshape(BATCH_SIZE * SEQ_LEN, -1)[indices]
    torch.testing.assert_close(model(input_ids, positions, indices), expected)


@torch.inference_mode()
def test_rows_of_a_batch_do_not_interact(model: LlamaModel) -> None:
    input_ids, positions = token_batch(model.config)
    batched = all_logits(model, input_ids, positions)
    for row in range(BATCH_SIZE):
        alone = all_logits(model, input_ids[row : row + 1], positions[row : row + 1])
        torch.testing.assert_close(alone[0], batched[row])


@torch.inference_mode()
def test_later_tokens_do_not_change_earlier_logits(model: LlamaModel) -> None:
    input_ids, positions = token_batch(model.config)
    changed = input_ids.clone()
    changed[:, -1] = (changed[:, -1] + 1) % model.config.vocab_size
    before = all_logits(model, input_ids, positions)
    after = all_logits(model, changed, positions)
    torch.testing.assert_close(after[:, :-1], before[:, :-1])


@torch.inference_mode()
def test_shifting_all_positions_keeps_logits(model: LlamaModel) -> None:
    # Rotary attention scores depend only on the distance between tokens.
    input_ids, positions = token_batch(model.config)
    torch.testing.assert_close(model(input_ids, positions + 40), model(input_ids, positions))


@torch.inference_mode()
def test_spreading_positions_changes_logits(model: LlamaModel) -> None:
    input_ids, positions = token_batch(model.config)
    assert not torch.allclose(model(input_ids, positions * 2), model(input_ids, positions))


def cached_logits(model: LlamaModel, input_ids: Tensor, positions: Tensor, chunks: list[int]):
    """All-position logits from feeding the batch to a cached forward in pieces."""
    batch_size, seq_len = input_ids.shape
    cache = ContiguousKVCache(model.config, batch_size, seq_len)
    pieces: list[Tensor] = []
    for size in chunks:
        piece = slice(cache.length, cache.length + size)
        logits = model(
            input_ids[:, piece], positions[:, piece], torch.arange(batch_size * size), cache
        )
        pieces.append(logits.view(batch_size, size, -1))
    assert cache.length == seq_len
    return torch.cat(pieces, dim=1)


@torch.inference_mode()
def test_prefill_into_a_cache_matches_the_uncached_forward(model: LlamaModel) -> None:
    input_ids, positions = token_batch(model.config)
    expected = all_logits(model, input_ids, positions)
    torch.testing.assert_close(cached_logits(model, input_ids, positions, [SEQ_LEN]), expected)


@pytest.mark.parametrize("prompt_len", [1, 3, SEQ_LEN - 1])
@torch.inference_mode()
def test_decoding_from_a_cache_matches_the_uncached_forward(
    model: LlamaModel, prompt_len: int
) -> None:
    input_ids, positions = token_batch(model.config)
    chunks = [prompt_len] + [1] * (SEQ_LEN - prompt_len)
    expected = all_logits(model, input_ids, positions)
    torch.testing.assert_close(
        cached_logits(model, input_ids, positions, chunks),
        expected,
        rtol=CACHED_RTOL,
        atol=CACHED_ATOL,
    )


@torch.inference_mode()
def test_prefill_in_pieces_matches_the_uncached_forward(model: LlamaModel) -> None:
    input_ids, positions = token_batch(model.config)
    expected = all_logits(model, input_ids, positions)
    torch.testing.assert_close(
        cached_logits(model, input_ids, positions, [2, 3, 1]),
        expected,
        rtol=CACHED_RTOL,
        atol=CACHED_ATOL,
    )


@torch.inference_mode()
def test_a_cached_forward_returns_last_token_logits_by_default(model: LlamaModel) -> None:
    input_ids, positions = token_batch(model.config)
    cache = ContiguousKVCache(model.config, BATCH_SIZE, SEQ_LEN)
    expected = all_logits(model, input_ids, positions)
    prefill = model(input_ids[:, :4], positions[:, :4], cache=cache)
    assert cache.length == 4
    torch.testing.assert_close(prefill, expected[:, 3], rtol=CACHED_RTOL, atol=CACHED_ATOL)
    decode = model(input_ids[:, 4:5], positions[:, 4:5], cache=cache)
    assert cache.length == 5
    torch.testing.assert_close(decode, expected[:, 4], rtol=CACHED_RTOL, atol=CACHED_ATOL)


@torch.inference_mode()
def test_cached_decoding_depends_on_the_cached_tokens(model: LlamaModel) -> None:
    input_ids, positions = token_batch(model.config)
    changed = input_ids.clone()
    changed[:, 0] = (changed[:, 0] + 1) % model.config.vocab_size

    def last_logits(prompt: Tensor) -> Tensor:
        cache = ContiguousKVCache(model.config, BATCH_SIZE, SEQ_LEN)
        model(prompt[:, :-1], positions[:, :-1], cache=cache)
        return model(prompt[:, -1:], positions[:, -1:], cache=cache)

    assert not torch.allclose(last_logits(changed), last_logits(input_ids))


@torch.inference_mode()
def test_a_forward_that_overflows_the_cache_raises(model: LlamaModel) -> None:
    input_ids, positions = token_batch(model.config)
    cache = ContiguousKVCache(model.config, BATCH_SIZE, SEQ_LEN - 1)
    with pytest.raises(ValueError, match="cannot take 6 more"):
        model(input_ids, positions, cache=cache)
    assert cache.length == 0


def test_tied_head_shares_the_embedding(model: LlamaModel) -> None:
    assert model.config.tie_word_embeddings
    assert model.lm_head.weight is model.embed_tokens.weight
    assert "lm_head.weight" not in dict(model.named_parameters())


def test_untied_head_has_its_own_weight(tiny_config: ModelConfig) -> None:
    model = LlamaModel(dataclasses.replace(tiny_config, tie_word_embeddings=False))
    assert model.lm_head.weight is not model.embed_tokens.weight
    assert "lm_head.weight" in dict(model.named_parameters())


@torch.inference_mode()
def test_parameters_use_requested_dtype(tiny_config: ModelConfig) -> None:
    model = LlamaModel(tiny_config, device=torch.device("cpu"), dtype=torch.float16)
    assert all(p.dtype == torch.float16 for p in model.parameters())
    # The rotary tables stay fp32 and are cast on lookup.
    assert model.rotary.cos.dtype == torch.float32
    input_ids, positions = token_batch(tiny_config)
    assert model(input_ids, positions).dtype == torch.float16
