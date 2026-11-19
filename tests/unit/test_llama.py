import dataclasses

import pytest
import torch
from torch import Tensor

from kiyal.cache.contiguous import ContiguousKVCache, ContiguousKVPool
from kiyal.model.attention import AttentionMetadata
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


def all_logits(
    model: LlamaModel, input_ids: Tensor, positions: Tensor, left_padding: Tensor | None = None
) -> Tensor:
    batch_size, seq_len = input_ids.shape
    indices = torch.arange(batch_size * seq_len)
    logits = model(input_ids, positions, indices, left_padding=left_padding)
    return logits.view(batch_size, seq_len, -1)


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


PADDED_LENGTHS = [2, 6, 4]


def padded_batch(config: ModelConfig, seed: int = 0) -> tuple[list[Tensor], Tensor, Tensor, Tensor]:
    """Rows of PADDED_LENGTHS random tokens, left-padded with random tokens to one length.

    Returns:
        Each row's [T_b] tokens, and the [B, T] padded tokens, [B, T] positions and [B] left
        padding of the batch.
    """
    generator = torch.Generator().manual_seed(seed)
    longest = max(PADDED_LENGTHS)
    rows = [torch.randint(0, config.vocab_size, (n,), generator=generator) for n in PADDED_LENGTHS]
    padding = [longest - len(row) for row in rows]
    filler = torch.randint(0, config.vocab_size, (len(rows), longest), generator=generator)
    input_ids = torch.stack(
        [
            torch.cat([filler[b, :n], row])
            for b, (row, n) in enumerate(zip(rows, padding, strict=True))
        ]
    )
    left_padding = torch.tensor(padding)
    positions = (torch.arange(longest) - left_padding.unsqueeze(-1)).clamp(min=0)
    return rows, input_ids, positions, left_padding


@torch.inference_mode()
def test_a_padded_row_gets_the_logits_it_gets_alone(model: LlamaModel) -> None:
    rows, input_ids, positions, left_padding = padded_batch(model.config)
    batched = all_logits(model, input_ids, positions, left_padding)
    for b, row in enumerate(rows):
        alone = all_logits(model, row.unsqueeze(0), torch.arange(len(row)).unsqueeze(0))
        torch.testing.assert_close(
            batched[b, left_padding[b] :], alone[0], rtol=CACHED_RTOL, atol=CACHED_ATOL
        )


@torch.inference_mode()
def test_padding_tokens_do_not_change_the_logits_of_the_tokens(model: LlamaModel) -> None:
    _, input_ids, positions, left_padding = padded_batch(model.config)
    _, refilled, _, _ = padded_batch(model.config, seed=1)
    is_padding = torch.arange(input_ids.shape[1]) < left_padding.unsqueeze(-1)
    changed = torch.where(is_padding, refilled, input_ids)
    assert not torch.equal(changed, input_ids)
    before = all_logits(model, input_ids, positions, left_padding)
    after = all_logits(model, changed, positions, left_padding)
    # Masked slots get a weight of exactly 0, so the token rows are unchanged to the bit.
    assert torch.equal(after[~is_padding], before[~is_padding])


@torch.inference_mode()
def test_a_padded_batch_decodes_from_a_cache_like_each_row_alone(model: LlamaModel) -> None:
    rows, input_ids, positions, left_padding = padded_batch(model.config)
    num_decode = 3
    generator = torch.Generator().manual_seed(2)
    new_tokens = torch.randint(
        0, model.config.vocab_size, (len(rows), num_decode), generator=generator
    )
    cache = ContiguousKVCache(model.config, len(rows), input_ids.shape[1] + num_decode)
    steps = [model(input_ids, positions, cache=cache, left_padding=left_padding)]  # prefill
    for step in range(num_decode):
        step_positions = positions[:, -1:] + step + 1  # [B, 1]
        steps.append(
            model(
                new_tokens[:, step : step + 1],
                step_positions,
                cache=cache,
                left_padding=left_padding,
            )
        )
    batched = torch.stack(steps, dim=1)  # [B, 1 + num_decode, V]

    for b, row in enumerate(rows):
        sequence = torch.cat([row, new_tokens[b]]).unsqueeze(0)  # [1, T_b + num_decode]
        alone = all_logits(model, sequence, torch.arange(sequence.shape[1]).unsqueeze(0))
        torch.testing.assert_close(
            batched[b], alone[0, len(row) - 1 :], rtol=CACHED_RTOL, atol=CACHED_ATOL
        )


FLAT_LENGTHS = [5, 1, 9]
POOL_LEN = 12


def flat_sequences(config: ModelConfig, seed: int = 0) -> list[Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(0, config.vocab_size, (n,), generator=generator) for n in FLAT_LENGTHS]


def flat_step(
    model: LlamaModel,
    pool: ContiguousKVPool,
    pieces: list[tuple[int, Tensor, int]],
) -> list[Tensor]:
    """Runs one flattened step and returns the logits of every new token, per sequence.

    Args:
        model: The model.
        pool: The KV pool the sequences live in.
        pieces: (pool row, new tokens, number of tokens already in the row) per sequence.
    """
    metadata = AttentionMetadata.build(
        [row for row, _, _ in pieces],
        [cached for _, _, cached in pieces],
        [len(tokens) for _, tokens, _ in pieces],
        pool.max_len,
    )
    input_ids = torch.cat([tokens for _, tokens, _ in pieces])  # [N]
    positions = torch.cat(
        [torch.arange(cached, cached + len(tokens)) for _, tokens, cached in pieces]
    )  # [N]
    logits = model.forward_flat(
        input_ids, positions, pool, metadata, torch.arange(len(input_ids))
    )  # [N, V]
    return list(logits.split([len(tokens) for _, tokens, _ in pieces]))


def alone(model: LlamaModel, tokens: Tensor) -> Tensor:
    """[T, V] logits of every token of one sequence, from the uncached forward."""
    return all_logits(model, tokens.unsqueeze(0), torch.arange(len(tokens)).unsqueeze(0))[0]


@torch.inference_mode()
def test_a_flattened_prefill_gives_each_sequence_the_logits_it_gets_alone(
    model: LlamaModel,
) -> None:
    sequences = flat_sequences(model.config)
    pool = ContiguousKVPool(model.config, len(sequences), POOL_LEN)
    pieces = [(row, tokens, 0) for row, tokens in enumerate(sequences)]
    for tokens, logits in zip(sequences, flat_step(model, pool, pieces), strict=True):
        torch.testing.assert_close(logits, alone(model, tokens), rtol=CACHED_RTOL, atol=CACHED_ATOL)


@torch.inference_mode()
def test_flattened_decoding_from_the_pool_matches_the_uncached_forward(
    model: LlamaModel,
) -> None:
    sequences = flat_sequences(model.config)
    # Rows in another order than the sequences, so a mix-up of rows shows.
    rows = [2, 0, 1]
    pool = ContiguousKVPool(model.config, len(sequences), POOL_LEN)
    num_done = [3, 1, 6]
    prefill = [
        (row, tokens[:n], 0) for row, tokens, n in zip(rows, sequences, num_done, strict=True)
    ]
    logits = [[piece] for piece in flat_step(model, pool, prefill)]
    # Then one token per sequence and step. A sequence leaves the batch once it has used all
    # of its tokens, and the others continue.
    while active := [b for b, tokens in enumerate(sequences) if num_done[b] < len(tokens)]:
        pieces = [
            (rows[b], sequences[b][num_done[b] : num_done[b] + 1], num_done[b]) for b in active
        ]
        for b, piece in zip(active, flat_step(model, pool, pieces), strict=True):
            logits[b].append(piece)
            num_done[b] += 1
    for b, tokens in enumerate(sequences):
        torch.testing.assert_close(
            torch.cat(logits[b]), alone(model, tokens), rtol=CACHED_RTOL, atol=CACHED_ATOL
        )


@torch.inference_mode()
def test_a_step_can_mix_prefill_decode_and_a_prompt_after_a_cached_prefix(
    model: LlamaModel,
) -> None:
    first, second, third = flat_sequences(model.config)
    pool = ContiguousKVPool(model.config, 3, POOL_LEN)
    before = flat_step(model, pool, [(0, first[:4], 0), (1, third[:2], 0)])
    # First decodes its last token, second starts, third continues after two cached tokens.
    mixed = flat_step(model, pool, [(0, first[4:], 4), (2, second, 0), (1, third[2:], 2)])
    torch.testing.assert_close(
        torch.cat([before[0], mixed[0]]), alone(model, first), rtol=CACHED_RTOL, atol=CACHED_ATOL
    )
    torch.testing.assert_close(mixed[1], alone(model, second), rtol=CACHED_RTOL, atol=CACHED_ATOL)
    torch.testing.assert_close(
        torch.cat([before[1], mixed[2]]), alone(model, third), rtol=CACHED_RTOL, atol=CACHED_ATOL
    )


@torch.inference_mode()
def test_a_reused_row_ignores_what_its_previous_sequence_left(model: LlamaModel) -> None:
    short, _, long = flat_sequences(model.config)
    other = flat_sequences(model.config, seed=1)[2]
    fresh = ContiguousKVPool(model.config, 2, POOL_LEN)
    used = ContiguousKVPool(model.config, 2, POOL_LEN)
    # A longer sequence held row 0 before and ended; its keys and values stay behind.
    flat_step(model, used, [(0, other, 0)])
    # A prefill, which gathers the rows, then a decode, which reads them in place up to the
    # longer sequence, past the shorter one's end into what the old sequence left.
    steps = [
        [(0, short[:-1], 0), (1, long[:-1], 0)],
        [(0, short[-1:], len(short) - 1), (1, long[-1:], len(long) - 1)],
    ]
    assert len(other) > len(short)
    for pieces in steps:
        for expected, actual in zip(
            flat_step(model, fresh, pieces), flat_step(model, used, pieces), strict=True
        ):
            # Stale positions get a weight of exactly 0, so the logits are unchanged to the
            # bit.
            assert torch.equal(actual, expected)


@torch.inference_mode()
def test_flattened_logits_are_computed_for_the_requested_tokens_only(model: LlamaModel) -> None:
    sequences = flat_sequences(model.config)
    pool = ContiguousKVPool(model.config, len(sequences), POOL_LEN)
    metadata = AttentionMetadata.build([0, 1, 2], [0, 0, 0], FLAT_LENGTHS, POOL_LEN)
    input_ids = torch.cat(sequences)
    positions = torch.cat([torch.arange(n) for n in FLAT_LENGTHS])
    last_tokens = torch.tensor(FLAT_LENGTHS).cumsum(0) - 1  # [3]
    logits = model.forward_flat(input_ids, positions, pool, metadata, last_tokens)
    assert logits.shape == (3, model.config.vocab_size)
    for b, tokens in enumerate(sequences):
        torch.testing.assert_close(
            logits[b], alone(model, tokens)[-1], rtol=CACHED_RTOL, atol=CACHED_ATOL
        )


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
