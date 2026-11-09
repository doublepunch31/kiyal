import dataclasses

import pytest
import torch
from torch import Tensor

from kiyal.model.config import ModelConfig
from kiyal.model.llama import LlamaModel

BATCH_SIZE = 2
SEQ_LEN = 6


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
