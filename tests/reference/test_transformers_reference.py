from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import Tensor
from transformers import LlamaForCausalLM

from kiyal.model.llama import LlamaModel
from kiyal.model.loader import load_model

CPU = torch.device("cpu")

# Overrides of the tiny config, covering features the SmolLM2 reference does not use.
TINY_VARIANTS: dict[str, dict[str, object]] = {
    "gqa-tied": {},
    "mha": {"num_key_value_heads": 4},
    "mqa": {"num_key_value_heads": 1},
    "attention-bias": {"attention_bias": True},
    "untied": {"tie_word_embeddings": False},
    "head-dim-32": {"head_dim": 32},
    # With head_dim 16 and a 32-token original context, the 8 frequency pairs fall in all three
    # bands: one kept, one interpolated and six slowed down.
    "llama3-rope": {
        "rope_parameters": {
            "rope_type": "llama3",
            "rope_theta": 10000.0,
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 32,
        }
    },
}


def load_reference(checkpoint: Path) -> LlamaForCausalLM:
    # Eager attention computes softmax and masking explicitly, an independent path from the
    # SDPA call under test.
    return LlamaForCausalLM.from_pretrained(
        checkpoint, dtype=torch.float32, attn_implementation="eager"
    ).eval()


def all_logits(model: LlamaModel, input_ids: Tensor) -> Tensor:
    batch_size, seq_len = input_ids.shape
    positions = torch.arange(seq_len).expand(batch_size, seq_len)
    indices = torch.arange(batch_size * seq_len)
    return model(input_ids, positions, indices).view(batch_size, seq_len, -1)  # [B, T, V]


def greedy_continuation(model: LlamaModel, prompt: Tensor, steps: int) -> Tensor:
    """Extends a [1, T] prompt by `steps` argmax tokens, recomputing everything each step."""
    ids = prompt
    for _ in range(steps):
        positions = torch.arange(ids.shape[1]).unsqueeze(0)  # [1, T]
        next_id = model(ids, positions).argmax(dim=-1, keepdim=True)  # [1, 1]
        ids = torch.cat([ids, next_id], dim=1)
    return ids


def assert_reference_picks_same_tokens(
    reference: LlamaForCausalLM, ids: Tensor, prompt_len: int
) -> None:
    # With the generated sequence as input, the reference's argmax at each position must be the
    # token generated next. If it is everywhere, the reference's own greedy decoding produces the
    # same sequence.
    logits = reference(ids).logits[0]  # [T, V]
    predicted = logits[prompt_len - 1 : -1].argmax(dim=-1)  # [steps]
    generated = ids[0, prompt_len:]  # [steps]
    mismatches = torch.nonzero(predicted != generated).flatten().tolist()
    assert not mismatches, f"first differing step: {mismatches[0]} of {len(generated)}"


@pytest.mark.parametrize("overrides", TINY_VARIANTS.values(), ids=list(TINY_VARIANTS))
@torch.inference_mode()
def test_tiny_model_matches_transformers(
    make_tiny_checkpoint: Callable[..., Path], overrides: dict[str, object]
) -> None:
    checkpoint = make_tiny_checkpoint(**overrides)
    model = load_model(checkpoint, device=CPU, dtype=torch.float32)
    reference = load_reference(checkpoint)

    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 40), generator=generator)
    torch.testing.assert_close(
        all_logits(model, input_ids), reference(input_ids).logits, atol=1e-5, rtol=0
    )

    prompt = input_ids[:1, :8]
    ids = greedy_continuation(model, prompt, steps=32)
    assert_reference_picks_same_tokens(reference, ids, prompt_len=prompt.shape[1])
