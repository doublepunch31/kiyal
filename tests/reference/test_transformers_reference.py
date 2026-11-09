from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import Tensor
from transformers import LlamaForCausalLM, PreTrainedTokenizerBase

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

# Plain text, or chat messages rendered with the model's chat template.
SMOLLM2_PROMPTS: dict[str, str | list[dict[str, str]]] = {
    "short": "The capital of France is",
    "code": "def fibonacci(n):\n",
    "paragraph": (
        "A paged KV cache stores the keys and values of each sequence in fixed-size blocks "
        "instead of one contiguous buffer. A block table maps each sequence's logical blocks "
        "to physical blocks in a shared pool, so memory is allocated as sequences grow and "
        "freed blocks can be reused by other requests. The main benefit is"
    ),
    "chat": [{"role": "user", "content": "What is the boiling point of water at sea level?"}],
}
SMOLLM2_GREEDY_STEPS = 64


def load_reference(checkpoint: Path, attn_implementation: str) -> LlamaForCausalLM:
    return LlamaForCausalLM.from_pretrained(
        checkpoint, dtype=torch.float32, attn_implementation=attn_implementation
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
    # Eager attention computes softmax and masking explicitly, independently of the SDPA call
    # under test.
    reference = load_reference(checkpoint, attn_implementation="eager")

    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 40), generator=generator)
    torch.testing.assert_close(
        all_logits(model, input_ids), reference(input_ids).logits, atol=1e-5, rtol=0
    )

    prompt = input_ids[:1, :8]
    ids = greedy_continuation(model, prompt, steps=32)
    assert_reference_picks_same_tokens(reference, ids, prompt_len=prompt.shape[1])


def encode(tokenizer: PreTrainedTokenizerBase, prompt: str | list[dict[str, str]]) -> Tensor:
    if isinstance(prompt, str):
        return tokenizer(prompt, return_tensors="pt")["input_ids"]  # [1, T]
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    assert isinstance(text, str)
    # The rendered template already contains its special tokens.
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]


@pytest.fixture(scope="module")
def smollm2_model(smollm2_checkpoint: Path) -> LlamaModel:
    return load_model(smollm2_checkpoint, device=CPU, dtype=torch.float32)


@pytest.fixture(scope="module")
def smollm2_reference(smollm2_checkpoint: Path) -> LlamaForCausalLM:
    # SmolLM2's hidden states reach about 24,000, where fp32 values are spaced about 2e-3 apart.
    # At that scale transformers' own eager and SDPA paths differ by up to 2e-4 in the logits,
    # so the 1e-4 bound is checked against its default SDPA path. The tiny-model test above
    # covers the eager comparison.
    return load_reference(smollm2_checkpoint, attn_implementation="sdpa")


@pytest.mark.slow
@pytest.mark.parametrize("prompt", SMOLLM2_PROMPTS.values(), ids=list(SMOLLM2_PROMPTS))
@torch.inference_mode()
def test_smollm2_matches_transformers(
    smollm2_model: LlamaModel,
    smollm2_reference: LlamaForCausalLM,
    smollm2_tokenizer: PreTrainedTokenizerBase,
    prompt: str | list[dict[str, str]],
) -> None:
    prompt_ids = encode(smollm2_tokenizer, prompt)
    ids = greedy_continuation(smollm2_model, prompt_ids, steps=SMOLLM2_GREEDY_STEPS)
    # Logits are compared over the prompt and the generated tokens.
    torch.testing.assert_close(
        all_logits(smollm2_model, ids), smollm2_reference(ids).logits, atol=1e-4, rtol=0
    )
    assert_reference_picks_same_tokens(smollm2_reference, ids, prompt_len=prompt_ids.shape[1])
