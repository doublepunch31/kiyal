import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kiyal.model.config import ModelConfig

# A tiny Llama in Hugging Face config.json form: 2 layers, 4 query heads sharing 2 KV heads.
TINY_CONFIG: dict[str, object] = {
    "model_type": "llama",
    "vocab_size": 256,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-5,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    "tie_word_embeddings": True,
    "attention_bias": False,
    "bos_token_id": 1,
    "eos_token_id": 2,
}


@pytest.fixture
def tiny_config() -> ModelConfig:
    return ModelConfig.from_dict(TINY_CONFIG)


def write_tiny_checkpoint(
    path: Path, config: Mapping[str, object], max_shard_size: str | None, seed: int = 0
) -> Path:
    model = LlamaForCausalLM(LlamaConfig.from_dict(dict(config)))
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            noise = 0.1 * torch.randn(param.shape, generator=generator)
            # Norm weights near 1 and nonzero biases, unlike the transformers defaults of exactly
            # 1 and 0, so a loader that skips either one changes the outputs.
            param.copy_(noise + 1.0 if name.endswith("norm.weight") else noise)
    if max_shard_size is None:
        model.save_pretrained(path)
    else:
        model.save_pretrained(path, max_shard_size=max_shard_size)
    return path


@pytest.fixture(scope="session")
def make_tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., Path]:
    """Returns a function that writes a tiny random checkpoint with transformers.

    Keyword arguments override fields of TINY_CONFIG; `max_shard_size` splits the weights into
    several files. Each distinct checkpoint is written once per session and shared, so tests
    must copy it before modifying it.
    """
    written: dict[str, Path] = {}

    def make(*, max_shard_size: str | None = None, **overrides: object) -> Path:
        key = json.dumps([overrides, max_shard_size], sort_keys=True)
        if key not in written:
            config = {**TINY_CONFIG, **overrides}
            written[key] = write_tiny_checkpoint(
                tmp_path_factory.mktemp("tiny"), config, max_shard_size
            )
        return written[key]

    return make


@pytest.fixture(scope="session")
def tiny_checkpoint(make_tiny_checkpoint: Callable[..., Path]) -> Path:
    """Directory of a tiny random checkpoint written by transformers."""
    return make_tiny_checkpoint()
