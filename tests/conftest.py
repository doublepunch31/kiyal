import pytest

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
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
    "attention_bias": False,
    "bos_token_id": 1,
    "eos_token_id": 2,
}


@pytest.fixture
def tiny_config() -> ModelConfig:
    return ModelConfig.from_dict(TINY_CONFIG)
