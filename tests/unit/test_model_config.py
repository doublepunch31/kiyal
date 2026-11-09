import json
from pathlib import Path

import pytest

from kiyal.model.config import ConfigError, Llama3RopeScaling, ModelConfig

# config.json of HuggingFaceTB/SmolLM2-135M-Instruct at revision 12fd25f, without the fields that
# do not describe the architecture.
SMOLLM2_135M: dict[str, object] = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 576,
    "intermediate_size": 1536,
    "max_position_embeddings": 8192,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 9,
    "num_hidden_layers": 30,
    "num_key_value_heads": 3,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_interleaved": False,
    "rope_scaling": None,
    "rope_theta": 100000,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "vocab_size": 49152,
}

# Values from the Llama 3.1 release configs.
LLAMA3_SCALING: dict[str, object] = {
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3",
}


def with_fields(**fields: object) -> dict[str, object]:
    return {**SMOLLM2_135M, **fields}


def without(*keys: str) -> dict[str, object]:
    return {k: v for k, v in SMOLLM2_135M.items() if k not in keys}


def test_parses_smollm2_config() -> None:
    assert ModelConfig.from_dict(SMOLLM2_135M) == ModelConfig(
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        head_dim=64,
        rms_norm_eps=1e-5,
        rope_theta=100000.0,
        rope_scaling=None,
        vocab_size=49152,
        max_position_embeddings=8192,
        tie_word_embeddings=True,
        attention_bias=False,
        bos_token_id=1,
        eos_token_id=(2,),
    )


def test_reads_config_file(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(SMOLLM2_135M))
    assert ModelConfig.from_file(path) == ModelConfig.from_dict(SMOLLM2_135M)


def test_file_must_hold_an_object(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("[1, 2]")
    with pytest.raises(ConfigError, match="JSON object"):
        ModelConfig.from_file(path)


def test_explicit_head_dim_overrides_derived_value() -> None:
    assert ModelConfig.from_dict(with_fields(head_dim=32)).head_dim == 32


def test_missing_optional_fields_take_llama_defaults() -> None:
    config = ModelConfig.from_dict(
        without(
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_theta",
            "max_position_embeddings",
            "tie_word_embeddings",
            "attention_bias",
            "hidden_act",
            "mlp_bias",
            "bos_token_id",
            "eos_token_id",
        )
    )
    assert config.num_key_value_heads == config.num_attention_heads
    assert config.rms_norm_eps == 1e-6
    assert config.rope_theta == 10000.0
    assert config.max_position_embeddings == 2048
    assert not config.tie_word_embeddings
    assert not config.attention_bias
    assert config.bos_token_id is None
    assert config.eos_token_id == ()


@pytest.mark.parametrize(
    "key", ["model_type", "hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size"]
)
def test_missing_required_field_raises(key: str) -> None:
    with pytest.raises(ConfigError, match=f"missing '{key}'"):
        ModelConfig.from_dict(without(key))


def test_eos_token_id_list_becomes_tuple() -> None:
    assert ModelConfig.from_dict(with_fields(eos_token_id=[128001, 128009])).eos_token_id == (
        128001,
        128009,
    )


@pytest.mark.parametrize("value", ["2", [2, "3"], -1, True])
def test_invalid_eos_token_id_raises(value: object) -> None:
    with pytest.raises(ConfigError, match="eos_token_id"):
        ModelConfig.from_dict(with_fields(eos_token_id=value))


@pytest.mark.parametrize(
    ("key", "value"),
    [("hidden_size", "576"), ("hidden_size", 0), ("num_hidden_layers", True), ("rms_norm_eps", -1)],
)
def test_invalid_field_type_raises(key: str, value: object) -> None:
    with pytest.raises(ConfigError, match=f"'{key}' must be"):
        ModelConfig.from_dict(with_fields(**{key: value}))


def test_legacy_llama3_rope_scaling() -> None:
    config = ModelConfig.from_dict(with_fields(rope_theta=500000.0, rope_scaling=LLAMA3_SCALING))
    assert config.rope_theta == 500000.0
    assert config.rope_scaling == Llama3RopeScaling(
        factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192
    )


def test_rope_scaling_with_legacy_type_key() -> None:
    scaling = {k: v for k, v in LLAMA3_SCALING.items() if k != "rope_type"} | {"type": "llama3"}
    config = ModelConfig.from_dict(with_fields(rope_scaling=scaling))
    assert config.rope_scaling is not None


def test_rope_parameters_default() -> None:
    raw = without("rope_theta", "rope_scaling") | {
        "rope_parameters": {"rope_type": "default", "rope_theta": 50000.0}
    }
    config = ModelConfig.from_dict(raw)
    assert config.rope_theta == 50000.0
    assert config.rope_scaling is None


def test_rope_parameters_llama3() -> None:
    config = ModelConfig.from_dict(
        with_fields(rope_parameters={**LLAMA3_SCALING, "rope_theta": 500000.0})
    )
    assert config.rope_theta == 500000.0
    assert config.rope_scaling is not None
    assert config.rope_scaling.factor == 8.0


@pytest.mark.parametrize("rope_type", ["linear", "dynamic", "yarn"])
def test_unsupported_rope_type_raises(rope_type: str) -> None:
    with pytest.raises(ConfigError, match=f"unsupported RoPE type '{rope_type}'"):
        ModelConfig.from_dict(with_fields(rope_scaling={"rope_type": rope_type, "factor": 2.0}))


def test_llama3_scaling_needs_ordered_frequency_factors() -> None:
    scaling = {**LLAMA3_SCALING, "low_freq_factor": 4.0, "high_freq_factor": 1.0}
    with pytest.raises(ConfigError, match="high_freq_factor"):
        ModelConfig.from_dict(with_fields(rope_scaling=scaling))


def test_unsupported_model_type_raises() -> None:
    with pytest.raises(ConfigError, match="unsupported model_type 'qwen2'"):
        ModelConfig.from_dict(with_fields(model_type="qwen2"))


def test_heads_must_divide_into_kv_groups() -> None:
    with pytest.raises(ConfigError, match="multiple of num_key_value_heads"):
        ModelConfig.from_dict(with_fields(num_key_value_heads=4))


def test_odd_head_dim_raises() -> None:
    with pytest.raises(ConfigError, match="head_dim must be even"):
        ModelConfig.from_dict(with_fields(head_dim=63))


def test_unsupported_activation_raises() -> None:
    with pytest.raises(ConfigError, match="hidden_act 'gelu'"):
        ModelConfig.from_dict(with_fields(hidden_act="gelu"))


def test_mlp_bias_raises() -> None:
    with pytest.raises(ConfigError, match="mlp_bias"):
        ModelConfig.from_dict(with_fields(mlp_bias=True))
