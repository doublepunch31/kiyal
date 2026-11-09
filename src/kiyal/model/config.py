import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self, cast

SUPPORTED_MODEL_TYPES: Final = ("llama",)
SUPPORTED_ROPE_TYPES: Final = ("default", "llama3")

# transformers' LlamaConfig defaults, used when a field is absent so that a config means the same
# model here as in the reference.
DEFAULT_ROPE_THETA: Final = 10000.0
DEFAULT_RMS_NORM_EPS: Final = 1e-6
DEFAULT_MAX_POSITION_EMBEDDINGS: Final = 2048


class ConfigError(ValueError):
    """Raised when a model config is malformed or uses an unsupported feature."""


@dataclass(frozen=True, slots=True)
class Llama3RopeScaling:
    """RoPE frequency scaling from Llama 3.1, which stretches a trained context length.

    Rotations with a wavelength shorter than `original_max_position_embeddings /
    high_freq_factor` are kept, those longer than `original_max_position_embeddings /
    low_freq_factor` are slowed down by `factor`, and the band in between is interpolated.
    """

    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture of a Llama-family model, parsed from a Hugging Face config.json."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Llama3RopeScaling | None
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    bos_token_id: int | None
    eos_token_id: tuple[int, ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> Self:
        """Builds a config from the contents of a Hugging Face config.json.

        Sizes are required. Other missing fields take transformers' LlamaConfig defaults.
        RoPE settings are read from `rope_parameters` (written by transformers 5) or from
        `rope_theta` and `rope_scaling` (older checkpoints).

        Raises:
            ConfigError: If a field is missing or has the wrong type, or the config describes
                an architecture or feature that is not supported.
        """
        model_type = _get_str(raw, "model_type")
        if model_type not in SUPPORTED_MODEL_TYPES:
            supported = ", ".join(SUPPORTED_MODEL_TYPES)
            raise ConfigError(f"unsupported model_type {model_type!r}; supported: {supported}")
        hidden_act = _get_str(raw, "hidden_act", default="silu")
        if hidden_act != "silu":
            raise ConfigError(f"unsupported hidden_act {hidden_act!r}; only 'silu' is supported")
        if _get_bool(raw, "mlp_bias", default=False):
            raise ConfigError("mlp_bias is not supported")

        hidden_size = _get_int(raw, "hidden_size")
        num_attention_heads = _get_int(raw, "num_attention_heads")
        num_key_value_heads = _get_int(raw, "num_key_value_heads", default=num_attention_heads)
        if num_attention_heads % num_key_value_heads != 0:
            raise ConfigError(
                f"num_attention_heads ({num_attention_heads}) must be a multiple of "
                f"num_key_value_heads ({num_key_value_heads})"
            )
        head_dim = _get_int(raw, "head_dim", default=hidden_size // num_attention_heads)
        if head_dim % 2 != 0:
            # Rotary embeddings rotate the two halves of each head against each other.
            raise ConfigError(f"head_dim must be even, got {head_dim}")
        rope_theta, rope_scaling = _parse_rope(raw)

        return cls(
            hidden_size=hidden_size,
            intermediate_size=_get_int(raw, "intermediate_size"),
            num_hidden_layers=_get_int(raw, "num_hidden_layers"),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=_get_float(raw, "rms_norm_eps", default=DEFAULT_RMS_NORM_EPS),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            vocab_size=_get_int(raw, "vocab_size"),
            max_position_embeddings=_get_int(
                raw, "max_position_embeddings", default=DEFAULT_MAX_POSITION_EMBEDDINGS
            ),
            tie_word_embeddings=_get_bool(raw, "tie_word_embeddings", default=False),
            attention_bias=_get_bool(raw, "attention_bias", default=False),
            bos_token_id=_get_optional_token_id(raw, "bos_token_id"),
            eos_token_id=_get_token_ids(raw, "eos_token_id"),
        )

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Reads a Hugging Face config.json.

        Raises:
            ConfigError: If the file is not a JSON object or fails `from_dict` validation.
        """
        raw: object = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} must contain a JSON object")
        return cls.from_dict(cast(dict[str, object], raw))


def _parse_rope(raw: Mapping[str, object]) -> tuple[float, Llama3RopeScaling | None]:
    if raw.get("rope_parameters") is not None:
        params = _get_mapping(raw, "rope_parameters")
        theta = _get_float(params, "rope_theta", default=DEFAULT_ROPE_THETA)
        rope_type = _get_str(params, "rope_type", default="default")
    else:
        theta = _get_float(raw, "rope_theta", default=DEFAULT_ROPE_THETA)
        if raw.get("rope_scaling") is None:
            return theta, None
        params = _get_mapping(raw, "rope_scaling")
        # Configs written before transformers 4.43 call this field "type".
        legacy_type = _get_str(params, "type", default="default")
        rope_type = _get_str(params, "rope_type", default=legacy_type)

    if rope_type == "default":
        return theta, None
    if rope_type == "llama3":
        scaling = Llama3RopeScaling(
            factor=_get_float(params, "factor"),
            low_freq_factor=_get_float(params, "low_freq_factor"),
            high_freq_factor=_get_float(params, "high_freq_factor"),
            original_max_position_embeddings=_get_int(params, "original_max_position_embeddings"),
        )
        if scaling.high_freq_factor <= scaling.low_freq_factor:
            raise ConfigError("llama3 RoPE scaling needs high_freq_factor > low_freq_factor")
        return theta, scaling
    raise ConfigError(
        f"unsupported RoPE type {rope_type!r}; supported: {', '.join(SUPPORTED_ROPE_TYPES)}"
    )


def _get(raw: Mapping[str, object], key: str, default: object) -> object:
    value = raw.get(key)
    if value is None:
        if default is None:
            raise ConfigError(f"config is missing {key!r}")
        return default
    return value


def _get_int(raw: Mapping[str, object], key: str, default: int | None = None) -> int:
    value = _get(raw, key, default)
    # bool is a subclass of int, so `true` would otherwise pass as 1.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{key!r} must be a positive integer, got {value!r}")
    return value


def _get_float(raw: Mapping[str, object], key: str, default: float | None = None) -> float:
    value = _get(raw, key, default)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ConfigError(f"{key!r} must be a positive number, got {value!r}")
    return float(value)


def _get_bool(raw: Mapping[str, object], key: str, default: bool) -> bool:
    value = _get(raw, key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key!r} must be true or false, got {value!r}")
    return value


def _get_str(raw: Mapping[str, object], key: str, default: str | None = None) -> str:
    value = _get(raw, key, default)
    if not isinstance(value, str):
        raise ConfigError(f"{key!r} must be a string, got {value!r}")
    return value


def _get_mapping(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key!r} must be an object, got {value!r}")
    return cast(dict[str, object], value)


def _is_token_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _get_optional_token_id(raw: Mapping[str, object], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if not _is_token_id(value):
        raise ConfigError(f"{key!r} must be a token id, got {value!r}")
    return cast(int, value)


def _get_token_ids(raw: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = raw.get(key)
    if value is None:
        return ()
    # Some models stop on several tokens, so eos_token_id may be a list.
    values = cast(list[object], value) if isinstance(value, list) else [value]
    if not all(_is_token_id(v) for v in values):
        raise ConfigError(f"{key!r} must be a token id or a list of token ids, got {value!r}")
    return tuple(cast(list[int], values))
