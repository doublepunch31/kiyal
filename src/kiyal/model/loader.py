import json
import logging
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final, Protocol, cast

import torch

# huggingface_hub annotates one of snapshot_download's parameters as a bare `dict`, which strict
# mode reports as partially unknown. The arguments passed here are all typed.
from huggingface_hub import snapshot_download  # pyright: ignore[reportUnknownVariableType]
from safetensors import safe_open
from torch import Tensor

from kiyal.model.config import ModelConfig
from kiyal.model.llama import LlamaModel

logger = logging.getLogger(__name__)

CONFIG_FILE: Final = "config.json"
SINGLE_WEIGHTS_FILE: Final = "model.safetensors"
WEIGHTS_INDEX_FILE: Final = "model.safetensors.index.json"
# Top-level files only, so that copies of the weights in other formats or subfolders are skipped.
HUB_FILE_PATTERNS: Final = [CONFIG_FILE, "model*.safetensors", WEIGHTS_INDEX_FILE]

# Hugging Face tensor names and the parameters they fill.
_MODEL_NAMES: Final = {
    "model.embed_tokens.weight": "embed_tokens.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "lm_head.weight",
}
# Per-layer names, prefixed with "model.layers.{i}." in checkpoints and "layers.{i}." here.
_LAYER_NAMES: Final = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn.q_proj.weight",
    "self_attn.k_proj.weight": "attn.k_proj.weight",
    "self_attn.v_proj.weight": "attn.v_proj.weight",
    "self_attn.o_proj.weight": "attn.o_proj.weight",
    "self_attn.q_proj.bias": "attn.q_proj.bias",
    "self_attn.k_proj.bias": "attn.k_proj.bias",
    "self_attn.v_proj.bias": "attn.v_proj.bias",
    "self_attn.o_proj.bias": "attn.o_proj.bias",
    "post_attention_layernorm.weight": "mlp_norm.weight",
    "mlp.gate_proj.weight": "mlp.gate_proj.weight",
    "mlp.up_proj.weight": "mlp.up_proj.weight",
    "mlp.down_proj.weight": "mlp.down_proj.weight",
}
_MAX_LISTED_NAMES: Final = 8


class CheckpointError(Exception):
    """Raised when checkpoint files are missing or do not match the model."""


class _TensorSlice(Protocol):
    def get_shape(self) -> list[int]: ...


class _TensorFile(Protocol):
    def keys(self) -> list[str]: ...
    def get_slice(self, name: str) -> _TensorSlice: ...
    def get_tensor(self, name: str) -> Tensor: ...


def checkpoint_name_map(num_layers: int) -> dict[str, str]:
    """Maps each tensor name a Hugging Face Llama checkpoint can hold to a parameter name.

    The map covers optional tensors such as biases and an untied LM head; which of them a
    model uses depends on its config.
    """
    names = dict(_MODEL_NAMES)
    for i in range(num_layers):
        for checkpoint_name, param_name in _LAYER_NAMES.items():
            names[f"model.layers.{i}.{checkpoint_name}"] = f"layers.{i}.{param_name}"
    return names


def load_model(
    model: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    revision: str | None = None,
) -> LlamaModel:
    """Builds a model from a Hugging Face checkpoint.

    Args:
        model: A local checkpoint directory, or a Hub repo id such as
            "HuggingFaceTB/SmolLM2-135M-Instruct".
        device: Device to hold the weights.
        dtype: Parameter dtype. Checkpoint tensors are converted on load.
        revision: Branch, tag or commit to download. Only used for Hub repo ids.

    Raises:
        CheckpointError: If the checkpoint is missing files or does not match its config.
        ConfigError: If config.json is invalid or describes an unsupported model.
    """
    model_dir = resolve_model_dir(model, revision=revision)
    config_path = model_dir / CONFIG_FILE
    if not config_path.is_file():
        raise CheckpointError(f"no {CONFIG_FILE} in {model_dir}")
    llama = LlamaModel(ModelConfig.from_file(config_path), device=device, dtype=dtype)
    llama.requires_grad_(False)
    load_weights(llama, model_dir)
    return llama.eval()


def resolve_model_dir(model: str | Path, *, revision: str | None = None) -> Path:
    """Returns the local directory of a checkpoint, downloading it from the Hub if needed.

    An existing directory is used as is. A Path, or a string that looks like a filesystem
    path, must exist. Anything else is treated as a Hub repo id.

    Raises:
        CheckpointError: If a local path was given and it is not a directory.
    """
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    if isinstance(model, Path) or path.is_absolute() or str(model).startswith("."):
        raise CheckpointError(f"model directory {path} does not exist")
    return Path(snapshot_download(str(model), revision=revision, allow_patterns=HUB_FILE_PATTERNS))


def load_weights(model: LlamaModel, model_dir: Path) -> None:
    """Copies checkpoint tensors into a model's parameters.

    Loading is strict: every checkpoint tensor must fill a parameter of the same shape and
    every parameter must be filled. All names and shapes are checked before any data is read.
    Tensors are read one at a time onto the parameters' device, so peak memory is the model
    plus its largest tensor.

    Raises:
        CheckpointError: If weight files are missing or the tensors do not match the model.
    """
    files = _weight_files(model_dir)
    params = dict(model.named_parameters())
    name_map = checkpoint_name_map(model.config.num_hidden_layers)
    device = model.embed_tokens.weight.device

    names_in_file: dict[Path, list[str]] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for path in files:
        with _open_weights(path, device) as weights:
            names_in_file[path] = weights.keys()
            for name in names_in_file[path]:
                if name in shapes:
                    raise CheckpointError(f"tensor {name} appears in more than one file")
                shapes[name] = tuple(weights.get_slice(name).get_shape())
    _check_tensors_match(shapes, params, name_map, model.config, model_dir)

    with torch.no_grad():
        for path in files:
            with _open_weights(path, device) as weights:
                for name in names_in_file[path]:
                    # copy_ converts to the parameter's dtype.
                    params[name_map[name]].copy_(weights.get_tensor(name))
    logger.info("loaded %d tensors from %d file(s) in %s", len(shapes), len(files), model_dir)


def _weight_files(model_dir: Path) -> list[Path]:
    index_path = model_dir / WEIGHTS_INDEX_FILE
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        try:
            # weight_map maps each tensor name to the file that holds it.
            weight_map = cast(dict[str, object], index["weight_map"])
            file_names = sorted({str(name) for name in weight_map.values()})
        except (KeyError, TypeError, AttributeError) as e:
            raise CheckpointError(f"{index_path} has no weight_map") from e
        files = [model_dir / name for name in file_names]
        absent = [f.name for f in files if not f.is_file()]
        if absent:
            raise CheckpointError(f"{index_path} lists missing files: {_join(absent)}")
        return files
    single_path = model_dir / SINGLE_WEIGHTS_FILE
    if single_path.is_file():
        return [single_path]
    raise CheckpointError(f"no {SINGLE_WEIGHTS_FILE} or {WEIGHTS_INDEX_FILE} in {model_dir}")


@contextmanager
def _open_weights(path: Path, device: torch.device) -> Generator[_TensorFile]:
    # safetensors ships stubs without annotations; the Protocol gives its reader types.
    handle = safe_open(str(path), framework="pt", device=str(device))
    with handle:
        yield cast(_TensorFile, handle)


def _check_tensors_match(
    shapes: Mapping[str, tuple[int, ...]],
    params: Mapping[str, Tensor],
    name_map: Mapping[str, str],
    config: ModelConfig,
    model_dir: Path,
) -> None:
    checkpoint_name_of = {param: name for name, param in name_map.items() if param in params}
    # A parameter without a checkpoint name is reported under its own name.
    missing = [
        checkpoint_name_of.get(p, p) for p in params if checkpoint_name_of.get(p) not in shapes
    ]
    unexpected = [name for name in shapes if name_map.get(name) not in params]
    wrong_shapes = [
        f"{name} {list(shape)} vs {list(params[name_map[name]].shape)}"
        for name, shape in shapes.items()
        if name_map.get(name) in params and shape != tuple(params[name_map[name]].shape)
    ]
    problems: list[str] = []
    if missing:
        problems.append(f"missing tensors: {_join(missing)}")
    if unexpected:
        problems.append(f"unexpected tensors: {_join(unexpected)}")
        if "lm_head.weight" in unexpected and config.tie_word_embeddings:
            problems.append("lm_head.weight is unused because tie_word_embeddings is true")
    if wrong_shapes:
        problems.append(f"shape mismatches (checkpoint vs model): {_join(wrong_shapes)}")
    if problems:
        raise CheckpointError(
            f"checkpoint in {model_dir} does not match its config; " + "; ".join(problems)
        )


def _join(names: Sequence[str]) -> str:
    listed = ", ".join(sorted(names)[:_MAX_LISTED_NAMES])
    extra = len(names) - _MAX_LISTED_NAMES
    return f"{listed} and {extra} more" if extra > 0 else listed
