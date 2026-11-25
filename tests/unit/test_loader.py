import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from kiyal.model import loader
from kiyal.model.loader import CheckpointError, checkpoint_name_map, load_model
from kiyal.quant.int8 import QuantLinear, quantize_per_channel

CPU = torch.device("cpu")


def checkpoint_tensors(checkpoint: Path) -> dict[str, Tensor]:
    tensors: dict[str, Tensor] = {}
    for path in sorted(checkpoint.glob("*.safetensors")):
        tensors |= load_file(path)
    return tensors


def write_variant(source: Path, target: Path, tensors: dict[str, Tensor]) -> Path:
    # A copy of the source checkpoint's config with different weights.
    target.mkdir()
    shutil.copy(source / "config.json", target / "config.json")
    save_file(tensors, target / "model.safetensors", metadata={"format": "pt"})
    return target


def test_fills_every_parameter_from_the_checkpoint(tiny_checkpoint: Path) -> None:
    model = load_model(tiny_checkpoint, device=CPU, dtype=torch.float32)
    tensors = checkpoint_tensors(tiny_checkpoint)
    params = dict(model.named_parameters())
    name_map = checkpoint_name_map(model.config.num_hidden_layers)
    assert len(params) == len(tensors)
    for name, tensor in tensors.items():
        torch.testing.assert_close(params[name_map[name]], tensor, rtol=0, atol=0)


def test_names_map_to_the_matching_modules(tiny_checkpoint: Path) -> None:
    model = load_model(tiny_checkpoint, device=CPU, dtype=torch.float32)
    params = dict(model.named_parameters())
    tensors = checkpoint_tensors(tiny_checkpoint)
    pairs = {
        "layers.1.attn_norm.weight": "model.layers.1.input_layernorm.weight",
        "layers.1.mlp_norm.weight": "model.layers.1.post_attention_layernorm.weight",
        "layers.0.attn.k_proj.weight": "model.layers.0.self_attn.k_proj.weight",
        "layers.0.mlp.down_proj.weight": "model.layers.0.mlp.down_proj.weight",
        "norm.weight": "model.norm.weight",
    }
    for param_name, checkpoint_name in pairs.items():
        assert torch.equal(params[param_name], tensors[checkpoint_name]), param_name


def test_tied_lm_head_uses_the_embedding(tiny_checkpoint: Path) -> None:
    model = load_model(tiny_checkpoint, device=CPU, dtype=torch.float32)
    assert "lm_head.weight" not in checkpoint_tensors(tiny_checkpoint)
    assert model.lm_head.weight is model.embed_tokens.weight


def test_untied_lm_head_is_loaded(make_tiny_checkpoint: Callable[..., Path]) -> None:
    checkpoint = make_tiny_checkpoint(tie_word_embeddings=False)
    model = load_model(checkpoint, device=CPU, dtype=torch.float32)
    tensors = checkpoint_tensors(checkpoint)
    assert torch.equal(model.lm_head.weight, tensors["lm_head.weight"])
    assert not torch.equal(model.lm_head.weight, model.embed_tokens.weight)


def test_attention_biases_are_loaded(make_tiny_checkpoint: Callable[..., Path]) -> None:
    checkpoint = make_tiny_checkpoint(attention_bias=True)
    model = load_model(checkpoint, device=CPU, dtype=torch.float32)
    tensors = checkpoint_tensors(checkpoint)
    params = dict(model.named_parameters())
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        expected = tensors[f"model.layers.0.self_attn.{proj}.bias"]
        assert torch.equal(params[f"layers.0.attn.{proj}.bias"], expected), proj


def test_sharded_checkpoint_matches_single_file(
    tiny_checkpoint: Path, make_tiny_checkpoint: Callable[..., Path]
) -> None:
    sharded = make_tiny_checkpoint(max_shard_size="40KB")
    assert (sharded / "model.safetensors.index.json").is_file()
    assert len(list(sharded.glob("model-*.safetensors"))) > 1
    single_state = load_model(tiny_checkpoint, device=CPU, dtype=torch.float32).state_dict()
    sharded_state = load_model(sharded, device=CPU, dtype=torch.float32).state_dict()
    assert single_state.keys() == sharded_state.keys()
    for name, tensor in single_state.items():
        assert torch.equal(sharded_state[name], tensor), name


def test_converts_to_the_requested_dtype(tiny_checkpoint: Path) -> None:
    model = load_model(tiny_checkpoint, device=CPU, dtype=torch.float16)
    assert all(p.dtype == torch.float16 for p in model.parameters())
    expected = checkpoint_tensors(tiny_checkpoint)["model.norm.weight"].half()
    assert torch.equal(model.norm.weight, expected)


def test_model_is_ready_for_inference(tiny_checkpoint: Path) -> None:
    model = load_model(tiny_checkpoint, device=CPU, dtype=torch.float32)
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())


PROJECTIONS = [
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def test_int8_quantizes_the_projections_of_every_layer(tiny_checkpoint: Path) -> None:
    model = load_model(tiny_checkpoint, device=CPU, dtype=torch.float32, quantization="int8")
    tensors = checkpoint_tensors(tiny_checkpoint)
    checkpoint_name_of = {
        param: name for name, param in checkpoint_name_map(model.config.num_hidden_layers).items()
    }
    for i in range(model.config.num_hidden_layers):
        for projection in PROJECTIONS:
            name = f"layers.{i}.{projection}"
            layer = model.get_submodule(name)
            assert isinstance(layer, QuantLinear), name
            values, scale = quantize_per_channel(tensors[checkpoint_name_of[f"{name}.weight"]])
            assert torch.equal(layer.weight, values), name
            assert torch.equal(layer.scale, scale), name
    assert type(model.lm_head) is nn.Linear
    assert model.lm_head.weight is model.embed_tokens.weight
    assert model.embed_tokens.weight.dtype == torch.float32
    assert not model.training


def test_int8_model_is_close_to_the_float_model(tiny_checkpoint: Path) -> None:
    float_model = load_model(tiny_checkpoint, device=CPU, dtype=torch.float32)
    int8_model = load_model(tiny_checkpoint, device=CPU, dtype=torch.float32, quantization="int8")
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, float_model.config.vocab_size, (2, 16), generator=generator)
    positions = torch.arange(16).expand(2, 16)
    every_token = torch.arange(32)
    with torch.inference_mode():
        expected = float_model(input_ids, positions, every_token)
        actual = int8_model(input_ids, positions, every_token)
    assert not torch.equal(actual, expected)
    # Measured: at most 0.058 apart, on logits of up to 3.5 in magnitude (std 0.81).
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.15)


def test_int8_keeps_an_untied_lm_head_and_biases_in_the_model_dtype(
    make_tiny_checkpoint: Callable[..., Path],
) -> None:
    checkpoint = make_tiny_checkpoint(tie_word_embeddings=False, attention_bias=True)
    model = load_model(checkpoint, device=CPU, dtype=torch.float16, quantization="int8")
    tensors = checkpoint_tensors(checkpoint)
    assert type(model.lm_head) is nn.Linear
    assert torch.equal(model.lm_head.weight, tensors["lm_head.weight"].half())
    q_proj = model.get_submodule("layers.0.attn.q_proj")
    assert isinstance(q_proj, QuantLinear)
    assert q_proj.scale.dtype == torch.float16
    assert q_proj.bias is not None
    assert torch.equal(q_proj.bias, tensors["model.layers.0.self_attn.q_proj.bias"].half())


def test_unknown_quantization_raises(tiny_checkpoint: Path) -> None:
    with pytest.raises(ValueError, match="unknown quantization 'int4'; supported: int8"):
        load_model(tiny_checkpoint, device=CPU, dtype=torch.float32, quantization=cast(Any, "int4"))


def test_missing_tensors_are_named(tiny_checkpoint: Path, tmp_path: Path) -> None:
    tensors = checkpoint_tensors(tiny_checkpoint)
    del tensors["model.norm.weight"]
    del tensors["model.layers.1.mlp.up_proj.weight"]
    variant = write_variant(tiny_checkpoint, tmp_path / "variant", tensors)
    with pytest.raises(
        CheckpointError,
        match=r"missing tensors: model\.layers\.1\.mlp\.up_proj\.weight, model\.norm\.weight$",
    ):
        load_model(variant, device=CPU, dtype=torch.float32)


def test_unexpected_tensors_are_named(tiny_checkpoint: Path, tmp_path: Path) -> None:
    tensors = checkpoint_tensors(tiny_checkpoint)
    tensors["model.layers.0.self_attn.rotary_emb.inv_freq"] = torch.ones(8)
    variant = write_variant(tiny_checkpoint, tmp_path / "variant", tensors)
    with pytest.raises(
        CheckpointError, match=r"unexpected tensors: model\.layers\.0\.self_attn\.rotary_emb"
    ):
        load_model(variant, device=CPU, dtype=torch.float32)


def test_lm_head_in_a_tied_checkpoint_is_rejected(tiny_checkpoint: Path, tmp_path: Path) -> None:
    tensors = checkpoint_tensors(tiny_checkpoint)
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
    variant = write_variant(tiny_checkpoint, tmp_path / "variant", tensors)
    with pytest.raises(CheckpointError, match="tie_word_embeddings is true"):
        load_model(variant, device=CPU, dtype=torch.float32)


def test_shape_mismatches_are_named(tiny_checkpoint: Path, tmp_path: Path) -> None:
    tensors = checkpoint_tensors(tiny_checkpoint)
    tensors["model.layers.0.mlp.gate_proj.weight"] = torch.zeros(96, 64)
    variant = write_variant(tiny_checkpoint, tmp_path / "variant", tensors)
    with pytest.raises(
        CheckpointError, match=r"model\.layers\.0\.mlp\.gate_proj\.weight \[96, 64\] vs \[128, 64\]"
    ):
        load_model(variant, device=CPU, dtype=torch.float32)


def test_long_name_lists_are_shortened(tiny_checkpoint: Path, tmp_path: Path) -> None:
    tensors = checkpoint_tensors(tiny_checkpoint)
    layer_1 = [name for name in tensors if name.startswith("model.layers.1.")]
    for name in layer_1:
        del tensors[name]
    variant = write_variant(tiny_checkpoint, tmp_path / "variant", tensors)
    with pytest.raises(CheckpointError, match=f"and {len(layer_1) - 8} more$"):
        load_model(variant, device=CPU, dtype=torch.float32)


def test_missing_shard_is_reported(
    make_tiny_checkpoint: Callable[..., Path], tmp_path: Path
) -> None:
    variant = shutil.copytree(make_tiny_checkpoint(max_shard_size="40KB"), tmp_path / "variant")
    shard = sorted(variant.glob("model-*.safetensors"))[1]
    shard.unlink()
    with pytest.raises(CheckpointError, match=f"lists missing files: {shard.name}"):
        load_model(variant, device=CPU, dtype=torch.float32)


def test_directory_without_weights_raises(tiny_checkpoint: Path, tmp_path: Path) -> None:
    shutil.copy(tiny_checkpoint / "config.json", tmp_path / "config.json")
    with pytest.raises(CheckpointError, match=r"no model\.safetensors"):
        load_model(tmp_path, device=CPU, dtype=torch.float32)


def test_directory_without_config_raises(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match=r"no config\.json"):
        load_model(tmp_path, device=CPU, dtype=torch.float32)


@pytest.mark.parametrize("name", ["./no-such-model", "/no/such/model", "~/no-such-model"])
def test_missing_local_path_raises(name: str) -> None:
    with pytest.raises(CheckpointError, match="does not exist"):
        load_model(name, device=CPU, dtype=torch.float32)


def test_repo_id_downloads_from_the_hub(
    tiny_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str | None, list[str]]] = []

    def fake_download(repo_id: str, *, revision: str | None, allow_patterns: list[str]) -> str:
        calls.append((repo_id, revision, allow_patterns))
        return str(tiny_checkpoint)

    monkeypatch.setattr(loader, "snapshot_download", fake_download)
    model = load_model("org/tiny-llama", device=CPU, dtype=torch.float32, revision="abc123")
    assert model.config.num_hidden_layers == 2
    assert calls == [("org/tiny-llama", "abc123", loader.HUB_FILE_PATTERNS)]
    assert {"config.json", "model*.safetensors", "tokenizer.json"} <= set(calls[0][2])
