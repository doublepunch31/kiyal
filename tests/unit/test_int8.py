import pytest
import torch
from torch import Tensor, nn

from kiyal.model.config import ModelConfig
from kiyal.model.llama import LlamaModel
from kiyal.quant.int8 import QuantLinear, quantize_linear_layers, quantize_per_channel


def randn(*shape: int, seed: int = 0) -> Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def test_values_round_to_the_nearest_step() -> None:
    weight = torch.tensor([[1.0, 0.5, -0.25, 0.0, -1.0 / 127]])
    values, scale = quantize_per_channel(weight)
    assert scale.item() == pytest.approx(1 / 127)
    # 63.5 rounds to the even 64, and -31.75 to -32.
    assert values.tolist() == [[127, 64, -32, 0, -1]]


def test_values_are_int8_in_a_symmetric_range() -> None:
    weight = randn(16, 40) * torch.linspace(0.01, 5.0, 16).unsqueeze(1)
    values, scale = quantize_per_channel(weight)
    assert values.dtype == torch.int8
    assert scale.shape == (16, 1)
    # Each row's largest element sets the scale, so it lands on the end of the range.
    assert values.abs().amax(dim=1).tolist() == [127] * 16
    assert values.min() >= -127


def test_dequantized_weight_is_within_half_a_step() -> None:
    weight = randn(32, 64) * torch.linspace(0.05, 3.0, 32).unsqueeze(1)
    values, scale = quantize_per_channel(weight)
    error = (weight - values.float() * scale).abs()
    assert (error <= scale / 2 * (1 + 1e-6)).all()
    assert error.max() > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scale_keeps_the_weight_dtype(dtype: torch.dtype) -> None:
    weight = randn(8, 24).to(dtype)
    values, scale = quantize_per_channel(weight)
    assert scale.dtype == dtype
    # The values are computed from the rounded scale, so half a step holds for the stored
    # scale too.
    error = (weight.float() - values.float() * scale.float()).abs()
    assert (error <= scale.float() / 2 * (1 + 1e-6)).all()


def test_zero_row_stays_zero() -> None:
    weight = randn(3, 8)
    weight[1] = 0.0
    values, scale = quantize_per_channel(weight)
    assert torch.isfinite(scale).all()
    assert scale[1].item() > 0
    assert (values[1] == 0).all()


def test_tiny_rows_keep_a_normal_fp16_scale() -> None:
    # 1e-6 / 127 is far below fp16's smallest normal number (6.1e-5) and would round to zero.
    weight = torch.full((1, 4), 1e-6).to(torch.float16)
    _, scale = quantize_per_channel(weight)
    assert scale.item() == torch.finfo(torch.float16).tiny


@pytest.mark.parametrize("weight", [torch.ones(4), torch.ones(2, 3, 4), torch.ones(2, 3).int()])
def test_quantize_rejects_other_tensors(weight: Tensor) -> None:
    with pytest.raises(ValueError, match="expected a 2-D floating-point weight"):
        quantize_per_channel(weight)


def make_linear(in_features: int, out_features: int, *, bias: bool, seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(in_features, out_features, bias=bias).requires_grad_(False)


@pytest.mark.parametrize("bias", [False, True])
def test_forward_is_a_matmul_with_the_dequantized_weight(bias: bool) -> None:
    float_layer = make_linear(24, 10, bias=bias)
    layer = QuantLinear.from_linear(float_layer)
    values, scale = quantize_per_channel(float_layer.weight)
    x = randn(2, 5, 24, seed=1)
    expected = x @ (values.float() * scale).T
    if layer.bias is not None:
        expected = expected + layer.bias
    torch.testing.assert_close(layer(x), expected)
    torch.testing.assert_close(layer(x.reshape(10, 24)), expected.reshape(10, 10))


def test_forward_error_is_bounded_by_half_a_step() -> None:
    float_layer = make_linear(64, 16, bias=True)
    layer = QuantLinear.from_linear(float_layer)
    x = randn(7, 64, seed=2)
    # Each weight is off by at most half its row's step, so output i is off by at most
    # sum_j |x_j| * scale_i / 2.
    bound = x.abs().sum(dim=1, keepdim=True) * layer.scale.T / 2  # [7, 16]
    error = (layer(x) - float_layer(x)).abs()
    assert (error <= bound * (1 + 1e-5) + 1e-6).all()
    assert error.max() > 0


def test_from_linear_keeps_the_bias_device_and_dtype() -> None:
    float_layer = make_linear(12, 6, bias=True).to(torch.float16)
    layer = QuantLinear.from_linear(float_layer)
    assert (layer.in_features, layer.out_features) == (12, 6)
    assert layer.weight.dtype == torch.int8
    assert layer.scale.dtype == torch.float16
    assert layer.bias is not None
    assert torch.equal(layer.bias, float_layer.bias)
    assert not layer.bias.requires_grad
    assert layer.weight.device == float_layer.weight.device
    assert "in_features=12, out_features=6, bias=True" in repr(layer)


def test_weight_takes_one_byte_per_element() -> None:
    float_layer = make_linear(256, 128, bias=False).to(torch.float16)
    layer = QuantLinear.from_linear(float_layer)
    assert layer.weight.nbytes == 128 * 256
    assert layer.weight.nbytes + layer.scale.nbytes < float_layer.weight.nbytes * 0.51


def test_buffers_are_in_the_state_dict_and_move_with_the_module() -> None:
    layer = QuantLinear.from_linear(make_linear(8, 4, bias=False))
    assert set(layer.state_dict()) == {"weight", "scale"}
    layer = layer.to(torch.float16)
    # Moving to a floating dtype converts floating tensors only, so the values stay int8.
    assert layer.weight.dtype == torch.int8
    assert layer.scale.dtype == torch.float16


def test_constructor_checks_shapes_and_dtypes() -> None:
    values = torch.zeros(4, 8, dtype=torch.int8)
    scale = torch.ones(4, 1)
    with pytest.raises(ValueError, match="2-D int8 weight"):
        QuantLinear(values.float(), scale)
    with pytest.raises(ValueError, match=r"scale of shape \[4, 1\]"):
        QuantLinear(values, torch.ones(4))
    with pytest.raises(ValueError, match=r"bias of shape \[4\]"):
        QuantLinear(values, scale, torch.ones(4, dtype=torch.float16))


def test_quantize_linear_layers_replaces_all_but_the_skipped(tiny_config: ModelConfig) -> None:
    model = LlamaModel(tiny_config).requires_grad_(False)
    replaced = quantize_linear_layers(model, skip={"lm_head"})
    projections = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert len(replaced) == tiny_config.num_hidden_layers * len(projections)
    assert replaced[:4] == [f"layers.0.attn.{name}" for name in projections[:4]]
    for name in replaced:
        assert isinstance(model.get_submodule(name), QuantLinear)
    assert type(model.lm_head) is nn.Linear
    # The tied LM head still shares the embedding weight.
    assert model.lm_head.weight is model.embed_tokens.weight
    assert not any(isinstance(m, nn.Linear) for n, m in model.named_modules() if n != "lm_head")


def test_quantized_model_runs_close_to_the_float_one(tiny_config: ModelConfig) -> None:
    torch.manual_seed(0)
    model = LlamaModel(tiny_config).requires_grad_(False)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 9))
    positions = torch.arange(9).expand(2, 9)
    expected = model(input_ids, positions)
    quantize_linear_layers(model, skip={"lm_head"})
    actual = model(input_ids, positions)
    assert not torch.equal(actual, expected)
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
