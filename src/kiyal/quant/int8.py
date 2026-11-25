from collections.abc import Collection, Iterator
from typing import Final, Self, cast

import torch
from torch import Tensor, nn
from torch.nn.functional import linear

INT8_MAX: Final = 127
# Weights are [out, in] matrices.
WEIGHT_DIMS: Final = 2


def quantize_per_channel(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Quantizes a weight to int8 with one symmetric scale per output channel.

    Row i is stored as round(w_i / s_i) with s_i = max|w_i| / 127, so the row's largest
    element maps to 127 or -127 and every element lies within half a step (s_i / 2) of its
    value. The range is symmetric around zero and -128 is never used, so zero stays exactly
    zero and no offset has to be stored.

    Args:
        weight: [out, in] floating-point weight.

    Returns:
        The [out, in] int8 values and the [out, 1] scales in the weight's dtype.

    Raises:
        ValueError: If the weight is not a 2-D floating-point tensor.
    """
    if weight.dim() != WEIGHT_DIMS or not weight.is_floating_point():
        raise ValueError(
            f"expected a 2-D floating-point weight, got {weight.dim()}-D {weight.dtype}"
        )
    w = weight.float()  # [out, in]
    absmax = w.abs().amax(dim=1, keepdim=True)  # [out, 1]
    # The scale is rounded to the weight's dtype before the values are computed from it, so
    # that the stored scale is the one the values are exact for. The smallest normal number
    # as a floor keeps an all-zero row finite (0 / 0 would be NaN) and keeps the scale out of
    # fp16's subnormal range, where it would lose precision.
    scale = (absmax / INT8_MAX).clamp(min=torch.finfo(weight.dtype).tiny).to(weight.dtype)
    values = torch.round(w / scale.float()).clamp(-INT8_MAX, INT8_MAX)  # [out, in]
    return values.to(torch.int8), scale


class QuantLinear(nn.Module):
    """Linear layer with int8 weights and one scale per output channel (weight-only int8).

    Only the weight is quantized: inputs, outputs and the bias keep the model's dtype. The
    forward dequantizes the whole weight and then runs an ordinary matmul. That halves the
    memory of an fp16 weight, but adds work to every call instead of saving any: decoding
    reads the int8 weight, writes the dequantized copy and reads that copy again.

    Scaling the output instead (`linear(x, q) * scale`) would skip the [out, in] multiply, but
    x @ q.T before the scale is about 1 / scale times the real output. In SmolLM2's layers
    with large activations that exceeds fp16's maximum of 65,504 several times over.
    """

    weight: Tensor
    scale: Tensor

    def __init__(self, weight: Tensor, scale: Tensor, bias: Tensor | None = None) -> None:
        """Wraps quantized values from `quantize_per_channel`.

        Args:
            weight: [out, in] int8 values.
            scale: [out, 1] scale of each output channel, in the dtype of the layer's inputs.
            bias: [out] bias in the same dtype, if the layer has one.

        Raises:
            ValueError: If the shapes or dtypes do not fit together.
        """
        super().__init__()
        if weight.dtype != torch.int8 or weight.dim() != WEIGHT_DIMS:
            raise ValueError(f"expected a 2-D int8 weight, got {weight.dim()}-D {weight.dtype}")
        out_features = weight.shape[0]
        if scale.shape != (out_features, 1) or not scale.is_floating_point():
            raise ValueError(
                f"expected a floating-point scale of shape {[out_features, 1]}, got "
                f"{list(scale.shape)} {scale.dtype}"
            )
        if bias is not None and (bias.shape != (out_features,) or bias.dtype != scale.dtype):
            raise ValueError(
                f"expected a {scale.dtype} bias of shape {[out_features]}, got "
                f"{list(bias.shape)} {bias.dtype}"
            )
        # Buffers, not parameters: int8 tensors cannot require gradients, and nothing here
        # is trained. They still move with `.to()` and appear in the state dict.
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    @classmethod
    def from_linear(cls, layer: nn.Linear) -> Self:
        """Quantizes a linear layer's weight. The bias is kept as it is."""
        values, scale = quantize_per_channel(layer.weight.detach())
        # torch types the bias as always present, but it is None in layers built with
        # bias=False.
        bias = cast(Tensor | None, layer.bias)
        return cls(values, scale, None if bias is None else bias.detach())

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    def dequantize(self) -> Tensor:
        """Returns the [out, in] weight in the scale's dtype."""
        return self.weight.to(self.scale.dtype) * self.scale

    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.dequantize(), self.bias)  # [..., out]

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


def quantize_linear_layers(module: nn.Module, *, skip: Collection[str] = ()) -> list[str]:
    """Replaces every `nn.Linear` inside a module with a `QuantLinear` of its weight.

    Args:
        module: The module to change in place. It cannot be a linear layer itself.
        skip: Qualified names of linear layers to keep, such as "lm_head".

    Returns:
        The qualified names of the replaced layers, in module order.
    """
    # named_modules has no return annotation.
    modules = cast(Iterator[tuple[str, nn.Module]], module.named_modules())
    # Only names are collected, so that each float layer is freed as soon as it is replaced
    # instead of all of them staying alive next to their int8 copies.
    names = [
        name
        for name, child in modules
        if name and isinstance(child, nn.Linear) and name not in skip
    ]
    for name in names:
        layer = module.get_submodule(name)
        assert isinstance(layer, nn.Linear)
        parent_name, _, attribute = name.rpartition(".")
        setattr(module.get_submodule(parent_name), attribute, QuantLinear.from_linear(layer))
    return names
