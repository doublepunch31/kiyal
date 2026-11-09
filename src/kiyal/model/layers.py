import math

import torch
from torch import Tensor, nn
from torch.nn.functional import silu

from kiyal.model.config import Llama3RopeScaling


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang and Sennrich, 2019)."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        # In fp16, squaring activations above 256 overflows to inf, and the mean loses precision.
        # Normalizing in fp32 avoids both.
        x32 = x.float()  # [..., hidden]
        rms_inv = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # [..., 1]
        # The weight is applied after casting back, as in the reference, so that reduced
        # precision outputs round the same way.
        return self.weight * (x32 * rms_inv).to(x.dtype)


def rope_inverse_frequencies(
    head_dim: int, theta: float, scaling: Llama3RopeScaling | None = None
) -> Tensor:
    """Returns the rotation speed of each pair of head dimensions, in radians per position.

    Pair i rotates at theta^(-2i / head_dim) (Su et al., 2021). Returns a [D/2] fp32 tensor.
    """
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim  # [D/2]
    inv_freq = 1.0 / (theta**exponents)  # [D/2]
    if scaling is not None:
        inv_freq = _llama3_scale(inv_freq, scaling)
    return inv_freq


def _llama3_scale(inv_freq: Tensor, scaling: Llama3RopeScaling) -> Tensor:
    # Llama 3.1 extends the context by slowing low-frequency rotations, whose wavelength exceeds
    # the trained context, by `factor`. High-frequency rotations encode local order and are kept.
    # The band between the two is interpolated linearly in context_len / wavelength.
    context_len = scaling.original_max_position_embeddings
    wavelength = 2 * math.pi / inv_freq  # [D/2]
    high_freq_wavelength = context_len / scaling.high_freq_factor
    low_freq_wavelength = context_len / scaling.low_freq_factor

    smooth = (context_len / wavelength - scaling.low_freq_factor) / (
        scaling.high_freq_factor - scaling.low_freq_factor
    )  # [D/2], 0 at the low-frequency edge, 1 at the high-frequency edge
    interpolated = (1 - smooth) * inv_freq / scaling.factor + smooth * inv_freq  # [D/2]
    scaled = torch.where(wavelength > low_freq_wavelength, inv_freq / scaling.factor, interpolated)
    return torch.where(wavelength < high_freq_wavelength, inv_freq, scaled)  # [D/2]


class RotaryEmbedding(nn.Module):
    """Precomputed cos and sin tables for rotary position embeddings.

    Tables are looked up with explicit per-token positions, so sequences at different
    positions can share a batch.
    """

    cos: Tensor
    sin: Tensor

    def __init__(
        self,
        head_dim: int,
        max_positions: int,
        theta: float,
        scaling: Llama3RopeScaling | None = None,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        inv_freq = rope_inverse_frequencies(head_dim, theta, scaling)  # [D/2]
        # Angles are computed in fp32 whatever the model dtype: bf16 cannot represent most
        # integers above 256, so positions themselves would be rounded.
        positions = torch.arange(max_positions, dtype=torch.float32)  # [max_positions]
        angles = torch.outer(positions, inv_freq)  # [max_positions, D/2]
        # Stored in fp32 and cast on lookup, so one table serves any activation dtype.
        # Non-persistent because they are derived from the config, not stored in checkpoints.
        self.register_buffer("cos", angles.cos().to(device), persistent=False)
        self.register_buffer("sin", angles.sin().to(device), persistent=False)

    def forward(self, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Returns cos and sin of shape [*positions.shape, D/2] in the given dtype."""
        return self.cos[positions].to(dtype), self.sin[positions].to(dtype)


def apply_rotary_embedding(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotates query or key heads by their positions.

    Uses the rotate-half layout of Hugging Face Llama checkpoints: dimension i is paired with
    dimension i + D/2, not with its neighbor as in the original paper. Such checkpoints were
    either trained in this layout or had their q and k projections permuted into it on
    conversion, so the interleaved form gives wrong outputs on them.

    Args:
        x: [B, H, T, D] heads.
        cos: [B, T, D/2] from RotaryEmbedding.
        sin: [B, T, D/2] from RotaryEmbedding.
    """
    cos = cos.unsqueeze(1)  # [B, 1, T, D/2], shared by all heads
    sin = sin.unsqueeze(1)  # [B, 1, T, D/2]
    x1, x2 = x.chunk(2, dim=-1)  # [B, H, T, D/2] each
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)  # [B, H, T, D]


class MLP(nn.Module):
    """SwiGLU feed-forward block (Shazeer, 2020): down(silu(gate(x)) * up(x))."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, device=device, dtype=dtype
        )
        self.up_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, device=device, dtype=dtype
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, bias=False, device=device, dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        gate = silu(self.gate_proj(x))  # [..., intermediate]
        return self.down_proj(gate * self.up_proj(x))  # [..., hidden]
