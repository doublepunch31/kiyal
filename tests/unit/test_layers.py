import math

import torch
from torch import Tensor
from torch.nn.functional import silu

from kiyal.model.config import Llama3RopeScaling
from kiyal.model.layers import (
    MLP,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_embedding,
    rope_inverse_frequencies,
)

# Values from the Llama 3.1 release configs.
LLAMA3 = Llama3RopeScaling(
    factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192
)


def randn(*shape: int, seed: int = 0) -> Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def test_rms_norm_matches_formula() -> None:
    norm = RMSNorm(8, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(randn(8, seed=1))
    x = randn(3, 5, 8).double()
    expected = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * norm.weight.double()
    torch.testing.assert_close(norm(x.float()), expected.float())


def test_rms_norm_does_not_overflow_in_fp16() -> None:
    norm = RMSNorm(4, eps=1e-5, dtype=torch.float16)
    # 300 squared is above the fp16 maximum of 65504.
    x = torch.full((2, 4), 300.0, dtype=torch.float16)
    assert torch.isinf(x.pow(2)).all()
    out = norm(x)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out, torch.ones(2, 4, dtype=torch.float16))


def test_inverse_frequencies_follow_theta_power() -> None:
    expected = torch.tensor([10000.0 ** (-2 * i / 8) for i in range(4)])
    torch.testing.assert_close(rope_inverse_frequencies(8, theta=10000.0), expected)


def rotate_reference(x: Tensor, positions: Tensor, inv_freq: Tensor) -> Tensor:
    # Treat dimensions (i, i + D/2) as the real and imaginary parts of one complex number and
    # multiply it by e^(j * position * inv_freq[i]).
    half = x.shape[-1] // 2
    z = torch.complex(x[..., :half].double(), x[..., half:].double())  # [B, H, T, D/2]
    angles = positions[:, None, :, None].double() * inv_freq.double()  # [B, 1, T, D/2]
    rotated = z * torch.polar(torch.ones_like(angles), angles)  # [B, H, T, D/2]
    return torch.cat((rotated.real, rotated.imag), dim=-1).float()  # [B, H, T, D]


def test_rotary_embedding_matches_complex_rotation() -> None:
    rope = RotaryEmbedding(head_dim=8, max_positions=64, theta=10000.0)
    x = randn(2, 3, 5, 8)
    # Rows sit at different positions, as they will in a batch of sequences.
    positions = torch.tensor([[0, 1, 2, 3, 4], [17, 30, 31, 32, 63]])
    cos, sin = rope(positions, torch.float32)
    expected = rotate_reference(x, positions, rope_inverse_frequencies(8, 10000.0))
    torch.testing.assert_close(apply_rotary_embedding(x, cos, sin), expected)


def test_position_zero_is_not_rotated() -> None:
    rope = RotaryEmbedding(head_dim=8, max_positions=4, theta=10000.0)
    x = randn(1, 2, 1, 8)
    cos, sin = rope(torch.tensor([[0]]), torch.float32)
    torch.testing.assert_close(apply_rotary_embedding(x, cos, sin), x)


def test_rotated_dot_product_depends_only_on_offset() -> None:
    rope = RotaryEmbedding(head_dim=8, max_positions=64, theta=10000.0)
    q = randn(1, 1, 1, 8, seed=1)
    k = randn(1, 1, 1, 8, seed=2)

    def score(q_position: int, k_position: int) -> Tensor:
        q_rotated = apply_rotary_embedding(q, *rope(torch.tensor([[q_position]]), torch.float32))
        k_rotated = apply_rotary_embedding(k, *rope(torch.tensor([[k_position]]), torch.float32))
        return (q_rotated * k_rotated).sum()

    torch.testing.assert_close(score(5, 2), score(40, 37))
    assert not torch.allclose(score(5, 2), score(5, 3))


def test_lookup_returns_requested_dtype_and_shape() -> None:
    rope = RotaryEmbedding(head_dim=8, max_positions=16, theta=10000.0)
    cos, sin = rope(torch.arange(6).reshape(2, 3), torch.float16)
    assert cos.shape == sin.shape == (2, 3, 4)
    assert cos.dtype == sin.dtype == torch.float16


def test_llama3_scaling_keeps_high_and_slows_low_frequencies() -> None:
    base = rope_inverse_frequencies(128, 500000.0)
    scaled = rope_inverse_frequencies(128, 500000.0, LLAMA3)
    wavelength = 2 * math.pi / base
    high = wavelength < 8192 / LLAMA3.high_freq_factor
    low = wavelength > 8192 / LLAMA3.low_freq_factor
    middle = ~high & ~low
    assert high.any() and low.any() and middle.any()
    torch.testing.assert_close(scaled[high], base[high])
    torch.testing.assert_close(scaled[low], base[low] / LLAMA3.factor)
    assert torch.all(scaled[middle] < base[middle])
    assert torch.all(scaled[middle] > base[middle] / LLAMA3.factor)
    # The bands join without jumps, so frequencies still decrease across dimensions.
    assert torch.all(scaled[1:] < scaled[:-1])


def test_llama3_scaling_interpolates_middle_band() -> None:
    base = rope_inverse_frequencies(128, 500000.0)
    scaled = rope_inverse_frequencies(128, 500000.0, LLAMA3)
    wavelength = 2 * math.pi / base
    i = int(torch.nonzero((wavelength >= 2048) & (wavelength <= 8192))[0])
    freq = float(base[i])
    smooth = (8192 * freq / (2 * math.pi) - 1.0) / (4.0 - 1.0)
    expected = (1 - smooth) * freq / 8.0 + smooth * freq
    assert math.isclose(float(scaled[i]), expected, rel_tol=1e-6)


def test_mlp_is_swiglu() -> None:
    mlp = MLP(8, 16)
    x = randn(2, 3, 8)
    gate = silu(x @ mlp.gate_proj.weight.T)
    expected = (gate * (x @ mlp.up_proj.weight.T)) @ mlp.down_proj.weight.T
    torch.testing.assert_close(mlp(x), expected)
    assert mlp.gate_proj.bias is None
    assert mlp.up_proj.bias is None
    assert mlp.down_proj.bias is None
