import dataclasses
import math

import pytest
import torch
from torch import Tensor
from torch.nn.functional import linear

from kiyal.cache.contiguous import ContiguousKVCache
from kiyal.model.attention import Attention
from kiyal.model.config import ModelConfig
from kiyal.model.layers import RotaryEmbedding, apply_rotary_embedding

# Attention over cached keys and values multiplies matrices of other shapes than attention
# over a whole sequence, so its results round differently in the last bits.
CACHED_RTOL = 1e-5
CACHED_ATOL = 1e-5


def reference_attention(attn: Attention, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # Plain attention: copy each KV head to the query heads in its group, mask future tokens
    # with -inf, and softmax explicitly.
    batch_size, seq_len, _ = x.shape
    heads, kv_heads, head_dim = attn.num_heads, attn.num_kv_heads, attn.head_dim
    q = linear(x, attn.q_proj.weight, attn.q_proj.bias)
    k = linear(x, attn.k_proj.weight, attn.k_proj.bias)
    v = linear(x, attn.v_proj.weight, attn.v_proj.bias)
    q = q.view(batch_size, seq_len, heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, kv_heads, head_dim).transpose(1, 2)
    q = apply_rotary_embedding(q, cos, sin)
    k = apply_rotary_embedding(k, cos, sin)
    # Query head h belongs to group h // (heads / kv_heads).
    k = k.repeat_interleave(heads // kv_heads, dim=1)
    v = v.repeat_interleave(heads // kv_heads, dim=1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
    future = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(diagonal=1)
    weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
    out = (weights @ v).transpose(1, 2).reshape(batch_size, seq_len, heads * head_dim)
    return linear(out, attn.o_proj.weight, attn.o_proj.bias)


def inputs(config: ModelConfig, batch_size: int = 2, seq_len: int = 7) -> tuple[Tensor, ...]:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(batch_size, seq_len, config.hidden_size, generator=generator)
    rope = RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)
    positions = torch.arange(seq_len).expand(batch_size, seq_len)
    cos, sin = rope(positions, torch.float32)
    return x, cos, sin


@pytest.mark.parametrize("bias", [False, True], ids=["no-bias", "bias"])
@pytest.mark.parametrize("kv_heads", [4, 2, 1], ids=["mha", "gqa", "mqa"])
@torch.inference_mode()
def test_matches_reference_attention(tiny_config: ModelConfig, kv_heads: int, bias: bool) -> None:
    config = dataclasses.replace(tiny_config, num_key_value_heads=kv_heads, attention_bias=bias)
    torch.manual_seed(0)
    attn = Attention(config)
    x, cos, sin = inputs(config)
    torch.testing.assert_close(attn(x, cos, sin), reference_attention(attn, x, cos, sin))


@pytest.mark.parametrize("chunks", [(7,), (6, 1), (3, 4), (2, 3, 1, 1)])
@pytest.mark.parametrize("kv_heads", [4, 2, 1], ids=["mha", "gqa", "mqa"])
@torch.inference_mode()
def test_cached_attention_matches_reference_attention(
    tiny_config: ModelConfig, kv_heads: int, chunks: tuple[int, ...]
) -> None:
    # The sequence is fed in pieces: a first piece into an empty cache, then pieces of one
    # token (decode) or several (a prefill that continues a cached prefix).
    config = dataclasses.replace(tiny_config, num_key_value_heads=kv_heads)
    torch.manual_seed(0)
    attn = Attention(config)
    x, cos, sin = inputs(config)
    cache = ContiguousKVCache(config, x.shape[0], x.shape[1])
    outputs: list[Tensor] = []
    for size in chunks:
        piece = slice(cache.length, cache.length + size)
        outputs.append(attn(x[:, piece], cos[:, piece], sin[:, piece], cache))
        cache.advance(size)
    expected = reference_attention(attn, x, cos, sin)
    torch.testing.assert_close(
        torch.cat(outputs, dim=1), expected, rtol=CACHED_RTOL, atol=CACHED_ATOL
    )


@torch.inference_mode()
def test_each_layer_reads_its_own_cached_keys_and_values(tiny_config: ModelConfig) -> None:
    torch.manual_seed(0)
    first, second = Attention(tiny_config), Attention(tiny_config)
    x, cos, sin = inputs(tiny_config)
    cache = ContiguousKVCache(tiny_config, x.shape[0], x.shape[1])
    first(x[:, :-1], cos[:, :-1], sin[:, :-1], cache, layer_index=0)
    second(x[:, :-1], cos[:, :-1], sin[:, :-1], cache, layer_index=1)
    cache.advance(x.shape[1] - 1)

    last = second(x[:, -1:], cos[:, -1:], sin[:, -1:], cache, layer_index=1)
    expected = reference_attention(second, x, cos, sin)[:, -1:]
    torch.testing.assert_close(last, expected, rtol=CACHED_RTOL, atol=CACHED_ATOL)


@torch.inference_mode()
def test_later_tokens_do_not_change_earlier_outputs(tiny_config: ModelConfig) -> None:
    attn = Attention(tiny_config)
    x, cos, sin = inputs(tiny_config)
    changed = x.clone()
    changed[:, -1] += 1.0
    before = attn(x, cos, sin)
    after = attn(changed, cos, sin)
    torch.testing.assert_close(after[:, :-1], before[:, :-1])
    assert not torch.allclose(after[:, -1], before[:, -1])


@pytest.mark.parametrize("bias", [False, True])
def test_bias_applies_to_all_four_projections(tiny_config: ModelConfig, bias: bool) -> None:
    attn = Attention(dataclasses.replace(tiny_config, attention_bias=bias))
    projections = [attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj]
    assert all((p.bias is not None) == bias for p in projections)


def test_projection_shapes_follow_head_counts(tiny_config: ModelConfig) -> None:
    # head_dim is set independently of hidden_size / num_attention_heads.
    config = dataclasses.replace(tiny_config, head_dim=32)
    attn = Attention(config)
    assert attn.q_proj.weight.shape == (4 * 32, 64)
    assert attn.k_proj.weight.shape == (2 * 32, 64)
    assert attn.v_proj.weight.shape == (2 * 32, 64)
    assert attn.o_proj.weight.shape == (64, 4 * 32)
    x, cos, sin = inputs(config)
    with torch.inference_mode():
        assert attn(x, cos, sin).shape == x.shape
