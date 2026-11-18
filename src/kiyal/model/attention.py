import torch
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention

from kiyal.cache.contiguous import ContiguousKVCache
from kiyal.model.config import ModelConfig
from kiyal.model.layers import apply_rotary_embedding


class Attention(nn.Module):
    """Causal self-attention with grouped-query attention (Ainslie et al., 2023).

    Each group of num_attention_heads / num_key_value_heads query heads shares one key and
    value head, which shrinks the KV cache by that factor. Equal head counts give standard
    multi-head attention and a single KV head gives multi-query attention.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        layer_index: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Creates the projections of one layer.

        Args:
            config: Sizes of the model.
            layer_index: Index of this layer, which selects its keys and values in a cache.
            device: Device of the weights.
            dtype: Data type of the weights.
        """
        super().__init__()
        self.layer_index = layer_index
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        bias = config.attention_bias
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, q_size, bias=bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden, kv_size, bias=bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden, kv_size, bias=bias, device=device, dtype=dtype)
        self.o_proj = nn.Linear(q_size, hidden, bias=bias, device=device, dtype=dtype)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: ContiguousKVCache | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Attends each token to itself and the tokens before it.

        With a cache, `x` holds only the tokens that are not cached yet. Their keys and
        values are stored, and their queries attend to the cached tokens as well.

        Args:
            x: [B, T, hidden] normalized hidden states.
            cos: [B, T, D/2] rotary table rows for the tokens' positions.
            sin: [B, T, D/2] rotary table rows for the tokens' positions.
            cache: Keys and values of the tokens before `x`, extended with those of `x`.
            mask: [B, 1, T, T_all] from `padded_causal_mask`, for rows that start with
                padding. None when no row does.

        Returns:
            [B, T, hidden] attention output after the output projection.
        """
        batch_size, seq_len, _ = x.shape
        q_shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        kv_shape = (batch_size, seq_len, self.num_kv_heads, self.head_dim)
        q = self.q_proj(x).view(q_shape)  # [B, T, H, D]
        k = self.k_proj(x).view(kv_shape)  # [B, T, Hkv, D]
        v = self.v_proj(x).view(kv_shape)  # [B, T, Hkv, D]

        # SDPA takes heads before tokens.
        q = apply_rotary_embedding(q.transpose(1, 2), cos, sin)  # [B, H, T, D]
        k = apply_rotary_embedding(k.transpose(1, 2), cos, sin)  # [B, Hkv, T, D]
        v = v.transpose(1, 2)  # [B, Hkv, T, D]

        if cache is not None:
            # Keys are cached after rotation: a key's rotation depends only on its own
            # position, so it never has to be redone.
            k, v = cache.write(self.layer_index, k, v)  # [B, Hkv, T_all, D]
        num_all = k.shape[2]
        # The default scale is 1/sqrt(D). enable_gqa maps query head h to KV head
        # h // (H / Hkv) inside SDPA, so K and V are never expanded here.
        if mask is None and num_all == seq_len:
            # Nothing cached and no padding: SDPA builds the causal mask itself.
            out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            if mask is None:
                mask = _causal_mask(seq_len, num_all, x.device)
            out = scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        # out: [B, H, T, D]
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)  # [B, T, H*D]
        return self.o_proj(out)  # [B, T, hidden]


def _causal_mask(num_new: int, num_all: int, device: torch.device) -> Tensor | None:
    """Returns which of `num_all` tokens each of the last `num_new` tokens may attend to.

    The result is [num_new, num_all], True where attention is allowed, or None when nothing
    has to be hidden.
    """
    if num_new == 1:
        # Decode: the single new token is the last one, so every token is in its past.
        return None
    # Prefill: new token i sits at position num_all - num_new + i and sees the tokens up to
    # and including itself.
    allowed = torch.ones(num_new, num_all, dtype=torch.bool, device=device)
    return allowed.tril(diagonal=num_all - num_new)


def padded_causal_mask(num_new: int, num_cached: int, left_padding: Tensor) -> Tensor:
    """Returns which slots each new token may attend to when rows start with padding.

    The slots of a row are its cached tokens followed by the new ones. Row b holds padding
    in its first `left_padding[b]` slots and tokens after them. A token sees its own slot
    and the token slots before it, never a padding slot.

    Args:
        num_new: Number of new tokens per row, T.
        num_cached: Number of slots per row before them.
        left_padding: [B] number of padding slots at the start of each row.

    Returns:
        [B, 1, T, T_all] booleans, True where attention is allowed. The second dimension
        broadcasts over heads.
    """
    num_all = num_cached + num_new
    device = left_padding.device
    key_slots = torch.arange(num_all, device=device)  # [T_all]
    query_slots = torch.arange(num_cached, num_all, device=device).unsqueeze(-1)  # [T, 1]
    is_token = key_slots >= left_padding.unsqueeze(-1)  # [B, T_all]
    allowed = (key_slots <= query_slots) & is_token.unsqueeze(1)  # [B, T, T_all]
    # A padding slot has no token before it to attend to. A softmax over nothing is NaN on
    # some kernels, and NaN keys and values in the next layer would reach the token rows
    # too, because 0 * NaN is NaN. Seeing itself keeps padding finite, and no token reads it.
    allowed |= key_slots == query_slots
    return allowed.unsqueeze(1)  # [B, 1, T, T_all]
