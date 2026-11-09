import torch
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention

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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
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

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Attends each token to itself and the tokens before it.

        Args:
            x: [B, T, hidden] normalized hidden states.
            cos: [B, T, D/2] rotary table rows for the tokens' positions.
            sin: [B, T, D/2] rotary table rows for the tokens' positions.

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

        # The default scale is 1/sqrt(D). enable_gqa maps query head h to KV head
        # h // (H / Hkv) inside SDPA, so K and V are never expanded here.
        out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)  # [B, H, T, D]
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)  # [B, T, H*D]
        return self.o_proj(out)  # [B, T, hidden]
