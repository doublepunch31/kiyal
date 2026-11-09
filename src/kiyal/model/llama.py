import torch
from torch import Tensor, nn

from kiyal.model.attention import Attention
from kiyal.model.config import ModelConfig
from kiyal.model.layers import MLP, RMSNorm, RotaryEmbedding


class DecoderLayer(nn.Module):
    """Pre-norm transformer block with a residual connection around attention and the MLP."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.attn_norm = RMSNorm(hidden, eps, device=device, dtype=dtype)
        self.attn = Attention(config, device=device, dtype=dtype)
        self.mlp_norm = RMSNorm(hidden, eps, device=device, dtype=dtype)
        self.mlp = MLP(hidden, config.intermediate_size, device=device, dtype=dtype)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)  # [B, T, hidden]
        return x + self.mlp(self.mlp_norm(x))  # [B, T, hidden]


class LlamaModel(nn.Module):
    """Llama decoder stack with its language-model head."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=dtype
        )
        # One table serves every layer, and its rows are gathered once per forward.
        self.rotary = RotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.rope_scaling,
            device=device,
        )
        self.layers = nn.ModuleList(
            DecoderLayer(config, device=device, dtype=dtype)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, device=device, dtype=dtype)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype
        )
        if config.tie_word_embeddings:
            # Sharing one Parameter keeps the two in sync and makes named_parameters() list it
            # once, so the loader expects a single tensor.
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self, input_ids: Tensor, positions: Tensor, logits_indices: Tensor | None = None
    ) -> Tensor:
        """Computes next-token logits for selected tokens.

        Sampling needs only the last token of each sequence. Computing logits for every token
        would allocate [B*T, V], which for a long prompt and a large vocabulary is hundreds of
        megabytes, and spend LM-head FLOPs on rows nobody reads.

        Args:
            input_ids: [B, T] token ids.
            positions: [B, T] position of each token in its sequence.
            logits_indices: [K] indices into the flattened B*T tokens (row b, column t is
                b * T + t). Defaults to the last token of each row.

        Returns:
            [K, V] logits, or [B, V] when logits_indices is None.
        """
        batch_size, seq_len = input_ids.shape
        hidden = self.embed_tokens(input_ids)  # [B, T, hidden]
        cos, sin = self.rotary(positions, hidden.dtype)  # [B, T, D/2] each
        for layer in self.layers:
            hidden = layer(hidden, cos, sin)  # [B, T, hidden]

        if logits_indices is None:
            selected = hidden[:, -1]  # [B, hidden]
        else:
            selected = hidden.reshape(batch_size * seq_len, -1)[logits_indices]  # [K, hidden]
        # The final norm acts on each token separately, so it runs after selection.
        return self.lm_head(self.norm(selected))  # [K, V]
