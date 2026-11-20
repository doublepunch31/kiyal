import torch
from torch import Tensor, nn

from kiyal.cache.contiguous import ContiguousKVCache
from kiyal.cache.paged import PagedKVPool
from kiyal.model.attention import Attention, AttentionMetadata, padded_causal_mask
from kiyal.model.config import ModelConfig
from kiyal.model.layers import MLP, RMSNorm, RotaryEmbedding


class DecoderLayer(nn.Module):
    """Pre-norm transformer block with a residual connection around attention and the MLP."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        layer_index: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.attn_norm = RMSNorm(hidden, eps, device=device, dtype=dtype)
        self.attn = Attention(config, layer_index=layer_index, device=device, dtype=dtype)
        self.mlp_norm = RMSNorm(hidden, eps, device=device, dtype=dtype)
        self.mlp = MLP(hidden, config.intermediate_size, device=device, dtype=dtype)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: ContiguousKVCache | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, mask)  # [B, T, hidden]
        return x + self.mlp(self.mlp_norm(x))  # [B, T, hidden]

    def forward_flat(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        pool: PagedKVPool,
        metadata: AttentionMetadata,
    ) -> Tensor:
        """Runs the block on a flattened batch; see `LlamaModel.forward_flat`."""
        x = x + self.attn.forward_flat(self.attn_norm(x), cos, sin, pool, metadata)  # [N, hidden]
        return x + self.mlp(self.mlp_norm(x))  # [N, hidden]


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
            DecoderLayer(config, layer_index=index, device=device, dtype=dtype)
            for index in range(config.num_hidden_layers)
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
        self,
        input_ids: Tensor,
        positions: Tensor,
        logits_indices: Tensor | None = None,
        cache: ContiguousKVCache | None = None,
        left_padding: Tensor | None = None,
    ) -> Tensor:
        """Computes next-token logits for selected tokens.

        Sampling needs only the last token of each sequence. Computing logits for every token
        would allocate [B*T, V], which for a long prompt and a large vocabulary is hundreds of
        megabytes, and spend LM-head FLOPs on rows nobody reads.

        Without a cache, `input_ids` holds whole sequences. With one, it holds only the
        tokens that follow the cached ones: the whole prompt while the cache is empty
        (prefill), then one new token per step (decode). The new tokens are added to the
        cache.

        Rows of different lengths share a batch by starting with padding (left padding), so
        that every row's newest token is in the last column. Padding slots are hidden from
        attention, and `positions` count only a row's own tokens.

        Args:
            input_ids: [B, T] token ids.
            positions: [B, T] position of each token in its sequence.
            logits_indices: [K] indices into the flattened B*T tokens (row b, column t is
                b * T + t). Defaults to the last token of each row.
            cache: Keys and values of the tokens before `input_ids`.
            left_padding: [B] number of padding slots at the start of each row, counted from
                the first cached slot. None when no row has padding.

        Returns:
            [K, V] logits, or [B, V] when logits_indices is None.
        """
        batch_size, seq_len = input_ids.shape
        hidden = self.embed_tokens(input_ids)  # [B, T, hidden]
        cos, sin = self.rotary(positions, hidden.dtype)  # [B, T, D/2] each
        mask = None
        if left_padding is not None:
            # Built once here instead of in every layer, because all layers share it.
            num_cached = 0 if cache is None else cache.length
            mask = padded_causal_mask(seq_len, num_cached, left_padding)  # [B, 1, T, T_all]
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, cache, mask)  # [B, T, hidden]
        if cache is not None:
            cache.advance(seq_len)

        if logits_indices is None:
            selected = hidden[:, -1]  # [B, hidden]
        else:
            selected = hidden.reshape(batch_size * seq_len, -1)[logits_indices]  # [K, hidden]
        # The final norm acts on each token separately, so it runs after selection.
        return self.lm_head(self.norm(selected))  # [K, V]

    def forward_flat(
        self,
        input_ids: Tensor,
        positions: Tensor,
        pool: PagedKVPool,
        metadata: AttentionMetadata,
        logits_indices: Tensor,
    ) -> Tensor:
        """Computes next-token logits for a flattened batch of sequences.

        The new tokens of all sequences are laid out one after another without padding, so
        every token-wise layer (projections, MLP, norms) runs on exactly the tokens that
        exist. Only attention needs to know which sequence a token belongs to, which
        `metadata` tells it. Sequences in one batch may be at any stage: some can start
        with their whole prompt while others add one token.

        Args:
            input_ids: [N] new tokens of every sequence.
            positions: [N] position of each token in its own sequence.
            pool: Keys and values of the sequences' earlier tokens. The new tokens' keys
                and values are added to it.
            metadata: Which sequence and pool slot each token belongs to.
            logits_indices: [K] indices into the N tokens to compute logits for, usually
                the last new token of each sequence.

        Returns:
            [K, V] logits.
        """
        hidden = self.embed_tokens(input_ids)  # [N, hidden]
        cos, sin = self.rotary(positions, hidden.dtype)  # [N, D/2] each
        for layer in self.layers:
            # ModuleList yields plain Modules to the type checker, which knows no forward_flat.
            assert isinstance(layer, DecoderLayer)
            hidden = layer.forward_flat(hidden, cos, sin, pool, metadata)  # [N, hidden]
        return self.lm_head(self.norm(hidden[logits_indices]))  # [K, V]
