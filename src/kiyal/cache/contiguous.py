import torch
from torch import Tensor

from kiyal.model.config import ModelConfig


class ContiguousKVCache:
    """Attention keys and values of every layer, kept so that later steps can reuse them.

    Attention is causal, so a token's keys and values depend only on the tokens before it and
    never change once computed. Storing them lets each step run the model on the new tokens
    alone instead of on the whole sequence.

    All layers and all rows of the batch share one tensor that is allocated up front for
    `max_tokens` positions. Every row holds the same number of tokens.
    """

    def __init__(
        self,
        config: ModelConfig,
        batch_size: int,
        max_tokens: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Allocates room for `max_tokens` positions per sequence.

        Raises:
            ValueError: If `batch_size` or `max_tokens` is below 1.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be 1 or greater, got {batch_size}")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be 1 or greater, got {max_tokens}")
        # Left uninitialized: positions are always written before they are read, and filling
        # the tensor would touch memory that a short output never uses.
        self._data = torch.empty(
            config.num_hidden_layers,
            2,
            batch_size,
            config.num_key_value_heads,
            max_tokens,
            config.head_dim,
            device=device,
            dtype=dtype,
        )  # [layers, 2, B, Hkv, T_max, D], keys at index 0 and values at index 1
        self._length = 0

    @property
    def length(self) -> int:
        """Number of positions per sequence that every layer has stored."""
        return self._length

    @property
    def max_tokens(self) -> int:
        return self._data.shape[4]

    def write(self, layer: int, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        """Stores one layer's keys and values for new tokens, after the ones already cached.

        The length does not change until `advance`, so every layer of one forward pass
        writes at the same positions.

        Args:
            layer: Index of the decoder layer.
            keys: [B, Hkv, T, D] keys of the new tokens, with rotary embedding applied.
            values: [B, Hkv, T, D] values of the new tokens.

        Returns:
            Keys and values of the cached and the new tokens, each [B, Hkv, length + T, D].
            They are views into the cache, not copies.

        Raises:
            ValueError: If the shapes do not fit the cache or it has too little room left.
        """
        _, _, batch_size, num_kv_heads, max_tokens, head_dim = self._data.shape
        num_tokens = keys.shape[-2]
        # Checked because assignment would silently broadcast, for example one row's keys
        # into every row of the batch.
        expected = (batch_size, num_kv_heads, num_tokens, head_dim)
        if keys.shape != expected or values.shape != expected:
            raise ValueError(
                f"keys and values must be [{batch_size}, {num_kv_heads}, T, {head_dim}], "
                f"got {list(keys.shape)} and {list(values.shape)}"
            )
        end = self._length + num_tokens
        if end > max_tokens:
            raise ValueError(
                f"the cache holds {self._length} of {max_tokens} tokens and cannot take "
                f"{num_tokens} more"
            )
        self._data[layer, 0, :, :, self._length : end] = keys
        self._data[layer, 1, :, :, self._length : end] = values
        return self._data[layer, 0, :, :, :end], self._data[layer, 1, :, :, :end]

    def advance(self, num_tokens: int) -> None:
        """Counts `num_tokens` more positions as cached, once every layer has written them.

        Raises:
            ValueError: If the cache has too little room for them.
        """
        if not 0 < num_tokens <= self.max_tokens - self._length:
            raise ValueError(
                f"the cache holds {self._length} of {self.max_tokens} tokens and cannot "
                f"advance by {num_tokens}"
            )
        self._length += num_tokens
