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


class ContiguousKVPool:
    """Keys and values of many sequences, each in its own row of `max_len` positions.

    Unlike ContiguousKVCache, rows are not tied to one batch: a sequence gets a free row when
    it starts and gives it back when it ends, and rows hold different numbers of tokens. That
    lets sequences join and leave a running batch at any step. The pool does not track which
    rows are in use or how full they are; its callers do.

    Every row reserves `max_len` positions whatever its sequence will use. A paged pool,
    which hands out memory in small blocks as sequences grow, removes that waste.
    """

    def __init__(
        self,
        config: ModelConfig,
        num_rows: int,
        max_len: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Allocates `num_rows` rows of `max_len` positions each.

        Raises:
            ValueError: If `num_rows` or `max_len` is below 1.
        """
        if num_rows < 1:
            raise ValueError(f"num_rows must be 1 or greater, got {num_rows}")
        if max_len < 1:
            raise ValueError(f"max_len must be 1 or greater, got {max_len}")
        # Positions come before heads so that one layer's keys, viewed as [rows * max_len,
        # Hkv, D], take each new token's [Hkv, D] at a single flat slot index.
        # Zeroed, unlike ContiguousKVCache: a batch reads every row up to its longest
        # sequence, so a shorter sequence's row is read past its end. The mask gives those
        # positions a weight of exactly 0, but 0 times a NaN from uninitialized memory is
        # NaN. Values left by an earlier sequence in the row are finite and do no harm.
        self._data = torch.zeros(
            config.num_hidden_layers,
            2,
            num_rows,
            max_len,
            config.num_key_value_heads,
            config.head_dim,
            device=device,
            dtype=dtype,
        )  # [layers, 2, rows, T_max, Hkv, D], keys at index 0 and values at index 1

    @property
    def num_rows(self) -> int:
        return self._data.shape[2]

    @property
    def max_len(self) -> int:
        return self._data.shape[3]

    def write(self, layer: int, slot_mapping: Tensor, keys: Tensor, values: Tensor) -> None:
        """Stores one layer's keys and values for new tokens of any rows.

        Args:
            layer: Index of the decoder layer.
            slot_mapping: [N] slot of each token, `row * max_len + position`.
            keys: [N, Hkv, D] keys of the tokens, with rotary embedding applied.
            values: [N, Hkv, D] values of the tokens.
        """
        _, _, num_rows, max_len, num_kv_heads, head_dim = self._data.shape
        flat_shape = (num_rows * max_len, num_kv_heads, head_dim)
        # index_copy_ requires the source to match the slots exactly; it never broadcasts.
        self._data[layer, 0].view(flat_shape).index_copy_(0, slot_mapping, keys)
        self._data[layer, 1].view(flat_shape).index_copy_(0, slot_mapping, values)

    def read(self, layer: int, rows: Tensor, length: int) -> tuple[Tensor, Tensor]:
        """Returns one layer's keys and values at the first `length` positions of some rows.

        Positions past a row's own tokens hold zeros or values left by an earlier sequence,
        which the caller must mask out.

        Args:
            layer: Index of the decoder layer.
            rows: [B] rows to read, in the order of the result.
            length: Number of positions to read from each row.

        Returns:
            Keys and values, each [B, Hkv, length, D]. Selecting rows by index copies them.
        """
        keys = self._data[layer, 0, rows, :length]  # [B, L, Hkv, D]
        values = self._data[layer, 1, rows, :length]  # [B, L, Hkv, D]
        return keys.transpose(1, 2), values.transpose(1, 2)  # [B, Hkv, L, D] each

    def view(self, layer: int, num_rows: int, length: int) -> tuple[Tensor, Tensor]:
        """Returns one layer's keys and values at the first `length` positions of the first
        `num_rows` rows, without copying.

        Like `read`, positions past a row's own tokens must be masked out, and so must rows
        that hold no sequence of the batch.

        Returns:
            Keys and values, each [num_rows, Hkv, length, D], views into the pool.
        """
        keys = self._data[layer, 0, :num_rows, :length]  # [R, L, Hkv, D]
        values = self._data[layer, 1, :num_rows, :length]  # [R, L, Hkv, D]
        return keys.transpose(1, 2), values.transpose(1, 2)  # [R, Hkv, L, D] each
