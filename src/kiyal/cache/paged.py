import torch
from torch import Tensor

from kiyal.model.config import ModelConfig


class PagedKVPool:
    """Keys and values of many sequences, stored in blocks of `block_size` positions.

    A sequence's tokens are spread over blocks anywhere in the pool, which its block table
    lists in order (see BlockManager). The pool itself does not know which blocks are in use
    or by whom. Writes go to flat slots, `block * block_size + offset`, and reads gather a
    sequence's blocks back into one run of positions.

    With a block size of the longest sequence, each sequence has one block, which is a row
    reserved for its whole length as in a contiguous pool.
    """

    def __init__(
        self,
        config: ModelConfig,
        num_blocks: int,
        block_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Allocates `num_blocks` blocks of `block_size` positions each.

        Raises:
            ValueError: If `num_blocks` or `block_size` is below 1.
        """
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be 1 or greater, got {num_blocks}")
        if block_size < 1:
            raise ValueError(f"block_size must be 1 or greater, got {block_size}")
        # Positions before heads: one layer's keys, viewed as [NB * BS, Hkv, D], take each
        # new token's [Hkv, D] at a single flat slot index, and a gather of whole blocks
        # [B, MB, BS, Hkv, D] joins each sequence's positions with a view instead of a copy.
        # Zeroed: a batch reads every sequence up to the longest one, so a shorter sequence's
        # blocks are read past its end. The mask gives those positions a weight of exactly 0,
        # but 0 times a NaN from uninitialized memory is NaN. Values left by an earlier
        # sequence in a block are finite and do no harm.
        self._data = torch.zeros(
            config.num_hidden_layers,
            2,
            num_blocks,
            block_size,
            config.num_key_value_heads,
            config.head_dim,
            device=device,
            dtype=dtype,
        )  # [layers, 2, NB, BS, Hkv, D], keys at index 0 and values at index 1

    @property
    def num_blocks(self) -> int:
        return self._data.shape[2]

    @property
    def block_size(self) -> int:
        return self._data.shape[3]

    def write(self, layer: int, slot_mapping: Tensor, keys: Tensor, values: Tensor) -> None:
        """Stores one layer's keys and values for new tokens of any sequences.

        Args:
            layer: Index of the decoder layer.
            slot_mapping: [N] slot of each token, `block * block_size + offset`.
            keys: [N, Hkv, D] keys of the tokens, with rotary embedding applied.
            values: [N, Hkv, D] values of the tokens.
        """
        _, _, num_blocks, block_size, num_kv_heads, head_dim = self._data.shape
        flat_shape = (num_blocks * block_size, num_kv_heads, head_dim)
        # index_copy_ requires the source to match the slots exactly; it never broadcasts.
        self._data[layer, 0].view(flat_shape).index_copy_(0, slot_mapping, keys)
        self._data[layer, 1].view(flat_shape).index_copy_(0, slot_mapping, values)

    def gather(self, layer: int, block_tables: Tensor, length: int) -> tuple[Tensor, Tensor]:
        """Returns one layer's keys and values at the first `length` positions of sequences.

        Positions past a sequence's own tokens hold zeros or values left by an earlier
        sequence, which the caller must mask out.

        Args:
            layer: Index of the decoder layer.
            block_tables: [B, MB] blocks of each sequence in order, MB * block_size >=
                length. Entries past a sequence's own blocks may name any block.
            length: Number of positions to read for each sequence.

        Returns:
            Keys and values, each [B, Hkv, length, D], copied out of the pool.
        """
        batch_size, num_table_blocks = block_tables.shape
        _, _, _, block_size, num_kv_heads, head_dim = self._data.shape
        joined = (batch_size, num_table_blocks * block_size, num_kv_heads, head_dim)
        # index_select of whole blocks measured faster than advanced indexing on both the CPU
        # (23 against 29 ms per decode step) and MPS (5 against 32 ms), at 32 sequences of 300
        # tokens over 30 layers.
        flat_tables = block_tables.flatten()
        keys = self._data[layer, 0].index_select(0, flat_tables).view(joined)[:, :length]
        values = self._data[layer, 1].index_select(0, flat_tables).view(joined)[:, :length]
        # keys, values: [B, L, Hkv, D]
        return keys.transpose(1, 2), values.transpose(1, 2)  # [B, Hkv, L, D] each

    def view(self, layer: int, num_blocks: int, length: int) -> tuple[Tensor, Tensor]:
        """Returns one layer's keys and values at the first `length` positions of the first
        `num_blocks` blocks, without copying.

        For sequences that each fit in one block. Like `gather`, positions past a sequence's
        own tokens must be masked out, and so must blocks that hold no sequence of the
        batch.

        Returns:
            Keys and values, each [num_blocks, Hkv, length, D], views into the pool.

        Raises:
            ValueError: If `length` is longer than a block.
        """
        if length > self.block_size:
            raise ValueError(f"cannot view {length} positions of blocks of {self.block_size}")
        keys = self._data[layer, 0, :num_blocks, :length]  # [A, L, Hkv, D]
        values = self._data[layer, 1, :num_blocks, :length]  # [A, L, Hkv, D]
        return keys.transpose(1, 2), values.transpose(1, 2)  # [A, Hkv, L, D] each
