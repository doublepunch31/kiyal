import torch
from torch import Tensor

from kiyal.cache.paged import PagedKVPool
from kiyal.engine.sequence import Sequence
from kiyal.model.attention import AttentionMetadata
from kiyal.model.llama import LlamaModel


class ModelRunner:
    """Runs the model on the sequences of one step, and owns their KV pool.

    Turns the scheduled sequences into the inputs of the flattened forward: the tokens each
    sequence has not run yet, their positions, where their keys and values go, and which
    tokens to compute logits for.
    """

    def __init__(self, model: LlamaModel, num_blocks: int, block_size: int) -> None:
        """Allocates a KV pool of `num_blocks` blocks of `block_size` positions.

        Args:
            model: The model to run.
            num_blocks: Blocks of the pool, shared by all sequences.
            block_size: Positions per block.
        """
        self._model = model
        self._device = model.embed_tokens.weight.device
        self.pool = PagedKVPool(
            model.config,
            num_blocks,
            block_size,
            device=self._device,
            dtype=model.embed_tokens.weight.dtype,
        )

    @torch.inference_mode()
    def execute(self, sequences: list[Sequence]) -> Tensor:
        """Runs every sequence's uncomputed tokens and returns its next-token logits.

        Each sequence's block table must have room for all its tokens. The keys and values
        of the tokens run here are written to its blocks; updating `num_computed_tokens` is
        left to the caller.

        Returns:
            [B, V] logits, one row per sequence, in order.
        """
        token_ids: list[int] = []
        positions: list[int] = []
        block_tables: list[list[int]] = []
        num_cached: list[int] = []
        num_new: list[int] = []
        logits_indices: list[int] = []
        for seq in sequences:
            new_token_ids = seq.uncomputed_token_ids()
            start = seq.num_computed_tokens
            token_ids += new_token_ids
            positions += range(start, start + len(new_token_ids))
            block_tables.append(seq.block_table)
            num_cached.append(start)
            num_new.append(len(new_token_ids))
            # Only the last new token predicts the next one.
            logits_indices.append(len(token_ids) - 1)

        metadata = AttentionMetadata.build(
            block_tables, num_cached, num_new, self.pool.block_size, self._device
        )
        return self._model.forward_flat(
            torch.tensor(token_ids, device=self._device),  # [N]
            torch.tensor(positions, device=self._device),  # [N]
            self.pool,
            metadata,
            torch.tensor(logits_indices, device=self._device),  # [B]
        )  # [B, V]
