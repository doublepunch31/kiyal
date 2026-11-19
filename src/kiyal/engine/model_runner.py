import torch
from torch import Tensor

from kiyal.cache.contiguous import ContiguousKVPool
from kiyal.engine.sequence import Sequence
from kiyal.model.attention import AttentionMetadata
from kiyal.model.llama import LlamaModel


class ModelRunner:
    """Runs the model on the sequences of one step, and owns their KV pool.

    Turns the scheduled sequences into the inputs of the flattened forward: the tokens each
    sequence has not run yet, their positions, where their keys and values go, and which
    tokens to compute logits for.
    """

    def __init__(self, model: LlamaModel, num_rows: int, max_len: int) -> None:
        """Allocates a KV pool of `num_rows` rows of `max_len` positions.

        Args:
            model: The model to run.
            num_rows: Rows of the pool, one per sequence in the batch.
            max_len: Positions per row, the most tokens one sequence can hold.
        """
        self._model = model
        self._device = model.embed_tokens.weight.device
        self.pool = ContiguousKVPool(
            model.config,
            num_rows,
            max_len,
            device=self._device,
            dtype=model.embed_tokens.weight.dtype,
        )

    @torch.inference_mode()
    def execute(self, sequences: list[Sequence]) -> Tensor:
        """Runs every sequence's uncomputed tokens and returns its next-token logits.

        Each sequence must hold a pool row. The keys and values of the tokens run here are
        written to it; updating `num_computed_tokens` is left to the caller.

        Returns:
            [B, V] logits, one row per sequence, in order.
        """
        token_ids: list[int] = []
        positions: list[int] = []
        rows: list[int] = []
        num_cached: list[int] = []
        num_new: list[int] = []
        logits_indices: list[int] = []
        for seq in sequences:
            assert seq.cache_row is not None, f"request {seq.request_id} has no pool row"
            new_token_ids = seq.uncomputed_token_ids()
            start = seq.num_computed_tokens
            token_ids += new_token_ids
            positions += range(start, start + len(new_token_ids))
            rows.append(seq.cache_row)
            num_cached.append(start)
            num_new.append(len(new_token_ids))
            # Only the last new token predicts the next one.
            logits_indices.append(len(token_ids) - 1)

        metadata = AttentionMetadata.build(
            rows, num_cached, num_new, self.pool.max_len, self._device
        )
        return self._model.forward_flat(
            torch.tensor(token_ids, device=self._device),  # [N]
            torch.tensor(positions, device=self._device),  # [N]
            self.pool,
            metadata,
            torch.tensor(logits_indices, device=self._device),  # [B]
        )  # [B, V]
