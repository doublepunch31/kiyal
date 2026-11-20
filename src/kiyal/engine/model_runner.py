import torch
from torch import Tensor

from kiyal.cache.paged import PagedKVPool
from kiyal.config import EngineConfig
from kiyal.engine.sequence import Sequence
from kiyal.model.attention import AttentionMetadata
from kiyal.model.config import ModelConfig
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


def kv_block_bytes(config: ModelConfig, block_size: int, dtype: torch.dtype) -> int:
    """Bytes of one block of the KV pool: keys and values of `block_size` tokens in every
    layer."""
    per_token = 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim
    return per_token * block_size * dtype.itemsize


def kv_blocks_for_memory(model: LlamaModel, config: EngineConfig, max_len: int) -> int:
    """Returns how many KV blocks fit in the memory left for the pool.

    On CUDA the pool gets what `cuda_kv_cache_bytes` measures. The CPU and MPS share their
    memory with the rest of the machine and report no free figure worth trusting, so the
    pool gets `kv_cache_gb` there.
    """
    weight = model.embed_tokens.weight
    if weight.device.type == "cuda":
        budget = cuda_kv_cache_bytes(model, config, max_len)
    else:
        budget = int(config.kv_cache_gb * 2**30)
    return max(0, budget // kv_block_bytes(model.config, config.block_size, weight.dtype))


def cuda_kv_cache_bytes(model: LlamaModel, config: EngineConfig, max_len: int) -> int:
    """Measures the CUDA memory left for the KV pool, before the pool exists.

    The pool gets `gpu_memory_utilization` of the device's memory, less the peak that the
    weights and a forward pass of the largest possible step reach, and less the memory in
    use outside this process's allocator (the CUDA context, other processes). The forward
    pass is run once here to measure that peak.
    """
    device = model.embed_tokens.weight.device
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    free, total = torch.cuda.mem_get_info(device)
    outside_allocator = total - free - torch.cuda.memory_reserved(device)
    # A step runs at most the token budget, or a preempted sequence of up to max_len
    # tokens alone.
    num_tokens = max(config.max_num_batched_tokens, max_len)
    profile_forward(model, num_tokens, max_len, config.block_size)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    torch.cuda.empty_cache()
    return int(total * config.gpu_memory_utilization) - peak - outside_allocator


@torch.inference_mode()
def profile_forward(model: LlamaModel, num_tokens: int, max_len: int, block_size: int) -> None:
    """Runs the model once on `num_tokens` tokens, as prompts of at most `max_len` tokens.

    The prompts' keys and values go to a pool of their own, just large enough for them.
    """
    device = model.embed_tokens.weight.device
    lengths = [max_len] * (num_tokens // max_len)
    if num_tokens % max_len:
        lengths.append(num_tokens % max_len)
    block_tables: list[list[int]] = []
    num_blocks = 0
    for length in lengths:
        needed = -(-length // block_size)
        block_tables.append(list(range(num_blocks, num_blocks + needed)))
        num_blocks += needed
    pool = PagedKVPool(
        model.config, num_blocks, block_size, device=device, dtype=model.embed_tokens.weight.dtype
    )
    metadata = AttentionMetadata.build(
        block_tables, [0] * len(lengths), lengths, block_size, device
    )
    model.forward_flat(
        torch.zeros(num_tokens, dtype=torch.long, device=device),  # [N]
        torch.cat([torch.arange(length, device=device) for length in lengths]),  # [N]
        pool,
        metadata,
        torch.tensor(lengths, device=device).cumsum(0) - 1,  # [B]
    )
