from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class EngineConfig:
    """Limits of the continuous-batching engine.

    Attributes:
        max_num_seqs: The most sequences in the batch at once.
        max_num_batched_tokens: The most prompt tokens that start in one step. Bounds the
            activation memory of a step. A prompt longer than this is rejected, because it
            could never start. A preempted sequence, which reruns its output too, may be
            longer; it then runs alone in its step.
        max_model_len: The most prompt and output tokens one sequence can hold. The engine
            lowers it to the model's context if that is shorter.
        block_size: Tokens per block of the paged KV pool.
        num_kv_blocks: Blocks of the KV pool. None sizes the pool from memory, as below,
            but never beyond `max_num_seqs` sequences of `max_model_len` tokens, the most
            the batch can use. Fewer blocks than that let more sequences share less
            memory, at the risk of preemption when they grow.
        watermark: Share of the KV blocks that new sequences must leave free, so that the
            running ones can still grow right after an admission.
        enable_prefix_caching: Lets a sequence share the KV blocks of an earlier one whose
            tokens start the same way, such as a common system prompt, so its prefill starts
            after them. Outputs are the same as without it, up to rounding.
        gpu_memory_utilization: Share of a CUDA device's memory that the weights, the
            activations of a step and the KV pool may use together.
        kv_cache_gb: Memory of the KV pool on the CPU and MPS, in GiB.

    Raises:
        ValueError: If a limit is below 1, a decode step of `max_num_seqs` sequences
            would not fit in `max_num_batched_tokens`, the watermark is outside [0, 1),
            `gpu_memory_utilization` is outside (0, 1], or `kv_cache_gb` is not above 0.
    """

    max_num_seqs: int = 16
    max_num_batched_tokens: int = 2048
    max_model_len: int = 2048
    block_size: int = 16
    num_kv_blocks: int | None = None
    watermark: float = 0.01
    enable_prefix_caching: bool = True
    gpu_memory_utilization: float = 0.85
    kv_cache_gb: float = 1.0

    def __post_init__(self) -> None:
        limits = {
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_model_len": self.max_model_len,
            "block_size": self.block_size,
        }
        if self.num_kv_blocks is not None:
            limits["num_kv_blocks"] = self.num_kv_blocks
        for name, value in limits.items():
            if value < 1:
                raise ValueError(f"{name} must be 1 or greater, got {value}")
        # A decode step runs one token of every sequence in the batch.
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must be at least "
                f"max_num_seqs ({self.max_num_seqs})"
            )
        if not 0 <= self.watermark < 1:
            raise ValueError(f"watermark must be in [0, 1), got {self.watermark}")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}"
            )
        if not self.kv_cache_gb > 0:
            raise ValueError(f"kv_cache_gb must be above 0, got {self.kv_cache_gb}")
