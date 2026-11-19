from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class EngineConfig:
    """Limits of the continuous-batching engine.

    Attributes:
        max_num_seqs: The most sequences in the batch at once. The KV pool has one row for
            each.
        max_num_batched_tokens: The most tokens one step runs through the model. Bounds the
            prompts started together, and so the activation memory of a step. A prompt
            longer than this is rejected, because it could never start.
        max_model_len: The most prompt and output tokens one sequence can hold. Every row of
            the KV pool reserves this many positions. The engine lowers it to the model's
            context if that is shorter.

    Raises:
        ValueError: If a limit is below 1, or a decode step of `max_num_seqs` sequences
            would not fit in `max_num_batched_tokens`.
    """

    max_num_seqs: int = 16
    max_num_batched_tokens: int = 2048
    max_model_len: int = 2048

    def __post_init__(self) -> None:
        limits = {
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_model_len": self.max_model_len,
        }
        for name, value in limits.items():
            if value < 1:
                raise ValueError(f"{name} must be 1 or greater, got {value}")
        # A decode step runs one token of every sequence in the batch.
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must be at least "
                f"max_num_seqs ({self.max_num_seqs})"
            )
