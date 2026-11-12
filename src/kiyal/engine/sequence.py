import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """How one request picks its tokens.

    Attributes:
        temperature: Divides the logits before sampling. Values below 1 favor likely tokens,
            and 0 always picks the most likely one.
        top_k: Keeps only the k most likely tokens. 0 turns the filter off.
        top_p: Keeps the smallest set of most likely tokens whose probabilities add up to
            at least top_p (nucleus sampling). 1 turns the filter off.
        min_p: Drops tokens whose probability is below min_p times that of the most likely
            token. 0 turns the filter off.

    Raises:
        ValueError: If a value is outside its range.
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.temperature) and self.temperature >= 0):
            raise ValueError(f"temperature must be 0 or greater, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be 0 or greater, got {self.top_k}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if not 0 <= self.min_p <= 1:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")
