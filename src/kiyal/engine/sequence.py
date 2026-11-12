import math
from dataclasses import dataclass
from typing import Final

# The range the OpenAI API accepts for its presence and frequency penalties.
MAX_ADDITIVE_PENALTY: Final = 2.0
MAX_SEED: Final = 2**63 - 1


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
        repetition_penalty: Makes tokens that occur in the prompt or the output less likely
            when above 1, by dividing their positive logits and multiplying their negative
            ones. 1 turns it off.
        presence_penalty: Subtracted from the logit of every token generated so far.
        frequency_penalty: Subtracted from a token's logit once for each time it has been
            generated.
        seed: Seeds the request's own random generator, so that the same seed and prompt
            give the same output. None draws from a generator shared between requests.

    Raises:
        ValueError: If a value is outside its range.
    """

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.temperature) and self.temperature >= 0):
            raise ValueError(f"temperature must be 0 or greater, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be 0 or greater, got {self.top_k}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if not 0 <= self.min_p <= 1:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")
        if not (math.isfinite(self.repetition_penalty) and self.repetition_penalty > 0):
            raise ValueError(f"repetition_penalty must be above 0, got {self.repetition_penalty}")
        limit = MAX_ADDITIVE_PENALTY
        if not -limit <= self.presence_penalty <= limit:
            raise ValueError(
                f"presence_penalty must be in [-{limit}, {limit}], got {self.presence_penalty}"
            )
        if not -limit <= self.frequency_penalty <= limit:
            raise ValueError(
                f"frequency_penalty must be in [-{limit}, {limit}], got {self.frequency_penalty}"
            )
        if self.seed is not None and not 0 <= self.seed <= MAX_SEED:
            raise ValueError(f"seed must be in [0, {MAX_SEED}], got {self.seed}")
