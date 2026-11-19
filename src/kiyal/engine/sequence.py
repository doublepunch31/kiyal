import math
import time
from collections import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

import torch

from kiyal.model.config import ModelConfig
from kiyal.tokenizer import IncrementalDetokenizer, Tokenizer

# The range the OpenAI API accepts for its presence and frequency penalties.
MAX_ADDITIVE_PENALTY: Final = 2.0
MAX_SEED: Final = 2**63 - 1
# One prompt token and one output token.
MIN_SEQUENCE_LEN: Final = 2


class FinishReason(Enum):
    """Why a request stopped generating."""

    # An end-of-sequence token, a stop token or a stop string.
    STOP = "stop"
    # The max_tokens limit or the end of the model's context.
    LENGTH = "length"
    # The caller cancelled the request.
    ABORT = "abort"


class SequenceStatus(Enum):
    """Where a sequence is in the engine."""

    # Added, and waiting for room in the batch.
    WAITING = "waiting"
    # In the batch, with a row of the KV pool, until its output ends.
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplingParams:
    """How one request picks its tokens and when it stops.

    Attributes:
        max_tokens: The most tokens to generate.
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
        stop: Strings that end the output. The output stops before the first one generated
            and does not include it.
        stop_token_ids: Tokens that end the output, in addition to end-of-sequence tokens.
        ignore_eos: Keeps generating after an end-of-sequence token, up to max_tokens.

    Raises:
        ValueError: If a value is outside its range.
    """

    max_tokens: int = 16
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    stop: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be 1 or greater, got {self.max_tokens}")
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
        # A single string would be read as one stop string per character.
        if isinstance(self.stop, str) or not all(self.stop):
            raise ValueError(f"stop must be a tuple of non-empty strings, got {self.stop!r}")
        if any(token_id < 0 for token_id in self.stop_token_ids):
            raise ValueError(f"stop_token_ids must be 0 or greater, got {self.stop_token_ids}")


@dataclass(slots=True, kw_only=True)
class Sequence:
    """One request: its tokens, its place in the engine, and when its output ends.

    Attributes:
        request_id: Name of the request, unique within an engine.
        prompt_token_ids: Tokens of the prompt.
        params: Sampling settings of the request.
        max_tokens: `params.max_tokens`, lowered where needed so that the prompt and the
            output fit in the context.
        stop_token_ids: Tokens that end the output: the requested stop tokens, plus the
            end-of-sequence tokens unless `params.ignore_eos` is set.
        detokenizer: Turns the output tokens into text and watches for stop strings.
        generator: The request's own random generator when it has a seed, else None.
        output_token_ids: Tokens generated so far.
        status: Where the sequence is in the engine.
        finish_reason: Why the output ended, None until it has.
        num_computed_tokens: Tokens whose keys and values are in the KV cache.
        cache_row: Row of the KV pool while the sequence runs, None otherwise.
        arrival_time: When the request was added, from `time.perf_counter()`, like the
            other times.
        first_scheduled_time: When the sequence first entered the batch.
        first_token_time: When its first output token was added.
        last_token_time: When its latest output token was added.
        finished_time: When its output ended.
    """

    request_id: str
    prompt_token_ids: list[int]
    params: SamplingParams
    max_tokens: int
    stop_token_ids: frozenset[int]
    detokenizer: IncrementalDetokenizer
    generator: torch.Generator | None
    output_token_ids: list[int] = field(default_factory=list[int])
    status: SequenceStatus = SequenceStatus.WAITING
    finish_reason: FinishReason | None = None
    num_computed_tokens: int = 0
    cache_row: int | None = None
    arrival_time: float = field(default_factory=time.perf_counter)
    first_scheduled_time: float | None = None
    first_token_time: float | None = None
    last_token_time: float | None = None
    finished_time: float | None = None

    @property
    def num_tokens(self) -> int:
        """Prompt and output tokens."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status is SequenceStatus.FINISHED

    def uncomputed_token_ids(self) -> list[int]:
        """Tokens whose keys and values are not in the KV cache yet, in order.

        That is the prompt before the sequence first runs, then the latest output token.
        """
        num_prompt = len(self.prompt_token_ids)
        if self.num_computed_tokens < num_prompt:
            return self.prompt_token_ids[self.num_computed_tokens :] + self.output_token_ids
        return self.output_token_ids[self.num_computed_tokens - num_prompt :]

    def append(self, token_id: int) -> str:
        """Adds a generated token and ends the output if a stop condition is met.

        Returns:
            Text to add to the output. Empty when the token ends inside a character or in
            a possible stop string, and for a stop token.
        """
        now = time.perf_counter()
        if self.first_token_time is None:
            self.first_token_time = now
        self.last_token_time = now
        self.output_token_ids.append(token_id)
        if token_id in self.stop_token_ids:
            self.finish(FinishReason.STOP)
            # The stop token itself adds no text.
            return self.detokenizer.finish()
        text = self.detokenizer.add(token_id)
        if self.detokenizer.stopped:
            self.finish(FinishReason.STOP)
            return text
        if len(self.output_token_ids) == self.max_tokens:
            self.finish(FinishReason.LENGTH)
            return text + self.detokenizer.finish()
        return text

    def finish(self, reason: FinishReason) -> None:
        self.status = SequenceStatus.FINISHED
        self.finish_reason = reason
        self.finished_time = time.perf_counter()


@dataclass(frozen=True, slots=True)
class RequestOutput:
    """What one engine step added to the output of a request.

    Attributes:
        request_id: The request.
        token_id: The token generated in this step.
        text: Text to append to the output. Empty when the token ends inside a character or
            in a possible stop string, and for a stop token.
        finish_reason: Set on the request's last output, None before it.
    """

    request_id: str
    token_id: int
    text: str
    finish_reason: FinishReason | None


class SequenceFactory:
    """Turns prompts and sampling settings into sequences for one model and tokenizer.

    Both engines start their sequences here, so they apply the same limits and stop
    conditions.
    """

    def __init__(self, config: ModelConfig, tokenizer: Tokenizer, max_len: int) -> None:
        """Prepares sequences of at most `max_len` prompt and output tokens.

        Raises:
            ValueError: If the tokenizer has more tokens than the model, or `max_len` is
                outside [2, max_position_embeddings].
        """
        if tokenizer.vocab_size > config.vocab_size:
            raise ValueError(
                f"the tokenizer has {tokenizer.vocab_size} tokens, "
                f"but the model only has {config.vocab_size}"
            )
        if not MIN_SEQUENCE_LEN <= max_len <= config.max_position_embeddings:
            raise ValueError(
                f"max_len must be in [{MIN_SEQUENCE_LEN}, {config.max_position_embeddings}], "
                f"got {max_len}"
            )
        self._tokenizer = tokenizer
        self._max_len = max_len
        # config.json and the tokenizer can each name end-of-sequence tokens; any of them ends
        # the output.
        eos_token_ids = set(config.eos_token_id)
        if tokenizer.eos_token_id is not None:
            eos_token_ids.add(tokenizer.eos_token_id)
        self._eos_token_ids = frozenset(eos_token_ids)

    def create(
        self,
        request_id: str,
        prompt_token_ids: abc.Sequence[int],
        params: SamplingParams,
        name: str = "the prompt",
    ) -> Sequence:
        """Starts a sequence that has not run yet.

        Args:
            request_id: Name of the request.
            prompt_token_ids: Tokens of the prompt.
            params: Sampling settings.
            name: How error messages refer to the prompt.

        Raises:
            ValueError: If the prompt is empty or leaves no room for an output token.
        """
        if not prompt_token_ids:
            raise ValueError(f"{name} has no tokens")
        num_prompt = len(prompt_token_ids)
        if num_prompt >= self._max_len:
            raise ValueError(
                f"{name} has {num_prompt} tokens, but the context of {self._max_len} tokens "
                f"must also fit at least one output token"
            )
        stop_token_ids = set(params.stop_token_ids)
        if not params.ignore_eos:
            stop_token_ids |= self._eos_token_ids
        seed = params.seed
        return Sequence(
            request_id=request_id,
            prompt_token_ids=list(prompt_token_ids),
            params=params,
            max_tokens=min(params.max_tokens, self._max_len - num_prompt),
            stop_token_ids=frozenset(stop_token_ids),
            detokenizer=IncrementalDetokenizer(self._tokenizer, prompt_token_ids, params.stop),
            generator=None if seed is None else torch.Generator().manual_seed(seed),
        )
