from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from kiyal.cache.contiguous import ContiguousKVCache
from kiyal.engine.sequence import FinishReason, SamplingParams
from kiyal.model.llama import LlamaModel
from kiyal.sampling import SamplingTensors, apply_penalties, draw_uniform, sample, token_counts
from kiyal.tokenizer import IncrementalDetokenizer, Tokenizer

# Any valid id works: padding slots are hidden from attention, so their tokens never reach
# the tokens of the prompt or the output.
PADDING_TOKEN_ID = 0


@dataclass(frozen=True, slots=True)
class GenerationStep:
    """One generated token and the output text it completed.

    Attributes:
        token_id: The token generated in this step.
        text: Text to append to the output. Empty when the token ends inside a character or
            in a possible stop string, and for a token that ends the output.
        finish_reason: Set on the last step, None before it.
    """

    token_id: int
    text: str
    finish_reason: FinishReason | None


@dataclass(slots=True)
class _Row:
    """The output of one prompt in a batch, and when it ends."""

    prompt_token_ids: list[int]
    max_tokens: int
    stop_token_ids: frozenset[int]
    detokenizer: IncrementalDetokenizer
    generator: torch.Generator | None
    output_token_ids: list[int] = field(default_factory=list[int])
    finished: bool = False

    def append(self, token_id: int) -> GenerationStep:
        """Adds a generated token to the output and ends it if a stop condition is met."""
        self.output_token_ids.append(token_id)
        if token_id in self.stop_token_ids:
            self.finished = True
            # The stop token itself adds no text.
            return GenerationStep(token_id, self.detokenizer.finish(), FinishReason.STOP)
        text = self.detokenizer.add(token_id)
        if self.detokenizer.stopped:
            self.finished = True
            return GenerationStep(token_id, text, FinishReason.STOP)
        if len(self.output_token_ids) == self.max_tokens:
            self.finished = True
            return GenerationStep(token_id, text + self.detokenizer.finish(), FinishReason.LENGTH)
        return GenerationStep(token_id, text, None)


@dataclass(slots=True)
class _Batch:
    """The rows of a batch and the state they share."""

    rows: list[_Row]
    # The padding, prompt and sampled tokens of each row, one per slot. Rows have equal
    # lengths: padding evens out the prompts, and every row gets a token at every step, also
    # after its output has ended.
    slot_token_ids: list[list[int]]
    left_padding: Tensor  # [B] number of padding slots at the start of each row
    is_padded: bool
    sampling: SamplingTensors
    prompt_counts: Tensor | None  # [B, V], only when a row has penalties
    cache: ContiguousKVCache | None


class TextGenerator:
    """Generates text for a batch of prompts that start together (static batching).

    Prompts of different lengths are padded at the start (left padding), so the newest token
    of every row is in the same column. One forward pass then serves the whole batch, and the
    KV cache keeps a single length for all rows. Padding is hidden from attention and each
    row counts its own positions, so a row gets the same tokens as when it runs alone.

    The batch keeps its shape until its last output ends. A row that finishes early still
    runs through the model at every step and its tokens are thrown away; continuous
    batching removes that waste.

    With the KV cache, the model runs once on the prompts (prefill) and then on one new token
    per row and step (decode), reusing the cached keys and values of the tokens before it.

    Without it, every step runs the model on the whole sequences. Step n then recomputes all
    n tokens so far, so generating n tokens costs on the order of n^2 token computations.
    This mode stays as the reference that cached decoding is tested against, and as the
    benchmark baseline.
    """

    def __init__(
        self, model: LlamaModel, tokenizer: Tokenizer, *, use_kv_cache: bool = True
    ) -> None:
        """Pairs a model with its tokenizer.

        Args:
            model: The model to generate with.
            tokenizer: The model's tokenizer.
            use_kv_cache: Reuse keys and values between steps instead of recomputing the
                whole sequence.

        Raises:
            ValueError: If the tokenizer has more tokens than the model.
        """
        if tokenizer.vocab_size > model.config.vocab_size:
            raise ValueError(
                f"the tokenizer has {tokenizer.vocab_size} tokens, "
                f"but the model only has {model.config.vocab_size}"
            )
        self._model = model
        self._tokenizer = tokenizer
        self._use_kv_cache = use_kv_cache
        self._device = model.embed_tokens.weight.device
        self._dtype = model.embed_tokens.weight.dtype
        # config.json and the tokenizer can each name end-of-sequence tokens; any of them ends
        # the output.
        eos_token_ids = set(model.config.eos_token_id)
        if tokenizer.eos_token_id is not None:
            eos_token_ids.add(tokenizer.eos_token_id)
        self._eos_token_ids = frozenset(eos_token_ids)
        # A new generator starts from a fixed seed; seed() takes one from the OS so that
        # requests without a seed differ from run to run.
        self._shared_generator = torch.Generator()
        self._shared_generator.seed()

    def generate(
        self, prompt_token_ids: Sequence[int], params: SamplingParams
    ) -> Iterator[GenerationStep]:
        """Generates tokens for one prompt until a stop condition, yielding one step per token.

        The output ends at an end-of-sequence or stop token, at a stop string, after
        `params.max_tokens` tokens, or when the model's context is full.

        Raises:
            ValueError: If the prompt is empty or leaves no room in the context.
        """
        for steps in self.generate_batch([prompt_token_ids], [params]):
            step = steps[0]
            # A batch of one ends with its only row, so the row has a step every time.
            assert step is not None
            yield step

    def generate_batch(
        self, prompts: Sequence[Sequence[int]], params: Sequence[SamplingParams]
    ) -> Iterator[list[GenerationStep | None]]:
        """Generates tokens for several prompts at once, yielding one list per step.

        Entry i of each list is the step of prompt i, or None once its output has ended.
        Each output ends as in `generate`, and the batch runs until the last one has ended.

        Args:
            prompts: Token ids of each prompt.
            params: Sampling settings of each prompt.

        Raises:
            ValueError: If there are no prompts, the numbers of prompts and settings differ,
                or a prompt is empty or leaves no room in the context.
        """
        if not prompts:
            raise ValueError("no prompts given")
        if len(params) != len(prompts):
            raise ValueError(f"got {len(prompts)} prompts but {len(params)} sampling settings")
        batch = self._start_batch(prompts, params)
        while not all(row.finished for row in batch.rows):
            token_ids = self._next_tokens(batch)
            steps: list[GenerationStep | None] = []
            for row, slots, token_id in zip(
                batch.rows, batch.slot_token_ids, token_ids, strict=True
            ):
                slots.append(token_id)
                steps.append(None if row.finished else row.append(token_id))
            yield steps

    def _start_batch(
        self, prompts: Sequence[Sequence[int]], params: Sequence[SamplingParams]
    ) -> _Batch:
        max_positions = self._model.config.max_position_embeddings
        rows: list[_Row] = []
        for index, (prompt, row_params) in enumerate(zip(prompts, params, strict=True)):
            name = "the prompt" if len(prompts) == 1 else f"prompt {index}"
            if not prompt:
                raise ValueError(f"{name} has no tokens")
            if len(prompt) >= max_positions:
                raise ValueError(
                    f"{name} has {len(prompt)} tokens, but the model's context of "
                    f"{max_positions} tokens must also fit at least one output token"
                )
            stop_token_ids = set(row_params.stop_token_ids)
            if not row_params.ignore_eos:
                stop_token_ids |= self._eos_token_ids
            seed = row_params.seed
            rows.append(
                _Row(
                    prompt_token_ids=list(prompt),
                    max_tokens=min(row_params.max_tokens, max_positions - len(prompt)),
                    stop_token_ids=frozenset(stop_token_ids),
                    detokenizer=IncrementalDetokenizer(self._tokenizer, prompt, row_params.stop),
                    generator=None if seed is None else torch.Generator().manual_seed(seed),
                )
            )

        longest = max(len(row.prompt_token_ids) for row in rows)
        padding = [longest - len(row.prompt_token_ids) for row in rows]
        cache = None
        if self._use_kv_cache:
            # The last token generated is never fed back to the model, so it needs no slot.
            cache = ContiguousKVCache(
                self._model.config,
                batch_size=len(rows),
                max_tokens=longest + max(row.max_tokens for row in rows) - 1,
                device=self._device,
                dtype=self._dtype,
            )
        sampling = SamplingTensors.from_params(params, self._device)
        prompt_counts = None
        if sampling.has_penalties:
            prompt_counts = token_counts(
                [row.prompt_token_ids for row in rows], self._model.config.vocab_size, self._device
            )
        return _Batch(
            rows=rows,
            slot_token_ids=[
                [PADDING_TOKEN_ID] * num_padding + row.prompt_token_ids
                for row, num_padding in zip(rows, padding, strict=True)
            ],
            left_padding=torch.tensor(padding, device=self._device),
            is_padded=any(padding),
            sampling=sampling,
            prompt_counts=prompt_counts,
            cache=cache,
        )

    @torch.inference_mode()
    def _next_tokens(self, batch: _Batch) -> list[int]:
        # The model runs on the slots that are not cached yet: the padded prompts on the first
        # step (prefill) and the previous step's tokens afterwards (decode). Without a cache
        # nothing is ever cached, so that is every slot on every step.
        num_cached = 0 if batch.cache is None else batch.cache.length
        num_slots = len(batch.slot_token_ids[0])
        input_ids = torch.tensor(
            [slots[num_cached:] for slots in batch.slot_token_ids], device=self._device
        )  # [B, T]
        slots = torch.arange(num_cached, num_slots, device=self._device)  # [T]
        # A token's position counts the tokens before it in its own row, not the padding.
        # Padding slots get position 0, and a finished row can run past the end of the
        # context; the clamp keeps both inside the rotary table, and neither result is used.
        max_position = self._model.config.max_position_embeddings - 1
        positions = (slots - batch.left_padding.unsqueeze(-1)).clamp(0, max_position)  # [B, T]
        logits = self._model(
            input_ids,
            positions,
            cache=batch.cache,
            left_padding=batch.left_padding if batch.is_padded else None,
        )  # [B, V]
        if batch.prompt_counts is not None:
            output_counts = token_counts(
                [row.output_token_ids for row in batch.rows],
                self._model.config.vocab_size,
                self._device,
            )  # [B, V]
            logits = apply_penalties(logits, batch.sampling, batch.prompt_counts, output_counts)
        uniform = draw_uniform([row.generator for row in batch.rows], self._shared_generator)
        token_ids = sample(logits, batch.sampling, uniform)  # [B]
        # One copy from the device per step for the whole batch: the ids are needed on the
        # host to check the stop conditions and to detokenize before the next step can start.
        return [int(token_id) for token_id in token_ids.cpu()]
