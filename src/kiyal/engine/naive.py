from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch

from kiyal.engine.sequence import FinishReason, SamplingParams
from kiyal.model.llama import LlamaModel
from kiyal.sampling import SamplingTensors, apply_penalties, draw_uniform, sample, token_counts
from kiyal.tokenizer import IncrementalDetokenizer, Tokenizer


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


class NaiveGenerator:
    """Generates text for one request by running the model on the whole sequence each step.

    Step n recomputes all n tokens so far, so generating n tokens costs on the order of n^2
    token computations. A KV cache removes the repeated work. This version stays as the
    reference that the optimized engines are tested against, and as the benchmark baseline.
    """

    def __init__(self, model: LlamaModel, tokenizer: Tokenizer) -> None:
        """Pairs a model with its tokenizer.

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
        self._device = model.embed_tokens.weight.device
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
        """Generates tokens until a stop condition, yielding one step per token.

        The output ends at an end-of-sequence or stop token, at a stop string, after
        `params.max_tokens` tokens, or when the model's context is full.

        Raises:
            ValueError: If the prompt is empty or leaves no room in the context.
        """
        max_positions = self._model.config.max_position_embeddings
        if not prompt_token_ids:
            raise ValueError("the prompt has no tokens")
        if len(prompt_token_ids) >= max_positions:
            raise ValueError(
                f"the prompt has {len(prompt_token_ids)} tokens, but the model's context of "
                f"{max_positions} tokens must also fit at least one output token"
            )
        max_tokens = min(params.max_tokens, max_positions - len(prompt_token_ids))
        stop_token_ids = set(params.stop_token_ids)
        if not params.ignore_eos:
            stop_token_ids |= self._eos_token_ids

        sampling = SamplingTensors.from_params([params], self._device)
        generator = None if params.seed is None else torch.Generator().manual_seed(params.seed)
        detokenizer = IncrementalDetokenizer(self._tokenizer, prompt_token_ids, params.stop)
        token_ids = list(prompt_token_ids)
        for num_generated in range(1, max_tokens + 1):
            token_id = self._next_token(token_ids, len(prompt_token_ids), sampling, generator)
            token_ids.append(token_id)
            if token_id in stop_token_ids:
                # The stop token itself adds no text.
                yield GenerationStep(token_id, detokenizer.finish(), FinishReason.STOP)
                return
            text = detokenizer.add(token_id)
            if detokenizer.stopped:
                yield GenerationStep(token_id, text, FinishReason.STOP)
                return
            if num_generated == max_tokens:
                yield GenerationStep(token_id, text + detokenizer.finish(), FinishReason.LENGTH)
                return
            yield GenerationStep(token_id, text, None)

    @torch.inference_mode()
    def _next_token(
        self,
        token_ids: Sequence[int],
        prompt_len: int,
        sampling: SamplingTensors,
        generator: torch.Generator | None,
    ) -> int:
        input_ids = torch.tensor([token_ids], device=self._device)  # [1, T]
        positions = torch.arange(len(token_ids), device=self._device).unsqueeze(0)  # [1, T]
        logits = self._model(input_ids, positions)  # [1, V]
        if sampling.has_penalties:
            vocab_size = self._model.config.vocab_size
            logits = apply_penalties(
                logits,
                sampling,
                token_counts([token_ids[:prompt_len]], vocab_size, self._device),
                token_counts([token_ids[prompt_len:]], vocab_size, self._device),
            )
        uniform = draw_uniform([generator], self._shared_generator)  # [1]
        # One read from the device per token: the id is needed on the host to check the stop
        # conditions and to detokenize before the next step can start.
        return int(sample(logits, sampling, uniform).item())
