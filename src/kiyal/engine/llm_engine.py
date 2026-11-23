import itertools
from collections import abc
from dataclasses import dataclass

import torch

from kiyal.cache.block_manager import BlockManager
from kiyal.config import EngineConfig
from kiyal.engine.model_runner import ModelRunner, kv_blocks_for_memory
from kiyal.engine.scheduler import Scheduler
from kiyal.engine.sequence import (
    FinishReason,
    RequestOutput,
    SamplingParams,
    Sequence,
    SequenceFactory,
)
from kiyal.model.llama import LlamaModel
from kiyal.sampling import SamplingTensors, apply_penalties, draw_uniform, sample, token_counts
from kiyal.tokenizer import Tokenizer


@dataclass(frozen=True, slots=True)
class Completion:
    """The whole output of one request, as returned by `LLMEngine.generate`."""

    request_id: str
    token_ids: list[int]
    text: str
    finish_reason: FinishReason


class LLMEngine:
    """Generates text for requests that start and end at any time (continuous batching).

    Requests are added with `add_request`, and every call to `step` runs one forward pass
    for the batch that the scheduler picks: the prompts of requests that start, or one new
    token for every running request. A request leaves the batch as soon as its output ends
    and a waiting one takes its place, so no forward pass computes tokens for an output
    that has already ended, and a new request need not wait for a whole batch to finish.

    The tokens of all requests in a step go through the model flattened, one after
    another, so no compute goes to padding outside attention. Their keys and values live in
    a paged pool: each request takes blocks as its output grows, and when the blocks run
    out, the newest request gives its blocks up and is computed again later. With prefix
    caching, a request whose prompt starts with the same tokens as an earlier one shares
    that one's blocks for them and runs only the rest of its prompt.
    """

    def __init__(
        self, model: LlamaModel, tokenizer: Tokenizer, config: EngineConfig | None = None
    ) -> None:
        """Pairs a model with its tokenizer and allocates the KV pool.

        Args:
            model: The model to generate with.
            tokenizer: The model's tokenizer.
            config: Batch limits; the defaults of EngineConfig when None. `max_model_len`
                is lowered to the model's context if that is shorter. Without
                `num_kv_blocks`, the KV pool is sized from memory, which on CUDA runs one
                forward pass of the largest step to measure the memory it needs.

        Raises:
            ValueError: If the tokenizer has more tokens than the model, or the KV pool
                cannot hold one sequence of `max_model_len` tokens next to the watermark.
        """
        config = EngineConfig() if config is None else config
        max_len = min(config.max_model_len, model.config.max_position_embeddings)
        self._sequences = SequenceFactory(model.config, tokenizer, max_len)
        blocks_per_seq = -(-max_len // config.block_size)
        num_blocks = config.num_kv_blocks
        if num_blocks is None:
            # Never more than the batch can use.
            num_blocks = min(
                config.max_num_seqs * blocks_per_seq, kv_blocks_for_memory(model, config, max_len)
            )
        watermark = int(config.watermark * num_blocks)
        # Otherwise a sequence could grow to where it does not fit even alone, and would be
        # preempted and restarted forever.
        if num_blocks - watermark < blocks_per_seq:
            raise ValueError(
                f"a KV pool of {num_blocks} x {config.block_size} tokens with a watermark of "
                f"{watermark} blocks cannot hold one sequence of {max_len} tokens; give the "
                f"pool more blocks or memory, or lower max_model_len"
            )
        self._runner = ModelRunner(model, num_blocks, config.block_size)
        self._blocks = BlockManager(num_blocks, config.block_size, watermark)
        self._scheduler = Scheduler(config, self._blocks)
        self._max_model_len = max_len
        self._vocab_size = model.config.vocab_size
        self._device = model.embed_tokens.weight.device
        # Unfinished requests by id.
        self._requests: dict[str, Sequence] = {}
        self._ids = itertools.count()
        # A new generator starts from a fixed seed; seed() takes one from the OS so that
        # requests without a seed differ from run to run.
        self._shared_generator = torch.Generator()
        self._shared_generator.seed()

    def add_request(
        self,
        prompt_token_ids: abc.Sequence[int],
        params: SamplingParams,
        request_id: str | None = None,
    ) -> str:
        """Queues a request. It starts at a later step, once the batch has room for it.

        Args:
            prompt_token_ids: Tokens of the prompt.
            params: Sampling settings.
            request_id: Name of the request. A new number when None.

        Returns:
            The request id, which names the request in step outputs and in `abort`.

        Raises:
            ValueError: If the id belongs to an unfinished request, or the prompt is empty,
                longer than one step can run, or leaves no room for output.
        """
        if request_id is None:
            # A caller may have taken the next number as its own id.
            request_id = str(next(self._ids))
            while request_id in self._requests:
                request_id = str(next(self._ids))
        elif request_id in self._requests:
            raise ValueError(f"request {request_id} is already running")
        seq = self._sequences.create(request_id, prompt_token_ids, params)
        self._scheduler.add(seq)
        self._requests[request_id] = seq
        return request_id

    def abort(self, request_id: str) -> bool:
        """Ends a request at once and frees its place in the batch.

        Unknown and finished requests are ignored, since a request can finish between the
        caller's decision and this call.

        Returns:
            Whether an unfinished request was aborted.
        """
        if self._scheduler.abort(request_id) is None:
            return False
        del self._requests[request_id]
        return True

    def has_unfinished(self) -> bool:
        return self._scheduler.has_unfinished()

    @property
    def max_model_len(self) -> int:
        """The most prompt and output tokens one request can hold."""
        return self._max_model_len

    @property
    def num_running(self) -> int:
        """Requests in the batch, including those that finished at the last step."""
        return self._scheduler.num_running

    @property
    def num_waiting(self) -> int:
        """Requests waiting to start, including preempted ones."""
        return self._scheduler.num_waiting

    @property
    def num_kv_blocks(self) -> int:
        """Blocks of the KV pool."""
        return self._runner.pool.num_blocks

    @property
    def num_free_kv_blocks(self) -> int:
        """Blocks of the KV pool that no request holds, cached ones included."""
        return self._blocks.num_free_blocks

    @property
    def num_preemptions(self) -> int:
        """Times a request has given up its KV blocks so far."""
        return self._scheduler.num_preemptions

    @property
    def num_prefix_query_tokens(self) -> int:
        """Prompt tokens of the requests started so far, with the outputs rerun after
        preemption, while prefix caching is enabled."""
        return self._scheduler.num_prefix_query_tokens

    @property
    def num_prefix_hit_tokens(self) -> int:
        """Of those tokens, the ones whose keys and values were found in the prefix cache.
        Divided by `num_prefix_query_tokens`, this is the cache's hit rate."""
        return self._scheduler.num_prefix_hit_tokens

    @torch.inference_mode()
    def step(self) -> list[RequestOutput]:
        """Runs one forward pass and samples a token for every request in it.

        Returns:
            One output for each request in the step, empty when nothing is left to run.
        """
        scheduled = self._scheduler.schedule()
        if scheduled.is_empty:
            return []
        sequences = scheduled.sequences
        logits = self._runner.execute(sequences)  # [B, V]
        outputs: list[RequestOutput] = []
        for seq, token_id in zip(sequences, self._sample(sequences, logits), strict=True):
            # Every token the sequence had is in the KV cache now; the new one is not.
            seq.num_computed_tokens = seq.num_tokens
            text = seq.append(token_id)
            outputs.append(RequestOutput(seq.request_id, token_id, text, seq.finish_reason))
            if seq.is_finished:
                del self._requests[seq.request_id]
        return outputs

    def generate(
        self, prompts: abc.Sequence[abc.Sequence[int]], params: abc.Sequence[SamplingParams]
    ) -> list[Completion]:
        """Generates the whole output of several prompts at once, for offline use.

        Args:
            prompts: Token ids of each prompt.
            params: Sampling settings of each prompt.

        Returns:
            The completion of each prompt, in order.

        Raises:
            RuntimeError: If the engine still has unfinished requests, whose outputs would
                be lost.
            ValueError: If the numbers of prompts and settings differ, or a prompt cannot
                be added.
        """
        if self.has_unfinished():
            raise RuntimeError("generate needs an engine without unfinished requests")
        if len(params) != len(prompts):
            raise ValueError(f"got {len(prompts)} prompts but {len(params)} sampling settings")
        request_ids: list[str] = []
        try:
            for prompt, prompt_params in zip(prompts, params, strict=True):
                request_ids.append(self.add_request(prompt, prompt_params))
        except ValueError:
            for request_id in request_ids:
                self.abort(request_id)
            raise
        token_ids: dict[str, list[int]] = {request_id: [] for request_id in request_ids}
        texts: dict[str, list[str]] = {request_id: [] for request_id in request_ids}
        finish_reasons: dict[str, FinishReason] = {}
        while self.has_unfinished():
            for output in self.step():
                token_ids[output.request_id].append(output.token_id)
                texts[output.request_id].append(output.text)
                if output.finish_reason is not None:
                    finish_reasons[output.request_id] = output.finish_reason
        return [
            Completion(i, token_ids[i], "".join(texts[i]), finish_reasons[i]) for i in request_ids
        ]

    def _sample(self, sequences: list[Sequence], logits: torch.Tensor) -> list[int]:
        params = SamplingTensors.from_params([seq.params for seq in sequences], self._device)
        if params.has_penalties:
            # Rebuilt at every step, because the batch changes from step to step.
            prompt_counts = token_counts(
                [seq.prompt_token_ids for seq in sequences], self._vocab_size, self._device
            )  # [B, V]
            output_counts = token_counts(
                [seq.output_token_ids for seq in sequences], self._vocab_size, self._device
            )  # [B, V]
            logits = apply_penalties(logits, params, prompt_counts, output_counts)
        uniform = draw_uniform([seq.generator for seq in sequences], self._shared_generator)
        token_ids = sample(logits, params, uniform)  # [B]
        # One copy from the device per step for the whole batch: the ids are needed on the
        # host to check the stop conditions and to build the next step's inputs.
        return [int(token_id) for token_id in token_ids.cpu()]
