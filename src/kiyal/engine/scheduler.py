import time
from collections import deque
from dataclasses import dataclass

from kiyal.cache.block_manager import BlockManager, block_hash
from kiyal.config import EngineConfig
from kiyal.engine.sequence import FinishReason, Sequence, SequenceStatus


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    """The sequences that one engine step runs through the model.

    Attributes:
        prefills: Sequences that start in this step; the model runs on their prompts, and
            on their outputs so far if they were preempted.
        decodes: Running sequences that get one more token in this step.
    """

    prefills: list[Sequence]
    decodes: list[Sequence]

    @property
    def sequences(self) -> list[Sequence]:
        """All scheduled sequences, prefills first."""
        return self.prefills + self.decodes

    @property
    def is_empty(self) -> bool:
        return not self.prefills and not self.decodes


class Scheduler:
    """Picks the sequences of every engine step (iteration-level scheduling, as in Orca).

    Static batching fixes a batch until its last output ends, so a sequence that ends
    early holds its place, and a new request waits for the whole batch. Here the batch is
    rebuilt at every step: a sequence leaves as soon as its output ends, and a waiting one
    takes its place at the next step (Yu et al., "Orca", OSDI 2022).

    Policy:
    - First come, first served. Waiting sequences start in arrival order, and a sequence
      that does not fit blocks the ones behind it, so a long prompt cannot be passed over
      forever by shorter ones.
    - A step runs either prefills or decodes, never both, and prefills go first: while
      waiting sequences fit, the step starts them, and the running ones wait for one step.
      This keeps each forward pass uniform. The price is a pause in every running output
      while prompts run; chunked prefill, which mixes both, would remove it.
    - Three budgets bound a step: `max_num_seqs` sequences in the batch,
      `max_num_batched_tokens` prompt tokens started together, and the free KV blocks. A
      sequence starts only if blocks for all its tokens are free beyond the watermark.
    - Preemption by recompute. A decode step needs room for one more token of every running
      sequence. When the free blocks run out, the most recently arrived running sequence
      gives up its blocks and goes back to the front of the waiting queue. When it starts
      again, its prompt and its output so far run as one prefill, which computes the lost
      keys and values again. At these model sizes that is cheaper than copying the blocks
      to host memory and back, and it needs no second pool.
    - Prefix caching, when enabled. Once a full block's keys and values are computed, the
      block is cached under the hash of its tokens and all tokens before them. A starting
      sequence shares the cached blocks of its longest matching prefix and runs only the
      tokens after it, which also count against the token budget. A cached block is never
      written to, so its keys and values stay those of the tokens it was cached for.

    The scheduler hands out KV blocks and takes them back. It never runs the model and
    does not add tokens; the engine does both and reports finished sequences through their
    status.
    """

    def __init__(self, config: EngineConfig, block_manager: BlockManager) -> None:
        self._config = config
        self._blocks = block_manager
        self._waiting: deque[Sequence] = deque()
        # In arrival order, so the newest is preempted first. Admissions keep the order:
        # a preempted sequence goes back to the front of the waiting queue, and nothing
        # behind it starts before it does.
        self._running: list[Sequence] = []
        self._num_preemptions = 0
        self._num_prefix_query_tokens = 0
        self._num_prefix_hit_tokens = 0

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    @property
    def num_preemptions(self) -> int:
        """Times a running sequence has given up its blocks so far."""
        return self._num_preemptions

    @property
    def num_prefix_query_tokens(self) -> int:
        """Tokens of the sequences started so far, restarts included, while prefix caching
        is enabled."""
        return self._num_prefix_query_tokens

    @property
    def num_prefix_hit_tokens(self) -> int:
        """Of those tokens, the ones whose keys and values were found in the prefix cache."""
        return self._num_prefix_hit_tokens

    def has_unfinished(self) -> bool:
        return bool(self._waiting) or any(not seq.is_finished for seq in self._running)

    def add(self, seq: Sequence) -> None:
        """Queues a new sequence behind the waiting ones.

        Raises:
            ValueError: If the sequence is not new, or its prompt is longer than one step
                can run.
        """
        if seq.status is not SequenceStatus.WAITING or seq.num_computed_tokens:
            raise ValueError(f"request {seq.request_id} has already started")
        num_prompt = len(seq.prompt_token_ids)
        if num_prompt > self._config.max_num_batched_tokens:
            raise ValueError(
                f"the prompt has {num_prompt} tokens, more than the "
                f"{self._config.max_num_batched_tokens} that one step can run"
            )
        self._waiting.append(seq)

    def schedule(self) -> SchedulerOutput:
        """Picks the sequences of the next step.

        Blocks filled at the last step join the prefix cache first. Then the sequences that
        finished leave the batch, and their KV blocks become free.

        Returns:
            Prefills if a waiting sequence fits, otherwise a decode of every running
            sequence that keeps its blocks. Empty when nothing is left to run.
        """
        self._update_running()
        prefills = self._admit()
        if prefills:
            return SchedulerOutput(prefills=prefills, decodes=[])
        return SchedulerOutput(prefills=[], decodes=self._grow_running())

    def abort(self, request_id: str) -> Sequence | None:
        """Ends a waiting or running request at once and frees its KV blocks.

        Returns:
            The aborted sequence, or None if no unfinished request has this id, for example
            because it has just finished.
        """
        for seq in self._waiting:
            if seq.request_id == request_id:
                self._waiting.remove(seq)
                seq.finish(FinishReason.ABORT)
                return seq
        for seq in self._running:
            if seq.request_id == request_id and not seq.is_finished:
                seq.finish(FinishReason.ABORT)
                self._update_running()
                return seq
        return None

    def _admit(self) -> list[Sequence]:
        """Starts waiting sequences in order while they fit, and returns them."""
        prefills: list[Sequence] = []
        budget = self._config.max_num_batched_tokens
        while self._waiting and len(self._running) < self._config.max_num_seqs:
            seq = self._waiting[0]
            cached_blocks = self._cached_prefix(seq)
            num_cached_tokens = len(cached_blocks) * self._blocks.block_size
            num_new_tokens = seq.num_tokens - num_cached_tokens
            # Only a preempted sequence can be longer than the whole budget, since it
            # reruns its output too. It starts alone: without chunked prefill it could
            # never start otherwise.
            if num_new_tokens > budget and prefills:
                break
            if not self._blocks.can_allocate(seq.num_tokens, cached_blocks):
                break
            self._waiting.popleft()
            seq.block_table = self._blocks.allocate(seq.num_tokens, cached_blocks)
            seq.num_computed_tokens = num_cached_tokens
            seq.num_cached_blocks = len(cached_blocks)
            seq.status = SequenceStatus.RUNNING
            if seq.first_scheduled_time is None:
                seq.first_scheduled_time = time.perf_counter()
            if self._config.enable_prefix_caching:
                self._num_prefix_query_tokens += seq.num_tokens
                self._num_prefix_hit_tokens += num_cached_tokens
            self._running.append(seq)
            prefills.append(seq)
            budget -= num_new_tokens
        return prefills

    def _cached_prefix(self, seq: Sequence) -> list[int]:
        """Returns the cached blocks that hold the start of a waiting sequence's tokens."""
        if not self._config.enable_prefix_caching:
            return []
        # The last token always runs, because its logits give the next token. So the shared
        # prefix ends before it, and the block that the last token is written to is the
        # sequence's own, even when every block before it is cached.
        num_blocks = (seq.num_tokens - 1) // self._blocks.block_size
        return self._blocks.cached_prefix(self._block_hashes(seq, num_blocks)[:num_blocks])

    def _block_hashes(self, seq: Sequence, num_blocks: int) -> list[bytes]:
        """Hashes the sequence's first `num_blocks` full blocks where not done yet, and
        returns all its block hashes."""
        hashes = seq.block_hashes
        if len(hashes) < num_blocks:
            size = self._blocks.block_size
            token_ids = seq.prompt_token_ids + seq.output_token_ids
            for index in range(len(hashes), num_blocks):
                parent = hashes[-1] if hashes else None
                hashes.append(block_hash(parent, token_ids[index * size : (index + 1) * size]))
        return hashes

    def _cache_computed_blocks(self, seq: Sequence) -> None:
        """Offers the sequence's full blocks computed since the last offer to the cache."""
        if not self._config.enable_prefix_caching:
            return
        num_full = seq.num_computed_tokens // self._blocks.block_size
        if num_full <= seq.num_cached_blocks:
            return
        hashes = self._block_hashes(seq, num_full)
        for index in range(seq.num_cached_blocks, num_full):
            self._blocks.cache_block(seq.block_table[index], hashes[index])
        seq.num_cached_blocks = num_full

    def _grow_running(self) -> list[Sequence]:
        """Makes room for one more token of every running sequence, oldest first.

        When the free blocks run out, the newest running sequences are preempted until the
        older ones fit; a sequence that still does not fit is preempted itself.

        Returns:
            The running sequences that keep their blocks, in arrival order.
        """
        queue = deque(self._running)
        kept: list[Sequence] = []
        while queue:
            seq = queue.popleft()
            # The step writes the token added at the last step, so the table must hold
            # every token of the sequence.
            while queue and not self._blocks.can_append(seq.block_table, seq.num_tokens):
                self._preempt(queue.pop())
            if self._blocks.can_append(seq.block_table, seq.num_tokens):
                self._blocks.append_slot(seq.block_table, seq.num_tokens)
                kept.append(seq)
            else:
                self._preempt(seq)
        self._running = kept
        return list(kept)

    def _preempt(self, seq: Sequence) -> None:
        self._blocks.free(seq.block_table)
        seq.num_computed_tokens = 0
        seq.num_cached_blocks = 0
        seq.status = SequenceStatus.PREEMPTED
        # Ahead of the sequences that arrived later. Of several preempted in one step, the
        # newest goes first and so ends up behind the others.
        self._waiting.appendleft(seq)
        self._num_preemptions += 1

    def _update_running(self) -> None:
        """Offers the blocks computed at the last step to the prefix cache, then frees the
        blocks of finished sequences."""
        still_running: list[Sequence] = []
        for seq in self._running:
            # Before freeing, so that a finished or aborted sequence's blocks are cached too.
            self._cache_computed_blocks(seq)
            if seq.is_finished:
                self._blocks.free(seq.block_table)
            else:
                still_running.append(seq)
        self._running = still_running
