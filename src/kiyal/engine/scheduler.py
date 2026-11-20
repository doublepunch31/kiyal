import time
from collections import deque
from dataclasses import dataclass

from kiyal.cache.block_manager import BlockManager
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

        Sequences that finished since the last step leave the batch first, and their KV
        blocks become free.

        Returns:
            Prefills if a waiting sequence fits, otherwise a decode of every running
            sequence that keeps its blocks. Empty when nothing is left to run.
        """
        self._release_finished()
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
                self._release_finished()
                return seq
        return None

    def _admit(self) -> list[Sequence]:
        """Starts waiting sequences in order while they fit, and returns them."""
        prefills: list[Sequence] = []
        budget = self._config.max_num_batched_tokens
        while self._waiting and len(self._running) < self._config.max_num_seqs:
            seq = self._waiting[0]
            num_tokens = seq.num_tokens - seq.num_computed_tokens
            # Only a preempted sequence can be longer than the whole budget, since it
            # reruns its output too. It starts alone: without chunked prefill it could
            # never start otherwise.
            if num_tokens > budget and prefills:
                break
            if not self._blocks.can_allocate(seq.num_tokens):
                break
            self._waiting.popleft()
            seq.block_table = self._blocks.allocate(seq.num_tokens)
            seq.status = SequenceStatus.RUNNING
            if seq.first_scheduled_time is None:
                seq.first_scheduled_time = time.perf_counter()
            self._running.append(seq)
            prefills.append(seq)
            budget -= num_tokens
        return prefills

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
        seq.status = SequenceStatus.PREEMPTED
        # Ahead of the sequences that arrived later. Of several preempted in one step, the
        # newest goes first and so ends up behind the others.
        self._waiting.appendleft(seq)
        self._num_preemptions += 1

    def _release_finished(self) -> None:
        still_running: list[Sequence] = []
        for seq in self._running:
            if seq.is_finished:
                self._blocks.free(seq.block_table)
            else:
                still_running.append(seq)
        self._running = still_running
