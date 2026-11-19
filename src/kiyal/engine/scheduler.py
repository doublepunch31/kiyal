import time
from collections import deque
from dataclasses import dataclass

from kiyal.config import EngineConfig
from kiyal.engine.sequence import FinishReason, Sequence, SequenceStatus


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    """The sequences that one engine step runs through the model.

    Attributes:
        prefills: Sequences that start in this step; the model runs on their prompts.
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
    - Two budgets bound a step: `max_num_seqs` sequences in the batch (one KV pool row
      each) and `max_num_batched_tokens` prompt tokens started together.

    The scheduler hands out pool rows and takes them back. It never runs the model and
    does not add tokens; the engine does both and reports finished sequences through their
    status.
    """

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._waiting: deque[Sequence] = deque()
        self._running: list[Sequence] = []
        # A stack whose top is row 0, so rows are handed out in order.
        self._free_rows = list(reversed(range(config.max_num_seqs)))

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

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

        Sequences that finished since the last step leave the batch first, and their pool
        rows become free.

        Returns:
            Prefills if a waiting sequence fits, otherwise a decode of every running
            sequence. Empty when nothing is left to run.
        """
        self._release_finished()
        prefills: list[Sequence] = []
        budget = self._config.max_num_batched_tokens
        # A free row means the batch is below max_num_seqs.
        while self._waiting and self._free_rows:
            seq = self._waiting[0]
            num_tokens = seq.num_tokens - seq.num_computed_tokens
            if num_tokens > budget:
                break
            self._waiting.popleft()
            seq.cache_row = self._free_rows.pop()
            seq.status = SequenceStatus.RUNNING
            seq.first_scheduled_time = time.perf_counter()
            self._running.append(seq)
            prefills.append(seq)
            budget -= num_tokens
        if prefills:
            return SchedulerOutput(prefills=prefills, decodes=[])
        return SchedulerOutput(prefills=[], decodes=list(self._running))

    def abort(self, request_id: str) -> Sequence | None:
        """Ends a waiting or running request at once and frees its pool row.

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

    def _release_finished(self) -> None:
        still_running: list[Sequence] = []
        for seq in self._running:
            if seq.is_finished:
                assert seq.cache_row is not None
                self._free_rows.append(seq.cache_row)
                seq.cache_row = None
            else:
                still_running.append(seq)
        self._running = still_running
