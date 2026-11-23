import asyncio
import contextlib
import logging
import queue
import threading
from collections import abc
from dataclasses import dataclass

from kiyal.engine.llm_engine import LLMEngine
from kiyal.engine.sequence import RequestOutput, SamplingParams

logger = logging.getLogger(__name__)


class EngineStoppedError(RuntimeError):
    """The engine's step loop is not running, so it cannot serve the request."""


@dataclass(frozen=True, slots=True)
class EngineStats:
    """The state of the engine after its latest step.

    Attributes:
        num_running: Requests in the batch, including those that finished at the last step.
        num_waiting: Requests waiting to start, including preempted ones.
        num_kv_blocks: Blocks of the KV pool.
        num_free_kv_blocks: Blocks that no request holds, cached ones included.
        num_preemptions: Times a request has given up its KV blocks so far.
        num_aborted: Unfinished requests aborted so far.
        num_prefix_query_tokens: Prompt tokens of the requests started so far, while prefix
            caching is enabled.
        num_prefix_hit_tokens: Of those tokens, the ones found in the prefix cache.
    """

    num_running: int
    num_waiting: int
    num_kv_blocks: int
    num_free_kv_blocks: int
    num_preemptions: int
    num_aborted: int
    num_prefix_query_tokens: int
    num_prefix_hit_tokens: int


@dataclass(frozen=True, slots=True)
class _Add:
    request_id: str
    prompt_token_ids: list[int]
    params: SamplingParams


@dataclass(frozen=True, slots=True)
class _Abort:
    request_id: str


@dataclass(frozen=True, slots=True)
class _Stop:
    pass


type _Command = _Add | _Abort | _Stop
# What the engine thread puts into a request's queue: its outputs, or why it cannot run.
type _Item = RequestOutput | Exception


class AsyncEngine:
    """Serves an `LLMEngine` to asyncio code, with the step loop on a thread of its own.

    The event loop only adds requests and reads outputs; it never waits for a step. A
    dedicated thread runs `LLMEngine.step` back to back while there is work, so the batch
    keeps moving however busy the event loop is with HTTP, and a slow step does not stall
    the event loop.

    The two threads share no mutable state:
    - Requests and aborts go to the engine thread through a thread-safe command queue. The
      engine thread takes the queued commands between steps, so the `LLMEngine` is only
      ever touched by that thread.
    - Each request has an `asyncio.Queue` on the event loop. After a step, the engine
      thread hands all of the step's outputs to the event loop in one
      `loop.call_soon_threadsafe` call, which puts each output into its request's queue
      there, so the queues are only ever touched by the event loop.

    Only one thread runs Python code at a time (the GIL), so this does not make Python
    code run in parallel. It helps because PyTorch releases the GIL inside its kernels
    and while waiting for the GPU, and the event loop gets the GIL at least every few
    milliseconds otherwise (the interpreter's switch interval), so HTTP work proceeds
    during the forward pass.
    """

    def __init__(self, engine: LLMEngine) -> None:
        self._engine = engine
        self._commands: queue.SimpleQueue[_Command] = queue.SimpleQueue()
        self._streams: dict[str, asyncio.Queue[_Item]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        # Set by the engine thread before it hands the event loop its last callback.
        self._stopped = False
        self._num_aborted = 0
        self._stats = self._read_stats()

    @property
    def is_running(self) -> bool:
        """Whether the step loop has started and has not stopped."""
        return self._thread is not None and not self._stopped

    @property
    def max_model_len(self) -> int:
        """The most prompt and output tokens one request can hold."""
        return self._engine.max_model_len

    def stats(self) -> EngineStats:
        """The state of the engine after its latest step."""
        # The engine thread replaces the whole object, so a reader never sees a mix of two
        # steps.
        return self._stats

    def start(self) -> None:
        """Starts the step loop. Call it from the event loop that will read the outputs.

        Raises:
            RuntimeError: If the engine has been started before, or no event loop runs.
        """
        if self._thread is not None:
            raise RuntimeError("the engine has been started before")
        self._loop = asyncio.get_running_loop()
        # A daemon thread does not keep the process alive if the server exits without
        # calling shutdown.
        self._thread = threading.Thread(target=self._run, name="kiyal-engine", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        """Stops the step loop after its current step and waits for the thread to end.

        Requests that have not finished end with EngineStoppedError. The engine cannot be
        started again.
        """
        if self._thread is None or not self._thread.is_alive():
            return
        self._commands.put(_Stop())
        self._thread.join()

    async def generate(
        self, request_id: str, prompt_token_ids: abc.Sequence[int], params: SamplingParams
    ) -> abc.AsyncGenerator[RequestOutput, None]:
        """Adds a request and yields its outputs as the engine makes them.

        The request is added when the iteration starts. If the caller stops iterating
        before the last output, or its task is cancelled, the request is aborted and its
        KV blocks are freed.

        Args:
            request_id: Name of the request; it must not belong to a running request.
            prompt_token_ids: Tokens of the prompt.
            params: Sampling settings.

        Raises:
            ValueError: If the request id is in use or the engine rejects the request, for
                example because the prompt does not fit.
            EngineStoppedError: If the step loop is not running or stops.
        """
        if not self.is_running:
            raise EngineStoppedError("the engine is not running")
        if request_id in self._streams:
            raise ValueError(f"request {request_id} is already running")
        stream: asyncio.Queue[_Item] = asyncio.Queue()
        self._streams[request_id] = stream
        self._commands.put(_Add(request_id, list(prompt_token_ids), params))
        ended = False
        try:
            while True:
                item = await stream.get()
                if isinstance(item, Exception):
                    # The engine has dropped the request already.
                    ended = True
                    raise item
                yield item
                if item.finish_reason is not None:
                    ended = True
                    return
        finally:
            # Synchronous on purpose: this also runs while the task is being cancelled,
            # where an await could be cancelled again.
            del self._streams[request_id]
            if not ended:
                self._commands.put(_Abort(request_id))

    def _run(self) -> None:
        error: Exception | None = None
        try:
            self._step_until_stopped()
        except Exception as e:
            logger.exception("the engine loop failed")
            error = e
        self._stopped = True
        # Nothing is listening any more if the event loop has closed.
        with contextlib.suppress(RuntimeError):
            self._post(self._fail_all, error)

    def _step_until_stopped(self) -> None:
        engine = self._engine
        while True:
            # An engine that holds nothing waits for a command. Otherwise it takes only the
            # commands already queued, so the running requests keep stepping. Requests that
            # finished at the last step still hold their blocks until the next step frees
            # them, so the engine steps once more before it waits.
            commands: list[_Command] = []
            if not engine.has_unfinished() and engine.num_running == 0:
                commands.append(self._commands.get())
            commands.extend(self._queued_commands())
            for command in commands:
                match command:
                    case _Add():
                        self._add(command)
                    case _Abort():
                        if engine.abort(command.request_id):
                            self._num_aborted += 1
                    case _Stop():
                        return
            outputs = engine.step()
            self._stats = self._read_stats()
            if outputs:
                self._post(self._deliver, outputs)

    def _queued_commands(self) -> list[_Command]:
        commands: list[_Command] = []
        while True:
            try:
                commands.append(self._commands.get_nowait())
            except queue.Empty:
                return commands

    def _add(self, command: _Add) -> None:
        try:
            self._engine.add_request(command.prompt_token_ids, command.params, command.request_id)
        except ValueError as e:
            self._post(self._reject, command.request_id, e)

    def _post[*Ts](self, callback: abc.Callable[[*Ts], object], *args: *Ts) -> None:
        assert self._loop is not None, "start() sets the loop before the thread runs"
        self._loop.call_soon_threadsafe(callback, *args)

    def _read_stats(self) -> EngineStats:
        engine = self._engine
        return EngineStats(
            num_running=engine.num_running,
            num_waiting=engine.num_waiting,
            num_kv_blocks=engine.num_kv_blocks,
            num_free_kv_blocks=engine.num_free_kv_blocks,
            num_preemptions=engine.num_preemptions,
            num_aborted=self._num_aborted,
            num_prefix_query_tokens=engine.num_prefix_query_tokens,
            num_prefix_hit_tokens=engine.num_prefix_hit_tokens,
        )

    # The methods below run on the event loop's thread.

    def _deliver(self, outputs: list[RequestOutput]) -> None:
        for output in outputs:
            stream = self._streams.get(output.request_id)
            # None for a request whose caller has left; its abort is on the way.
            if stream is not None:
                stream.put_nowait(output)

    def _reject(self, request_id: str, error: ValueError) -> None:
        stream = self._streams.get(request_id)
        if stream is not None:
            stream.put_nowait(error)

    def _fail_all(self, cause: Exception | None) -> None:
        for stream in self._streams.values():
            # One exception per request, since raising it adds that request's traceback.
            error = EngineStoppedError("the engine has stopped")
            error.__cause__ = cause
            stream.put_nowait(error)
