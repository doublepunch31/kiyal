"""Baseline servers that run one batch of requests at a time.

`--engine hf` generates with transformers' `generate()`; `--engine static` with Kiyal's
static batching (`TextGenerator`, M4). A worker thread takes up to `--batch-size` waiting
requests and generates until every output in the batch has ended. Requests that arrive
meanwhile wait for the whole batch. With a batch size of 1, requests run one after
another, which is what a simple server around `generate()` does.

Both engines are served by Kiyal's own app, so the HTTP handling, the streaming and the
stop conditions are those of the real server, and a comparison measures the engines.

Run from the repository root:

    uv run python -m benchmarks.hf_baseline --model HuggingFaceTB/SmolLM2-135M-Instruct \\
        --engine hf --port 8001
"""

import argparse
import asyncio
import contextlib
import logging
import queue
import threading
from collections import abc, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
import uvicorn
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.generation.streamers import BaseStreamer

from kiyal.device import resolve_device, resolve_dtype
from kiyal.engine.async_engine import EngineStats, EngineStoppedError
from kiyal.engine.generator import TextGenerator
from kiyal.engine.sequence import (
    FinishReason,
    RequestOutput,
    SamplingParams,
    Sequence,
    SequenceFactory,
)
from kiyal.model.config import ModelConfig
from kiyal.model.loader import load_model, resolve_model_dir
from kiyal.server.app import build_app
from kiyal.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

ENGINES = ("hf", "static")

type StepCallback = abc.Callable[[list[int]], None]


class BatchRunner(Protocol):
    """Generates for a batch of sequences until every one of them has finished."""

    def check(self, params: SamplingParams) -> None:
        """Raises ValueError if the runner cannot honor the settings."""
        ...

    def __call__(self, batch: list[Sequence], on_step: StepCallback) -> None:
        """Runs the batch, calling `on_step` with the new token of every row after each
        step. The callback appends the tokens to the sequences, which decide when each
        output ends; tokens of rows that have ended are ignored."""
        ...


class HFModel(Protocol):
    """The members of a transformers model that HFRunner uses. transformers' type hints
    for its models do not resolve them."""

    @property
    def device(self) -> torch.device: ...

    @property
    def generation_config(self) -> GenerationConfig: ...

    def generate(self, **kwargs: object) -> object: ...


class HFRunner:
    """Runs a batch with transformers' `generate()`, greedily, with left padding."""

    def __init__(self, model: HFModel, pad_token_id: int) -> None:
        self._model = model
        self._pad_token_id = pad_token_id
        # The sequences decide when an output ends, with the same rules as Kiyal's engine.
        # With an end-of-sequence id, generate() would stop rows itself and pad them from
        # there on, also for requests that ignore EOS. Cleared on the model's own config:
        # None in the config given to generate() falls back to the model's value.
        model.generation_config.eos_token_id = None

    def check(self, params: SamplingParams) -> None:
        defaults = SamplingParams()
        penalties = (params.repetition_penalty, params.presence_penalty, params.frequency_penalty)
        if params.temperature != 0 or penalties != (
            defaults.repetition_penalty,
            defaults.presence_penalty,
            defaults.frequency_penalty,
        ):
            raise ValueError("the transformers baseline only decodes greedily, without penalties")

    def __call__(self, batch: list[Sequence], on_step: StepCallback) -> None:
        longest = max(len(seq.prompt_token_ids) for seq in batch)
        input_ids = torch.full((len(batch), longest), self._pad_token_id)  # [B, T]
        attention_mask = torch.zeros((len(batch), longest), dtype=torch.long)  # [B, T]
        for row, seq in enumerate(batch):
            start = longest - len(seq.prompt_token_ids)
            input_ids[row, start:] = torch.tensor(seq.prompt_token_ids)
            attention_mask[row, start:] = 1
        config = GenerationConfig(
            max_new_tokens=max(seq.max_tokens for seq in batch),
            do_sample=False,
            pad_token_id=self._pad_token_id,
        )
        device = self._model.device
        self._model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            generation_config=config,
            streamer=_StepStreamer(on_step),
            stopping_criteria=StoppingCriteriaList([_AllFinished(batch)]),
        )


class _StepStreamer(BaseStreamer):
    """Passes each step's new tokens on; generate() calls it once with the prompts first."""

    def __init__(self, on_step: StepCallback) -> None:
        self._on_step = on_step
        self._seen_prompts = False

    def put(self, value: torch.Tensor) -> None:
        if not self._seen_prompts:
            self._seen_prompts = True
            return
        self._on_step([int(token_id) for token_id in value])  # [B], on the CPU already

    def end(self) -> None:
        pass


class _AllFinished(StoppingCriteria):
    """Stops generate() once every sequence of the batch has finished, which it checks
    after the streamer has passed the step's tokens on."""

    def __init__(self, batch: list[Sequence]) -> None:
        self._batch = batch

    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        done = all(seq.is_finished for seq in self._batch)
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


class StaticRunner:
    """Runs a batch with Kiyal's static batching."""

    def __init__(self, generator: TextGenerator) -> None:
        self._generator = generator

    def check(self, params: SamplingParams) -> None:
        pass

    def __call__(self, batch: list[Sequence], on_step: StepCallback) -> None:
        steps = self._generator.generate_batch(
            [seq.prompt_token_ids for seq in batch], [seq.params for seq in batch]
        )
        for step in steps:
            # The generator ends each row at the same token as the sequence does; 0 fills
            # the rows it has ended.
            on_step([0 if row is None else row.token_id for row in step])
            # Also stops a batch whose remaining requests were all aborted.
            if all(seq.is_finished for seq in batch):
                return


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


class BatchAtATimeEngine:
    """Serves requests one batch at a time, with the interface of Kiyal's AsyncEngine.

    A worker thread takes up to `batch_size` waiting requests in arrival order and runs
    them until all their outputs have ended. Each step's outputs go to the event loop as
    in AsyncEngine. An aborted request stops getting outputs, but its row keeps running
    until the batch ends, as in static batching.
    """

    def __init__(
        self, runner: BatchRunner, sequences: SequenceFactory, max_model_len: int, batch_size: int
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be 1 or greater, got {batch_size}")
        self._runner = runner
        self._sequences = sequences
        self._max_model_len = max_model_len
        self._batch_size = batch_size
        self._commands: queue.SimpleQueue[_Command] = queue.SimpleQueue()
        self._streams: dict[str, asyncio.Queue[RequestOutput | Exception]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stopped = False
        # Worker thread only.
        self._stop_requested = False
        self._waiting: deque[Sequence] = deque()
        self._batch: list[Sequence] = []
        self._num_aborted = 0
        self._stats = self._read_stats()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and not self._stopped

    @property
    def max_model_len(self) -> int:
        return self._max_model_len

    def stats(self) -> EngineStats:
        return self._stats

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("the engine has been started before")
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._run, name="baseline-engine", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        self._commands.put(_Stop())
        self._thread.join()

    async def generate(
        self, request_id: str, prompt_token_ids: abc.Sequence[int], params: SamplingParams
    ) -> abc.AsyncGenerator[RequestOutput, None]:
        if not self.is_running:
            raise EngineStoppedError("the engine is not running")
        if request_id in self._streams:
            raise ValueError(f"request {request_id} is already running")
        stream: asyncio.Queue[RequestOutput | Exception] = asyncio.Queue()
        self._streams[request_id] = stream
        self._commands.put(_Add(request_id, list(prompt_token_ids), params))
        ended = False
        try:
            while True:
                item = await stream.get()
                if isinstance(item, Exception):
                    ended = True
                    raise item
                yield item
                if item.finish_reason is not None:
                    ended = True
                    return
        finally:
            del self._streams[request_id]
            if not ended:
                self._commands.put(_Abort(request_id))

    def _run(self) -> None:
        error: Exception | None = None
        try:
            while True:
                # Blocks for a command only when nothing is waiting.
                self._apply_commands(block=not self._waiting)
                if self._stop_requested:
                    break
                count = min(self._batch_size, len(self._waiting))
                self._batch = [self._waiting.popleft() for _ in range(count)]
                self._publish_stats()
                if self._batch:
                    self._runner(self._batch, self._on_step)
                    self._batch = []
                if self._stop_requested:
                    break
        except Exception as e:
            logger.exception("the baseline engine failed")
            error = e
        self._stopped = True
        with contextlib.suppress(RuntimeError):
            self._post(self._fail_all, error)

    def _on_step(self, tokens: list[int]) -> None:
        outputs: list[RequestOutput] = []
        for seq, token_id in zip(self._batch, tokens, strict=True):
            if not seq.is_finished:
                text = seq.append(token_id)
                outputs.append(RequestOutput(seq.request_id, token_id, text, seq.finish_reason))
        self._post(self._deliver, outputs)
        self._apply_commands(block=False)
        self._publish_stats()

    def _apply_commands(self, *, block: bool) -> None:
        commands: list[_Command] = [self._commands.get()] if block else []
        while True:
            try:
                commands.append(self._commands.get_nowait())
            except queue.Empty:
                break
        for command in commands:
            match command:
                case _Add():
                    self._add(command)
                case _Abort():
                    self._abort(command.request_id)
                case _Stop():
                    self._stop_requested = True
                    # Ends the running batch at its next step.
                    for seq in self._batch:
                        if not seq.is_finished:
                            seq.finish(FinishReason.ABORT)

    def _add(self, command: _Add) -> None:
        try:
            self._runner.check(command.params)
            seq = self._sequences.create(
                command.request_id, command.prompt_token_ids, command.params
            )
        except ValueError as e:
            self._post(self._reject, command.request_id, e)
            return
        self._waiting.append(seq)

    def _abort(self, request_id: str) -> None:
        for seq in [*self._waiting, *self._batch]:
            if seq.request_id == request_id and not seq.is_finished:
                seq.finish(FinishReason.ABORT)
                if seq in self._waiting:
                    self._waiting.remove(seq)
                self._num_aborted += 1
                return

    def _publish_stats(self) -> None:
        self._stats = self._read_stats()

    def _read_stats(self) -> EngineStats:
        # These engines keep their KV caches to themselves; one block, always free, keeps
        # the app's KV usage metric at 0.
        return EngineStats(
            num_running=sum(not seq.is_finished for seq in self._batch),
            num_waiting=len(self._waiting),
            num_kv_blocks=1,
            num_free_kv_blocks=1,
            num_preemptions=0,
            num_aborted=self._num_aborted,
            num_prefix_query_tokens=0,
            num_prefix_hit_tokens=0,
        )

    def _post[*Ts](self, callback: abc.Callable[[*Ts], object], *args: *Ts) -> None:
        assert self._loop is not None
        self._loop.call_soon_threadsafe(callback, *args)

    def _deliver(self, outputs: list[RequestOutput]) -> None:
        for output in outputs:
            stream = self._streams.get(output.request_id)
            if stream is not None:
                stream.put_nowait(output)

    def _reject(self, request_id: str, error: ValueError) -> None:
        stream = self._streams.get(request_id)
        if stream is not None:
            stream.put_nowait(error)

    def _fail_all(self, cause: Exception | None) -> None:
        for stream in self._streams.values():
            error = EngineStoppedError("the engine has stopped")
            error.__cause__ = cause
            stream.put_nowait(error)


def build_runner(
    engine: str, model_dir: Path, device: torch.device, dtype: torch.dtype
) -> BatchRunner:
    """Loads a model for one of the baseline engines.

    Raises:
        ValueError: If the engine name is unknown.
    """
    tokenizer = Tokenizer.from_dir(model_dir)
    if engine == "hf":
        loaded = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
        hf_model = cast(HFModel, cast(torch.nn.Module, loaded).to(device).eval())
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        return HFRunner(hf_model, pad_token_id)
    if engine == "static":
        model = load_model(model_dir, device=device, dtype=dtype)
        return StaticRunner(TextGenerator(model, tokenizer))
    raise ValueError(f"unknown engine {engine!r}; expected one of {', '.join(ENGINES)}")


def build_engine(
    runner: BatchRunner, model_dir: Path, batch_size: int, max_model_len: int
) -> BatchAtATimeEngine:
    """Wraps a runner in an engine for the model in `model_dir`, with Kiyal's request checks
    and stop conditions."""
    config = ModelConfig.from_file(model_dir / "config.json")
    max_len = min(max_model_len, config.max_position_embeddings)
    sequences = SequenceFactory(config, Tokenizer.from_dir(model_dir), max_len)
    return BatchAtATimeEngine(runner, sequences, max_len, batch_size)


def main(argv: abc.Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serves a baseline engine one batch at a time.")
    parser.add_argument("--model", required=True, help="checkpoint directory or Hub repo id")
    parser.add_argument("--revision", help="branch, tag or commit of a Hub repo")
    parser.add_argument("--engine", choices=ENGINES, default="hf")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=1, help="most requests per batch")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", help="name in the model field (default: --model)")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model_dir = resolve_model_dir(args.model, revision=args.revision)
    runner = build_runner(args.engine, model_dir, device, dtype)
    engine = build_engine(runner, model_dir, args.batch_size, args.max_model_len)
    app = build_app(engine, Tokenizer.from_dir(model_dir), args.served_model_name or args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
