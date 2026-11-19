"""Static against continuous batching, on requests whose outputs differ in length.

All requests arrive at once. Static batching takes them in arrival order, `batch_size` at a
time, and runs each batch until its longest output ends. Continuous batching keeps up to
`batch_size` requests in the batch and starts a waiting one as soon as another ends.

Run from the repository root:

    uv run python -m benchmarks.batching --model HuggingFaceTB/SmolLM2-135M-Instruct
"""

import argparse
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import torch

from benchmarks.common import (
    RAW_DIR,
    capture_environment,
    median,
    percentile,
    save_result,
    synchronize,
)
from benchmarks.latency import build_prompt
from kiyal.config import EngineConfig
from kiyal.device import resolve_device, resolve_dtype
from kiyal.engine.generator import TextGenerator
from kiyal.engine.llm_engine import LLMEngine
from kiyal.engine.sequence import SamplingParams
from kiyal.model.llama import LlamaModel
from kiyal.model.loader import load_model, resolve_model_dir
from kiyal.tokenizer import Tokenizer

ENGINES = ("static", "continuous")
DEFAULT_BATCH_SIZES = (8, 16, 32)


@dataclass(frozen=True, slots=True)
class Workload:
    """Requests with seeded random prompt and output lengths.

    Attributes:
        prompts: Token ids of each prompt.
        output_tokens: Number of tokens generated for each request.
    """

    prompts: list[list[int]]
    output_tokens: list[int]

    @property
    def params(self) -> list[SamplingParams]:
        # Greedy decoding that ignores EOS gives every request exactly its output length
        # and uses no random numbers, so every run does the same work.
        return [
            SamplingParams(max_tokens=n, temperature=0, ignore_eos=True) for n in self.output_tokens
        ]

    @property
    def max_len(self) -> int:
        """The most prompt and output tokens of any request."""
        return max(len(p) + n for p, n in zip(self.prompts, self.output_tokens, strict=True))


@dataclass(frozen=True, slots=True)
class Settings:
    """What a batching run measures, and how often.

    Attributes:
        num_requests: Requests in the workload.
        prompt_tokens: Smallest and largest prompt length, both included.
        output_tokens: Smallest and largest output length, both included.
        seed: Seed of the random lengths.
        batch_sizes: Batch sizes to measure, in order. For continuous batching this is
            `max_num_seqs`.
        max_num_batched_tokens: Token budget of a continuous batching step.
        repeats: Timed runs per engine and batch size.
        warmup: Untimed runs per engine and batch size, before the timed ones.
    """

    num_requests: int
    prompt_tokens: tuple[int, int]
    output_tokens: tuple[int, int]
    seed: int
    batch_sizes: tuple[int, ...]
    max_num_batched_tokens: int
    repeats: int
    warmup: int


@dataclass(frozen=True, slots=True)
class RunTiming:
    """One run of the whole workload. Times count from the start of the run, when every
    request has arrived.

    Attributes:
        total_s: Time to the last token of the last request.
        ttft_s: Time to each request's first token.
        e2e_s: Time to each request's last token.
        output_tokens: Tokens generated over all requests.
    """

    total_s: float
    ttft_s: list[float]
    e2e_s: list[float]
    output_tokens: int

    @property
    def output_tokens_per_s(self) -> float:
        return self.output_tokens / self.total_s

    @property
    def requests_per_s(self) -> float:
        return len(self.e2e_s) / self.total_s

    def summary(self) -> dict[str, float]:
        return {
            "total_s": self.total_s,
            "output_tokens_per_s": self.output_tokens_per_s,
            "requests_per_s": self.requests_per_s,
            "ttft_p50_s": percentile(self.ttft_s, 50),
            "ttft_p99_s": percentile(self.ttft_s, 99),
            "e2e_p50_s": percentile(self.e2e_s, 50),
            "e2e_p99_s": percentile(self.e2e_s, 99),
        }


def build_workload(
    tokenizer: Tokenizer,
    num_requests: int,
    prompt_tokens: tuple[int, int],
    output_tokens: tuple[int, int],
    seed: int,
) -> Workload:
    """Draws each request's prompt and output length uniformly from the given ranges.

    Prompt i starts i tokens into the same text, so no two prompts are equal.
    """
    generator = random.Random(seed)
    text = build_prompt(tokenizer, prompt_tokens[1] + num_requests)
    prompts: list[list[int]] = []
    lengths: list[int] = []
    for index in range(num_requests):
        prompt_len = generator.randint(*prompt_tokens)
        prompts.append(text[index : index + prompt_len])
        lengths.append(generator.randint(*output_tokens))
    return Workload(prompts, lengths)


def static_decode_utilization(workload: Workload, batch_size: int) -> float:
    """Returns the share of decoding slots that static batching spends on unfinished outputs.

    A static batch runs every row for as many steps as its longest output; the rest of a
    shorter row's steps compute tokens that are thrown away.
    """
    used = 0
    computed = 0
    for start in range(0, len(workload.output_tokens), batch_size):
        lengths = workload.output_tokens[start : start + batch_size]
        used += sum(lengths)
        computed += len(lengths) * max(lengths)
    return used / computed


class _Clock:
    """Records each request's first and last token time, from the start of a run."""

    def __init__(self, num_requests: int, device: torch.device) -> None:
        self._device = device
        self.ttft_s: list[float | None] = [None] * num_requests
        self.e2e_s: list[float | None] = [None] * num_requests
        self.output_tokens = 0
        synchronize(device)
        self._start = time.perf_counter()

    def token(self, index: int, now: float, is_last: bool) -> None:
        if self.ttft_s[index] is None:
            self.ttft_s[index] = now
        if is_last:
            self.e2e_s[index] = now
        self.output_tokens += 1

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def finish(self, workload: Workload) -> RunTiming:
        synchronize(self._device)
        total = self.elapsed()
        expected = sum(workload.output_tokens)
        ttft = [t for t in self.ttft_s if t is not None]
        e2e = [t for t in self.e2e_s if t is not None]
        if self.output_tokens != expected or len(e2e) != len(self.e2e_s):
            raise ValueError(
                f"generated {self.output_tokens} of {expected} tokens; the model's context "
                f"is too short for this workload"
            )
        return RunTiming(total, ttft, e2e, self.output_tokens)


def time_static(
    generator: TextGenerator, workload: Workload, batch_size: int, device: torch.device
) -> RunTiming:
    """Runs the workload `batch_size` requests at a time, each batch until it ends."""
    params = workload.params
    clock = _Clock(len(params), device)
    for start in range(0, len(params), batch_size):
        indices = range(start, min(start + batch_size, len(params)))
        prompts = [workload.prompts[i] for i in indices]
        # The generator reads each step's tokens back from the device before yielding, so
        # the step's work has finished by the time the clock is read.
        for steps in generator.generate_batch(prompts, [params[i] for i in indices]):
            now = clock.elapsed()
            for index, step in zip(indices, steps, strict=True):
                if step is not None:
                    clock.token(index, now, step.finish_reason is not None)
    return clock.finish(workload)


def time_continuous(engine: LLMEngine, workload: Workload, device: torch.device) -> RunTiming:
    """Adds every request at once and steps the engine until all have ended."""
    params = workload.params
    clock = _Clock(len(params), device)
    index = {
        engine.add_request(prompt, prompt_params): i
        for i, (prompt, prompt_params) in enumerate(zip(workload.prompts, params, strict=True))
    }
    while engine.has_unfinished():
        outputs = engine.step()
        now = clock.elapsed()
        for output in outputs:
            clock.token(index[output.request_id], now, output.finish_reason is not None)
    return clock.finish(workload)


def measure(
    model: LlamaModel, tokenizer: Tokenizer, settings: Settings, device: torch.device
) -> list[dict[str, object]]:
    """Times both engines at each batch size, after untimed runs of the same workload.

    Returns:
        One entry per batch size with every run and the median of each metric per engine.
    """
    workload = build_workload(
        tokenizer,
        settings.num_requests,
        settings.prompt_tokens,
        settings.output_tokens,
        settings.seed,
    )
    static = TextGenerator(model, tokenizer)
    results: list[dict[str, object]] = []
    for batch_size in settings.batch_sizes:
        # One pool row per request in the batch, each long enough for the longest request.
        config = EngineConfig(
            max_num_seqs=batch_size,
            max_num_batched_tokens=settings.max_num_batched_tokens,
            max_model_len=workload.max_len,
        )
        continuous = LLMEngine(model, tokenizer, config)
        runners: dict[str, Callable[[], RunTiming]] = {
            "static": partial(time_static, static, workload, batch_size, device),
            "continuous": partial(time_continuous, continuous, workload, device),
        }
        engines: dict[str, object] = {}
        for engine, run in runners.items():
            for _ in range(settings.warmup):
                run()
            summaries = [run().summary() for _ in range(settings.repeats)]
            engines[engine] = {
                "runs": summaries,
                "median": {key: median([s[key] for s in summaries]) for key in summaries[0]},
            }
        results.append(
            {
                "batch_size": batch_size,
                "static_decode_utilization": static_decode_utilization(workload, batch_size),
                "engines": engines,
            }
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compares static and continuous batching.")
    parser.add_argument("--model", required=True, help="checkpoint directory or Hub repo id")
    parser.add_argument("--revision", help="branch, tag or commit of a Hub repo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument(
        "--prompt-tokens", type=int, nargs=2, default=(32, 256), metavar=("MIN", "MAX")
    )
    parser.add_argument(
        "--output-tokens", type=int, nargs=2, default=(16, 256), metavar=("MIN", "MAX")
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of the random lengths")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=EngineConfig().max_num_batched_tokens,
        help="token budget of a continuous batching step",
    )
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per engine and size")
    parser.add_argument("--warmup", type=int, default=1, help="untimed runs per engine and size")
    parser.add_argument("--output", type=Path, help="result file (default: benchmarks/raw/)")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model_dir = resolve_model_dir(args.model, revision=args.revision)
    tokenizer = Tokenizer.from_dir(model_dir)
    model = load_model(model_dir, device=device, dtype=dtype)

    settings = Settings(
        num_requests=args.num_requests,
        prompt_tokens=tuple(args.prompt_tokens),
        output_tokens=tuple(args.output_tokens),
        seed=args.seed,
        batch_sizes=tuple(args.batch_sizes),
        max_num_batched_tokens=args.max_num_batched_tokens,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    results = measure(model, tokenizer, settings, device)
    result: dict[str, object] = {
        "benchmark": "batching",
        "model": args.model,
        "revision": args.revision,
        "num_requests": settings.num_requests,
        "prompt_tokens": list(settings.prompt_tokens),
        "output_tokens": list(settings.output_tokens),
        "seed": settings.seed,
        "max_num_batched_tokens": settings.max_num_batched_tokens,
        "repeats": settings.repeats,
        "warmup": settings.warmup,
        "environment": capture_environment(device, dtype),
        "results": results,
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = save_result(result, args.output or RAW_DIR / f"batching-{device.type}-{timestamp}.json")

    print(
        f"{'batch size':>10}  {'engine':>10}  {'tokens/s':>8}  {'p50 E2E s':>9}  {'p99 E2E s':>9}"
    )
    for entry in results:
        engines = entry["engines"]
        assert isinstance(engines, dict)
        for engine in ENGINES:
            summary = engines[engine]["median"]
            print(
                f"{entry['batch_size']:>10}  {engine:>10}  {summary['output_tokens_per_s']:>8.1f}  "
                f"{summary['e2e_p50_s']:>9.2f}  {summary['e2e_p99_s']:>9.2f}"
            )
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
