"""Time to first token with and without prefix caching, for requests that share a long
system prompt.

Every request is the same system prompt followed by a short user turn of random tokens. With
prefix caching, a request shares the KV blocks of the system prompt once an earlier request
has computed them, and its prefill runs only on the rest of its prompt. Two arrival patterns:

- one at a time: each request is added when the one before it has ended, so its time to
  first token (TTFT) is the prefill of its prompt and nothing else.
- all at once: every request arrives at the start, so a request's TTFT also counts the wait
  for the prompts ahead of it.

Each run starts a new engine, so the cache starts empty and the first request of a run
computes the system prompt. The KV pool holds every request without sharing, so memory never
limits the runs without the cache.

Run from the repository root:

    uv run python -m benchmarks.prefix --model HuggingFaceTB/SmolLM2-135M-Instruct
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

from benchmarks.batching import RunTiming, Workload, time_continuous
from benchmarks.common import RAW_DIR, capture_environment, median, save_result, synchronize
from benchmarks.latency import build_prompt
from kiyal.config import EngineConfig
from kiyal.device import resolve_device, resolve_dtype
from kiyal.engine.llm_engine import LLMEngine
from kiyal.model.llama import LlamaModel
from kiyal.model.loader import load_model, resolve_model_dir
from kiyal.tokenizer import Tokenizer

ARRIVALS = ("one_at_a_time", "all_at_once")
CACHE_SETTINGS = {"off": False, "on": True}
DEFAULT_SYSTEM_TOKENS = (250, 500, 1000, 2000)


@dataclass(frozen=True, slots=True)
class Settings:
    """What a prefix caching run measures, and how often.

    Attributes:
        system_tokens: Lengths of the shared system prompt to measure, in order.
        num_requests: Requests in the workload.
        user_tokens: Smallest and largest user turn after the system prompt, both included.
        output_tokens: Tokens generated for each request.
        seed: Seed of the user turns.
        max_num_batched_tokens: Token budget of a step. At least the longest prompt.
        repeats: Timed runs per system prompt, arrival pattern and cache setting.
        warmup: Untimed runs before the timed ones, each with a new engine as well.
    """

    system_tokens: tuple[int, ...]
    num_requests: int
    user_tokens: tuple[int, int]
    output_tokens: int
    seed: int
    max_num_batched_tokens: int
    repeats: int
    warmup: int


def build_workload(tokenizer: Tokenizer, system_tokens: int, settings: Settings) -> Workload:
    """Returns requests made of one system prompt of `system_tokens` tokens and a user turn
    of random tokens each.

    The user turns have seeded random lengths and tokens, so no two requests share more
    than the system prompt.
    """
    generator = random.Random(settings.seed)
    system = build_prompt(tokenizer, system_tokens)
    prompts: list[list[int]] = []
    for _ in range(settings.num_requests):
        length = generator.randint(*settings.user_tokens)
        prompts.append(system + [generator.randrange(tokenizer.vocab_size) for _ in range(length)])
    return Workload(prompts, [settings.output_tokens] * settings.num_requests)


def engine_config(workload: Workload, max_num_batched_tokens: int, cache: bool) -> EngineConfig:
    """Returns the engine settings of one run: room in the batch and the pool for every
    request at once, with or without prefix caching.

    Raises:
        ValueError: If the longest prompt does not fit in one step.
    """
    longest_prompt = max(len(prompt) for prompt in workload.prompts)
    if longest_prompt > max_num_batched_tokens:
        raise ValueError(
            f"prompts of up to {longest_prompt} tokens need a token budget of at least that, "
            f"got {max_num_batched_tokens}"
        )
    block_size = EngineConfig().block_size
    num_requests = len(workload.prompts)
    return EngineConfig(
        max_num_seqs=num_requests,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=workload.max_len,
        block_size=block_size,
        num_kv_blocks=num_requests * -(-workload.max_len // block_size),
        enable_prefix_caching=cache,
    )


def time_one_at_a_time(engine: LLMEngine, workload: Workload, device: torch.device) -> RunTiming:
    """Adds each request when the one before it has ended. Its TTFT and E2E count from when
    it is added."""
    ttft: list[float] = []
    e2e: list[float] = []
    synchronize(device)
    start = time.perf_counter()
    for prompt, params in zip(workload.prompts, workload.params, strict=True):
        arrival = time.perf_counter()
        engine.add_request(prompt, params)
        first: float | None = None
        last = arrival
        # The engine reads each step's tokens back from the device, so the step's work has
        # finished by the time the clock is read.
        while engine.has_unfinished():
            engine.step()
            last = time.perf_counter()
            if first is None:
                first = last
        assert first is not None
        ttft.append(first - arrival)
        e2e.append(last - arrival)
    synchronize(device)
    return RunTiming(time.perf_counter() - start, ttft, e2e, sum(workload.output_tokens))


def run_once(
    model: LlamaModel,
    tokenizer: Tokenizer,
    workload: Workload,
    config: EngineConfig,
    timer: Callable[[LLMEngine], RunTiming],
) -> dict[str, float]:
    """Times the workload on a new engine, so the run starts with an empty cache.

    `prefix_hit_rate` is the share of the prompt tokens whose keys and values came from the
    cache.
    """
    engine = LLMEngine(model, tokenizer, config)
    timing = timer(engine)
    prompt_tokens = sum(len(prompt) for prompt in workload.prompts)
    return timing.summary() | {
        "ttft_mean_s": sum(timing.ttft_s) / len(timing.ttft_s),
        "prefix_hit_rate": engine.num_prefix_hit_tokens / prompt_tokens,
        "preemptions": engine.num_preemptions,
    }


def measure(
    model: LlamaModel, tokenizer: Tokenizer, settings: Settings, device: torch.device
) -> list[dict[str, object]]:
    """Times both arrival patterns with the cache off and on, at each system prompt length.

    Returns:
        One entry per system prompt length with every run and the median of each metric per
        arrival pattern and cache setting.
    """
    results: list[dict[str, object]] = []
    for system_tokens in settings.system_tokens:
        workload = build_workload(tokenizer, system_tokens, settings)
        timers: dict[str, Callable[[LLMEngine], RunTiming]] = {
            "one_at_a_time": partial(time_one_at_a_time, workload=workload, device=device),
            "all_at_once": partial(time_continuous, workload=workload, device=device),
        }
        arrivals: dict[str, object] = {}
        for arrival in ARRIVALS:
            caches: dict[str, object] = {}
            for name, cache in CACHE_SETTINGS.items():
                config = engine_config(workload, settings.max_num_batched_tokens, cache)
                run = partial(run_once, model, tokenizer, workload, config, timers[arrival])
                for _ in range(settings.warmup):
                    run()
                summaries = [run() for _ in range(settings.repeats)]
                caches[name] = {
                    "runs": summaries,
                    "median": {key: median([s[key] for s in summaries]) for key in summaries[0]},
                }
            arrivals[arrival] = caches
        results.append(
            {
                "system_tokens": system_tokens,
                "max_len": workload.max_len,
                "arrivals": arrivals,
            }
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measures time to first token with and without prefix caching."
    )
    parser.add_argument("--model", required=True, help="checkpoint directory or Hub repo id")
    parser.add_argument("--revision", help="branch, tag or commit of a Hub repo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--system-tokens", type=int, nargs="+", default=DEFAULT_SYSTEM_TOKENS,
        help="lengths of the shared system prompt",
    )  # fmt: skip
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument(
        "--user-tokens", type=int, nargs=2, default=(16, 64), metavar=("MIN", "MAX")
    )
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0, help="seed of the user turns")
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=4096,
        help="token budget of a step, at least the longest prompt",
    )  # fmt: skip
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per setting")
    parser.add_argument("--warmup", type=int, default=1, help="untimed runs per setting")
    parser.add_argument("--output", type=Path, help="result file (default: benchmarks/raw/)")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model_dir = resolve_model_dir(args.model, revision=args.revision)
    tokenizer = Tokenizer.from_dir(model_dir)
    model = load_model(model_dir, device=device, dtype=dtype)

    settings = Settings(
        system_tokens=tuple(args.system_tokens),
        num_requests=args.num_requests,
        user_tokens=tuple(args.user_tokens),
        output_tokens=args.output_tokens,
        seed=args.seed,
        max_num_batched_tokens=args.max_num_batched_tokens,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    results = measure(model, tokenizer, settings, device)
    result: dict[str, object] = {
        "benchmark": "prefix",
        "model": args.model,
        "revision": args.revision,
        "num_requests": settings.num_requests,
        "user_tokens": list(settings.user_tokens),
        "output_tokens": settings.output_tokens,
        "seed": settings.seed,
        "block_size": EngineConfig().block_size,
        "max_num_batched_tokens": settings.max_num_batched_tokens,
        "repeats": settings.repeats,
        "warmup": settings.warmup,
        "environment": capture_environment(device, dtype),
        "results": results,
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = save_result(result, args.output or RAW_DIR / f"prefix-{device.type}-{timestamp}.json")

    print(
        f"{'system':>6}  {'arrival':>13}  {'cache':>5}  {'p50 TTFT ms':>11}  "
        f"{'p99 TTFT ms':>11}  {'tokens/s':>8}  {'hit rate':>8}"
    )
    for entry in results:
        arrivals = entry["arrivals"]
        assert isinstance(arrivals, dict)
        for arrival in ARRIVALS:
            for name in CACHE_SETTINGS:
                summary = arrivals[arrival][name]["median"]
                p50_ms, p99_ms = summary["ttft_p50_s"] * 1000, summary["ttft_p99_s"] * 1000
                print(
                    f"{entry['system_tokens']:>6}  {arrival:>13}  {name:>5}  {p50_ms:>11.1f}  "
                    f"{p99_ms:>11.1f}  {summary['output_tokens_per_s']:>8.1f}  "
                    f"{summary['prefix_hit_rate']:>8.1%}"
                )
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
