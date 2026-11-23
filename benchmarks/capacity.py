"""Contiguous against paged KV memory, under the same memory budget.

All requests arrive at once, with prompt and output lengths drawn as in the batching
benchmark. The contiguous pool reserves the longest request's length for every sequence,
so a budget of M tokens holds M // max_len sequences at once. The paged pool hands out
blocks of 16 tokens as sequences grow, so the same budget holds as many sequences as their
actual lengths fit; when they outgrow it, the newest is preempted and computed again later.

Run from the repository root:

    uv run python -m benchmarks.capacity --model HuggingFaceTB/SmolLM2-135M-Instruct
"""

import argparse
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from benchmarks.batching import Workload, build_workload, time_continuous
from benchmarks.common import RAW_DIR, capture_environment, median, save_result
from kiyal.config import EngineConfig
from kiyal.device import resolve_device, resolve_dtype
from kiyal.engine.llm_engine import LLMEngine
from kiyal.engine.model_runner import kv_block_bytes
from kiyal.model.llama import LlamaModel
from kiyal.model.loader import load_model, resolve_model_dir
from kiyal.tokenizer import Tokenizer

LAYOUTS = ("contiguous", "paged")
DEFAULT_KV_TOKENS = (2048, 4096, 8192)
PAGED_BLOCK_SIZE = EngineConfig().block_size


@dataclass(frozen=True, slots=True)
class Settings:
    """What a capacity run measures, and how often.

    Attributes:
        num_requests: Requests in the workload.
        prompt_tokens: Smallest and largest prompt length, both included.
        output_tokens: Smallest and largest output length, both included.
        seed: Seed of the random lengths.
        kv_tokens: KV memory budgets to measure, in tokens of keys and values, in order.
        max_num_batched_tokens: Token budget of a step.
        repeats: Timed runs per layout and budget.
        warmup: Untimed runs per layout and budget, before the timed ones.
    """

    num_requests: int
    prompt_tokens: tuple[int, int]
    output_tokens: tuple[int, int]
    seed: int
    kv_tokens: tuple[int, ...]
    max_num_batched_tokens: int
    repeats: int
    warmup: int


def engine_config(
    layout: str, kv_tokens: int, workload: Workload, max_num_batched_tokens: int
) -> EngineConfig:
    """Returns the engine settings of a KV layout under a budget of `kv_tokens` tokens.

    The contiguous layout gives each sequence one block as long as the longest request, as
    the pool of continuous batching did before paging, and admits as many sequences as such
    blocks fit. The paged layout has blocks of 16 tokens and room in the batch for every
    request, so that memory is its only limit. Both run without prefix caching, which would
    let every run after the first reuse the prompts of the one before, and a preempted
    request reuse its own blocks.

    Raises:
        ValueError: If the layout is unknown, or the budget cannot hold the longest request.
    """
    max_len = workload.max_len
    if kv_tokens < max_len:
        raise ValueError(f"a budget of {kv_tokens} tokens cannot hold a request of {max_len}")
    if layout == "contiguous":
        num_seqs = kv_tokens // max_len
        return EngineConfig(
            max_num_seqs=num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_len,
            block_size=max_len,
            num_kv_blocks=num_seqs,
            enable_prefix_caching=False,
        )
    if layout == "paged":
        return EngineConfig(
            max_num_seqs=len(workload.prompts),
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_len,
            block_size=PAGED_BLOCK_SIZE,
            num_kv_blocks=kv_tokens // PAGED_BLOCK_SIZE,
            enable_prefix_caching=False,
        )
    raise ValueError(f"unknown layout {layout!r}; expected one of {', '.join(LAYOUTS)}")


def run_once(engine: LLMEngine, workload: Workload, device: torch.device) -> dict[str, float]:
    """Runs the workload once and summarizes its speed, latency and use of the batch.

    `peak_running` and `mean_running` count the requests that hold KV memory after each
    step: the most at any step, and the average over steps.
    """
    preemptions_before = engine.num_preemptions
    running: list[int] = []
    timing = time_continuous(engine, workload, device, running)
    return timing.summary() | {
        "peak_running": max(running),
        "mean_running": statistics.fmean(running),
        "preemptions": engine.num_preemptions - preemptions_before,
    }


def measure(
    model: LlamaModel, tokenizer: Tokenizer, settings: Settings, device: torch.device
) -> tuple[Workload, list[dict[str, object]]]:
    """Times both layouts at each budget, after untimed runs of the same workload.

    Returns:
        The workload, and one entry per budget with every run and the median of each
        metric per layout.
    """
    workload = build_workload(
        tokenizer,
        settings.num_requests,
        settings.prompt_tokens,
        settings.output_tokens,
        settings.seed,
    )
    token_bytes = kv_block_bytes(model.config, 1, model.embed_tokens.weight.dtype)
    results: list[dict[str, object]] = []
    for kv_tokens in settings.kv_tokens:
        layouts: dict[str, object] = {}
        for layout in LAYOUTS:
            config = engine_config(layout, kv_tokens, workload, settings.max_num_batched_tokens)
            engine = LLMEngine(model, tokenizer, config)
            for _ in range(settings.warmup):
                run_once(engine, workload, device)
            summaries = [run_once(engine, workload, device) for _ in range(settings.repeats)]
            layouts[layout] = {
                "max_num_seqs": config.max_num_seqs,
                "block_size": config.block_size,
                "num_kv_blocks": engine.num_kv_blocks,
                "runs": summaries,
                "median": {key: median([s[key] for s in summaries]) for key in summaries[0]},
            }
        results.append(
            {"kv_tokens": kv_tokens, "kv_bytes": kv_tokens * token_bytes, "layouts": layouts}
        )
    return workload, results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compares contiguous and paged KV memory under the same budget."
    )
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
    parser.add_argument(
        "--kv-tokens",
        type=int,
        nargs="+",
        default=DEFAULT_KV_TOKENS,
        help="KV memory budgets, in tokens",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=EngineConfig().max_num_batched_tokens,
        help="token budget of a step",
    )
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per layout and budget")
    parser.add_argument("--warmup", type=int, default=1, help="untimed runs per layout and budget")
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
        kv_tokens=tuple(args.kv_tokens),
        max_num_batched_tokens=args.max_num_batched_tokens,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    workload, results = measure(model, tokenizer, settings, device)
    result: dict[str, object] = {
        "benchmark": "capacity",
        "model": args.model,
        "revision": args.revision,
        "num_requests": settings.num_requests,
        "prompt_tokens": list(settings.prompt_tokens),
        "output_tokens": list(settings.output_tokens),
        "seed": settings.seed,
        "max_len": workload.max_len,
        "max_num_batched_tokens": settings.max_num_batched_tokens,
        "repeats": settings.repeats,
        "warmup": settings.warmup,
        "environment": capture_environment(device, dtype),
        "results": results,
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = save_result(result, args.output or RAW_DIR / f"capacity-{device.type}-{timestamp}.json")

    print(
        f"{'KV tokens':>9}  {'layout':>10}  {'peak seqs':>9}  {'mean seqs':>9}  "
        f"{'preempted':>9}  {'tokens/s':>8}  {'p99 E2E s':>9}"
    )
    for entry in results:
        layouts = entry["layouts"]
        assert isinstance(layouts, dict)
        for layout in LAYOUTS:
            summary = layouts[layout]["median"]
            print(
                f"{entry['kv_tokens']:>9}  {layout:>10}  {summary['peak_running']:>9.0f}  "
                f"{summary['mean_running']:>9.1f}  {summary['preemptions']:>9.0f}  "
                f"{summary['output_tokens_per_s']:>8.1f}  {summary['e2e_p99_s']:>9.2f}"
            )
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
