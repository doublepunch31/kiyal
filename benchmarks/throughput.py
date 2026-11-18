"""Static batching throughput: output tokens per second at several batch sizes.

Run from the repository root:

    uv run python -m benchmarks.throughput --model HuggingFaceTB/SmolLM2-135M-Instruct
"""

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from benchmarks.common import RAW_DIR, capture_environment, median, save_result, synchronize
from benchmarks.latency import build_prompt
from kiyal.device import resolve_device, resolve_dtype
from kiyal.engine.generator import TextGenerator
from kiyal.engine.sequence import SamplingParams
from kiyal.model.loader import load_model, resolve_model_dir
from kiyal.tokenizer import Tokenizer

DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)


@dataclass(frozen=True, slots=True)
class Settings:
    """What a throughput run measures, and how often.

    Attributes:
        prompt_tokens: Length of every prompt.
        output_tokens: Number of tokens generated for every prompt.
        batch_sizes: Batch sizes to measure, in order.
        repeats: Timed runs per batch size.
        warmup: Untimed runs per batch size, before the timed ones.
    """

    prompt_tokens: int
    output_tokens: int
    batch_sizes: tuple[int, ...]
    repeats: int
    warmup: int


@dataclass(frozen=True, slots=True)
class BatchTiming:
    """Timing of one batch, run until every row has its output.

    Attributes:
        batch_size: Number of prompts in the batch.
        output_tokens: Number of tokens generated for each prompt.
        total_s: Time from the start of the batch to the last token.
        ttft_s: Time from the start of the batch to the first token of every row, which is
            the prefill of all prompts.
    """

    batch_size: int
    output_tokens: int
    total_s: float
    ttft_s: float

    @property
    def step_s(self) -> float:
        """Average time of a decoding step, which gives every row its next token.

        This is also each row's time per output token after the first.
        """
        return (self.total_s - self.ttft_s) / (self.output_tokens - 1)

    @property
    def output_tokens_per_s(self) -> float:
        return self.batch_size * self.output_tokens / self.total_s


def build_prompts(tokenizer: Tokenizer, num_tokens: int, batch_size: int) -> list[list[int]]:
    """Returns `batch_size` different prompts of exactly `num_tokens` tokens each.

    Each prompt starts one token further into the same text, so the rows are different
    sequences of equal length and the batch needs no padding.
    """
    text = build_prompt(tokenizer, num_tokens + batch_size - 1)
    return [text[row : row + num_tokens] for row in range(batch_size)]


def time_batch(
    generator: TextGenerator,
    prompts: Sequence[Sequence[int]],
    output_tokens: int,
    device: torch.device,
) -> BatchTiming:
    """Generates exactly `output_tokens` tokens greedily for every prompt and times the batch.

    Raises:
        ValueError: If fewer than 2 tokens are requested or the context is too short.
    """
    if output_tokens < 2:
        raise ValueError("the time per step needs at least 2 output tokens")
    # Greedy decoding that ignores EOS gives every row the requested number of tokens and
    # uses no random numbers, so every run does the same work.
    params = SamplingParams(max_tokens=output_tokens, temperature=0, ignore_eos=True)
    synchronize(device)
    start = time.perf_counter()
    ttft = 0.0
    num_tokens = 0
    for index, steps in enumerate(generator.generate_batch(prompts, [params] * len(prompts))):
        # The generator reads the step's tokens back from the device before yielding, so
        # the step's work has finished by the time this line runs.
        if index == 0:
            ttft = time.perf_counter() - start
        num_tokens += sum(step is not None for step in steps)
    synchronize(device)
    total = time.perf_counter() - start
    if num_tokens != len(prompts) * output_tokens:
        raise ValueError(
            f"generated {num_tokens} of {len(prompts) * output_tokens} tokens; the model's "
            f"context is too short for these prompts and this output length"
        )
    return BatchTiming(len(prompts), output_tokens, total, ttft)


def measure(
    generator: TextGenerator, tokenizer: Tokenizer, settings: Settings, device: torch.device
) -> list[dict[str, object]]:
    """Times each batch size, after untimed runs at that size.

    Returns:
        One entry per batch size with every run and the median of each metric.
    """
    results: list[dict[str, object]] = []
    output_tokens = settings.output_tokens
    for batch_size in settings.batch_sizes:
        prompts = build_prompts(tokenizer, settings.prompt_tokens, batch_size)
        # Each batch size has its own tensor shapes, so the warm-up runs at every size: the
        # first pass at a shape pays for memory allocation and, on GPUs, kernel selection.
        for _ in range(settings.warmup):
            time_batch(generator, prompts, output_tokens, device)
        runs = [
            time_batch(generator, prompts, output_tokens, device) for _ in range(settings.repeats)
        ]
        results.append(
            {
                "batch_size": batch_size,
                "runs": [{"total_s": run.total_s, "ttft_s": run.ttft_s} for run in runs],
                "median": {
                    "total_s": median([run.total_s for run in runs]),
                    "ttft_s": median([run.ttft_s for run in runs]),
                    "step_s": median([run.step_s for run in runs]),
                    "output_tokens_per_s": median([run.output_tokens_per_s for run in runs]),
                },
            }
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measures static batching throughput.")
    parser.add_argument("--model", required=True, help="checkpoint directory or Hub repo id")
    parser.add_argument("--revision", help="branch, tag or commit of a Hub repo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per batch size")
    parser.add_argument("--warmup", type=int, default=1, help="untimed runs per batch size")
    parser.add_argument("--output", type=Path, help="result file (default: benchmarks/raw/)")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model_dir = resolve_model_dir(args.model, revision=args.revision)
    tokenizer = Tokenizer.from_dir(model_dir)
    model = load_model(model_dir, device=device, dtype=dtype)
    generator = TextGenerator(model, tokenizer)

    settings = Settings(
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        batch_sizes=tuple(args.batch_sizes),
        repeats=args.repeats,
        warmup=args.warmup,
    )
    results = measure(generator, tokenizer, settings, device)
    result: dict[str, object] = {
        "benchmark": "throughput",
        "engine": "static",
        "model": args.model,
        "revision": args.revision,
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "environment": capture_environment(device, dtype),
        "results": results,
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"throughput-static-{device.type}-{timestamp}.json"
    path = save_result(result, args.output or RAW_DIR / name)

    print(f"{'batch size':>10}  {'tokens/s':>8}  {'TTFT ms':>8}  {'step ms':>8}")
    for entry in results:
        summary = entry["median"]
        assert isinstance(summary, dict)
        print(
            f"{entry['batch_size']:>10}  {summary['output_tokens_per_s']:>8.1f}  "
            f"{summary['ttft_s'] * 1e3:>8.1f}  {summary['step_s'] * 1e3:>8.1f}"
        )
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
