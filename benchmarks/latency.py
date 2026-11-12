"""Single-request latency: decoding speed at several output lengths.

Run from the repository root:

    uv run python -m benchmarks.latency --model HuggingFaceTB/SmolLM2-135M-Instruct
"""

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from benchmarks.common import RAW_DIR, capture_environment, median, save_result, synchronize
from kiyal.device import resolve_device, resolve_dtype
from kiyal.engine.naive import NaiveGenerator
from kiyal.engine.sequence import SamplingParams
from kiyal.model.loader import load_model, resolve_model_dir
from kiyal.tokenizer import Tokenizer

# Repeated and cut to the requested number of tokens. Ordinary prose keeps the measured work
# close to a real request; the output itself does not matter because decoding is greedy with
# a fixed length.
PROMPT_TEXT = (
    "An inference server keeps many requests moving through one model at the same time. "
    "Each request waits in a queue, is admitted when there is memory for its tokens, and "
    "then produces one token per step until it reaches a stop condition. The time to the "
    "first token depends on the length of the prompt, and the time between later tokens "
    "depends on how much work each step repeats. "
)


@dataclass(frozen=True, slots=True)
class RunTiming:
    """Timing of one generation run.

    Attributes:
        output_tokens: Number of tokens generated.
        total_s: Time from the start of the request to the last token.
        ttft_s: Time from the start of the request to the first token.
    """

    output_tokens: int
    total_s: float
    ttft_s: float

    @property
    def tpot_s(self) -> float:
        """Average time per output token after the first."""
        return (self.total_s - self.ttft_s) / (self.output_tokens - 1)

    @property
    def output_tokens_per_s(self) -> float:
        return self.output_tokens / self.total_s


def build_prompt(tokenizer: Tokenizer, num_tokens: int) -> list[int]:
    """Returns a prompt of exactly `num_tokens` tokens."""
    token_ids = tokenizer.encode(PROMPT_TEXT)
    repeats = -(-num_tokens // len(token_ids))
    return (token_ids * repeats)[:num_tokens]


def time_generation(
    generator: NaiveGenerator, prompt: Sequence[int], output_tokens: int, device: torch.device
) -> RunTiming:
    """Generates exactly `output_tokens` tokens greedily and times the run.

    Raises:
        ValueError: If fewer than 2 tokens are requested or the context is too short.
    """
    if output_tokens < 2:
        raise ValueError("the time per output token needs at least 2 output tokens")
    # Greedy decoding that ignores EOS always produces the requested number of tokens and
    # uses no random numbers, so every run does the same work.
    params = SamplingParams(max_tokens=output_tokens, temperature=0, ignore_eos=True)
    synchronize(device)
    start = time.perf_counter()
    ttft = 0.0
    num_tokens = 0
    for _ in generator.generate(prompt, params):
        # The generator reads each token id back from the device before yielding, so the
        # token's work has finished by the time this line runs.
        if num_tokens == 0:
            ttft = time.perf_counter() - start
        num_tokens += 1
    synchronize(device)
    total = time.perf_counter() - start
    if num_tokens != output_tokens:
        raise ValueError(
            f"generated {num_tokens} of {output_tokens} tokens; the model's context is too "
            f"short for a {len(prompt)}-token prompt and this output length"
        )
    return RunTiming(num_tokens, total, ttft)


def measure(
    generator: NaiveGenerator,
    prompt: Sequence[int],
    output_lengths: Sequence[int],
    device: torch.device,
    repeats: int,
) -> list[dict[str, object]]:
    """Times each output length `repeats` times.

    Returns:
        One entry per output length with every run and the median of each metric.
    """
    results: list[dict[str, object]] = []
    for output_tokens in output_lengths:
        runs = [time_generation(generator, prompt, output_tokens, device) for _ in range(repeats)]
        results.append(
            {
                "output_tokens": output_tokens,
                "runs": [{"total_s": run.total_s, "ttft_s": run.ttft_s} for run in runs],
                "median": {
                    "total_s": median([run.total_s for run in runs]),
                    "ttft_s": median([run.ttft_s for run in runs]),
                    "tpot_s": median([run.tpot_s for run in runs]),
                    "output_tokens_per_s": median([run.output_tokens_per_s for run in runs]),
                },
            }
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measures single-request decoding speed.")
    parser.add_argument("--model", required=True, help="checkpoint directory or Hub repo id")
    parser.add_argument("--revision", help="branch, tag or commit of a Hub repo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per output length")
    parser.add_argument("--warmup", type=int, default=1, help="untimed runs before measuring")
    parser.add_argument("--output", type=Path, help="result file (default: benchmarks/raw/)")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model_dir = resolve_model_dir(args.model, revision=args.revision)
    tokenizer = Tokenizer.from_dir(model_dir)
    generator = NaiveGenerator(load_model(model_dir, device=device, dtype=dtype), tokenizer)
    prompt = build_prompt(tokenizer, args.prompt_tokens)

    # Untimed runs first: the first passes through a model pay for one-time setup such as
    # memory allocation and, on GPUs, kernel selection.
    for _ in range(args.warmup):
        time_generation(generator, prompt, min(args.output_tokens), device)
    results = measure(generator, prompt, args.output_tokens, device, args.repeats)
    result: dict[str, object] = {
        "benchmark": "latency",
        "engine": "naive",
        "model": args.model,
        "revision": args.revision,
        "prompt_tokens": len(prompt),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "environment": capture_environment(device, dtype),
        "results": results,
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = save_result(result, args.output or RAW_DIR / f"latency-naive-{timestamp}.json")

    print(f"{'output tokens':>13}  {'tokens/s':>8}  {'TTFT ms':>8}  {'TPOT ms':>8}")
    for entry in results:
        summary = entry["median"]
        assert isinstance(summary, dict)
        print(
            f"{entry['output_tokens']:>13}  {summary['output_tokens_per_s']:>8.1f}  "
            f"{summary['ttft_s'] * 1e3:>8.1f}  {summary['tpot_s'] * 1e3:>8.1f}"
        )
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
