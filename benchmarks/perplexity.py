"""Perplexity on WikiText-2 of a model and its int8 version, and how often they pick the same token.

The test split is joined into one stream of tokens, which sliding windows cover: each window
of `window` tokens starts `stride` tokens after the one before it and scores only the tokens
no earlier window scored, so every scored token sees at least `window - stride` tokens before
it (Hugging Face's "Perplexity of fixed-length models" recipe). Both models run on the same
windows. Perplexity is exp of the mean negative log-likelihood of the scored tokens; top-1
agreement is the share of scored positions where both models' most likely next token is the
same, given the same text before it.

Run from the repository root:

    uv run python -m benchmarks.perplexity --model HuggingFaceTB/SmolLM2-135M-Instruct
"""

import argparse
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from datasets import load_dataset
from torch import Tensor
from torch.nn.functional import cross_entropy

from benchmarks.common import RAW_DIR, capture_environment, save_result, synchronize
from kiyal.device import resolve_device, resolve_dtype
from kiyal.model.llama import LlamaModel
from kiyal.model.loader import load_model, resolve_model_dir
from kiyal.tokenizer import Tokenizer

WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
# Pinned so that a new upload cannot change the text that is evaluated.
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DEFAULT_WINDOW = 1024
DEFAULT_STRIDE = 512


def load_wikitext2(split: str = "test") -> str:
    """Returns a split of WikiText-2 (raw) as one text, downloading it on first use."""
    dataset = load_dataset(WIKITEXT_REPO, WIKITEXT_CONFIG, split=split, revision=WIKITEXT_REVISION)
    # Joined with blank lines, as in Hugging Face's recipe, so that the token stream is the
    # one that evaluations following it see.
    return "\n\n".join(dataset["text"])


@dataclass(frozen=True, slots=True)
class Window:
    """Tokens [start, end) of the stream, of which the last `num_scored` are scored."""

    start: int
    end: int
    num_scored: int


def sliding_windows(num_tokens: int, window: int, stride: int) -> list[Window]:
    """Covers a stream of tokens with windows that score every token but the first once.

    Window k starts at token k * stride and ends `window` tokens later or at the end of the
    stream. It scores the tokens after the ones the windows before it scored, so a scored
    token sees at least `window - stride` tokens of context, except in the first window.

    Raises:
        ValueError: If the stream has fewer than 2 tokens, or the stride is not between 1
            and the window.
    """
    if num_tokens < 2:
        raise ValueError(f"need at least 2 tokens to score one, got {num_tokens}")
    if not 1 <= stride <= window:
        raise ValueError(f"stride must be in [1, window={window}], got {stride}")
    windows: list[Window] = []
    # Token 0 has nothing before it to be predicted from.
    scored_end = 1
    start = 0
    while scored_end < num_tokens:
        end = min(start + window, num_tokens)
        windows.append(Window(start, end, end - scored_end))
        scored_end = end
        start += stride
    return windows


def score_window(model: LlamaModel, token_ids: Tensor, num_scored: int) -> tuple[Tensor, Tensor]:
    """Scores the last `num_scored` of a window's tokens.

    Args:
        model: The model to score with.
        token_ids: [T] tokens of the window, on the model's device.
        num_scored: How many tokens at the end of the window to score, at most T - 1.

    Returns:
        [num_scored] negative log-likelihood of each scored token in fp32, and [num_scored]
        the model's most likely token at each scored position.
    """
    num_tokens = token_ids.shape[0]
    positions = torch.arange(num_tokens, device=token_ids.device)  # [T]
    # The logits at position t predict token t + 1.
    predicting = torch.arange(num_tokens - num_scored - 1, num_tokens - 1, device=token_ids.device)
    logits = model(token_ids.unsqueeze(0), positions.unsqueeze(0), predicting)  # [K, V]
    # In fp32: the log-softmax over 49k entries loses small probabilities in fp16.
    logits = logits.float()
    nll = cross_entropy(logits, token_ids[num_tokens - num_scored :], reduction="none")  # [K]
    return nll, logits.argmax(dim=-1)


@dataclass(slots=True)
class ModelScore:
    """Running totals of one model over the scored tokens.

    Attributes:
        nll_sum: Sum of the scored tokens' negative log-likelihoods, in nats.
        forward_s: Time spent in the model's forward passes.
    """

    nll_sum: float = 0.0
    forward_s: float = 0.0


@dataclass(frozen=True, slots=True)
class Comparison:
    """Perplexity of a reference model and a candidate on the same tokens.

    Attributes:
        num_scored: Scored tokens.
        reference: Totals of the reference model.
        candidate: Totals of the candidate model.
        num_agreed: Scored positions where both models' most likely token is the same.
    """

    num_scored: int
    reference: ModelScore
    candidate: ModelScore
    num_agreed: int

    @property
    def reference_perplexity(self) -> float:
        return math.exp(self.reference.nll_sum / self.num_scored)

    @property
    def candidate_perplexity(self) -> float:
        return math.exp(self.candidate.nll_sum / self.num_scored)

    @property
    def perplexity_increase(self) -> float:
        """The candidate's perplexity relative to the reference's, minus 1."""
        return self.candidate_perplexity / self.reference_perplexity - 1

    @property
    def agreement(self) -> float:
        return self.num_agreed / self.num_scored


def compare(
    reference: LlamaModel,
    candidate: LlamaModel,
    token_ids: Sequence[int],
    windows: Sequence[Window],
    *,
    progress: bool = False,
) -> Comparison:
    """Scores a stream of tokens with two models, window by window.

    Args:
        reference: The model to compare against.
        candidate: The model to compare.
        token_ids: The stream of tokens.
        windows: Parts of the stream to run, from `sliding_windows`.
        progress: Whether to report every 50th window on stderr.

    Raises:
        ValueError: If a window is longer than a model's context.
    """
    longest = max(part.end - part.start for part in windows)
    for model in (reference, candidate):
        if model.config.max_position_embeddings < longest:
            raise ValueError(
                f"window of {longest} tokens exceeds the model's context of "
                f"{model.config.max_position_embeddings}"
            )
    device = reference.embed_tokens.weight.device
    stream = torch.tensor(token_ids, device=device)
    scores = (ModelScore(), ModelScore())
    num_agreed = 0
    with torch.inference_mode():
        for index, part in enumerate(windows):
            picks: list[Tensor] = []
            for model, score in zip((reference, candidate), scores, strict=True):
                synchronize(device)
                start = time.perf_counter()
                nll, argmax = score_window(model, stream[part.start : part.end], part.num_scored)
                score.nll_sum += float(nll.sum())
                score.forward_s += time.perf_counter() - start
                picks.append(argmax)
            num_agreed += int((picks[0] == picks[1]).sum())
            if progress and (index + 1) % 50 == 0:
                print(f"window {index + 1} of {len(windows)}", file=sys.stderr)
    num_scored = sum(part.num_scored for part in windows)
    return Comparison(num_scored, scores[0], scores[1], num_agreed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measures WikiText-2 perplexity of a model and its int8 version."
    )
    parser.add_argument("--model", required=True, help="checkpoint directory or Hub repo id")
    parser.add_argument("--revision", help="branch, tag or commit of a Hub repo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", help="dtype of the reference model")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument(
        "--max-tokens", type=int, help="tokens of the split to use (default: all of them)"
    )
    parser.add_argument("--output", type=Path, help="result file (default: benchmarks/raw/)")
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model_dir = resolve_model_dir(args.model, revision=args.revision)
    tokenizer = Tokenizer.from_dir(model_dir)
    # One stream for the whole split: windows cut across article boundaries, and no BOS is
    # added, as in the recipe.
    token_ids = tokenizer.encode(load_wikitext2(), add_special_tokens=False)
    num_tokens = len(token_ids)
    if args.max_tokens is not None:
        token_ids = token_ids[: args.max_tokens]
    reference = load_model(model_dir, device=device, dtype=dtype)
    candidate = load_model(model_dir, device=device, dtype=dtype, quantization="int8")
    windows = sliding_windows(len(token_ids), args.window, args.stride)
    comparison = compare(reference, candidate, token_ids, windows, progress=True)

    weights = str(dtype).removeprefix("torch.")
    models = {
        name: {
            "perplexity": perplexity,
            "mean_nll": score.nll_sum / comparison.num_scored,
            "forward_s": score.forward_s,
        }
        for name, score, perplexity in [
            (weights, comparison.reference, comparison.reference_perplexity),
            ("int8", comparison.candidate, comparison.candidate_perplexity),
        ]
    }
    result: dict[str, object] = {
        "benchmark": "perplexity",
        "model": args.model,
        "revision": args.revision,
        "dataset": {
            "repo": WIKITEXT_REPO,
            "config": WIKITEXT_CONFIG,
            "split": "test",
            "revision": WIKITEXT_REVISION,
            "tokens": num_tokens,
        },
        "window": args.window,
        "stride": args.stride,
        "tokens": len(token_ids),
        "scored_tokens": comparison.num_scored,
        "environment": capture_environment(device, dtype),
        "results": {
            "reference": weights,
            "models": models,
            "perplexity_increase": comparison.perplexity_increase,
            "top1_agreement": comparison.agreement,
        },
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = save_result(
        result, args.output or RAW_DIR / f"perplexity-{device.type}-{timestamp}.json"
    )

    print(f"{comparison.num_scored} tokens scored, windows of {args.window}, stride {args.stride}")
    for name, summary in models.items():
        perplexity, seconds = summary["perplexity"], summary["forward_s"]
        print(f"{name:>8}  perplexity {perplexity:.4f}  forward passes {seconds:.1f} s")
    print(
        f"perplexity increase {comparison.perplexity_increase:+.2%}, "
        f"top-1 agreement {comparison.agreement:.2%}"
    )
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
