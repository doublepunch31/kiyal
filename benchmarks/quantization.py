"""Weight memory and decoding speed of a model in its dtype and with int8 weights.

Each version is loaded in turn and decodes batches of equal-length prompts with static
batching, as in the throughput benchmark, so the two differ only in how the linear layers
store and use their weights.

Run from the repository root:

    uv run python -m benchmarks.quantization --model HuggingFaceTB/SmolLM2-135M-Instruct
"""

import argparse
import gc
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import torch
from torch import nn

from benchmarks.common import RAW_DIR, capture_environment, save_result
from benchmarks.throughput import Settings, measure
from kiyal.device import resolve_device, resolve_dtype
from kiyal.engine.generator import TextGenerator
from kiyal.model.llama import LlamaModel
from kiyal.model.loader import Quantization, load_model, resolve_model_dir
from kiyal.tokenizer import Tokenizer

DEFAULT_BATCH_SIZES = (1, 4, 16, 64)


def weight_bytes(model: nn.Module) -> int:
    """Returns the bytes of a model's weights: its parameters and persistent buffers.

    A tensor shared by two modules, such as a tied embedding and LM head, counts once.
    Non-persistent buffers, such as the rotary tables, are derived from the config and are
    not counted.
    """
    sizes = {tensor.data_ptr(): tensor.nbytes for tensor in model.state_dict().values()}
    return sum(sizes.values())


def allocated_bytes(device: torch.device) -> int | None:
    """Returns the memory held by live tensors on a GPU, or None on the CPU."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    if device.type == "mps":
        return torch.mps.current_allocated_memory()
    return None


def free_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def measure_weights(
    load: Callable[..., LlamaModel],
    quantization: Quantization | None,
    tokenizer: Tokenizer,
    settings: Settings,
    device: torch.device,
) -> dict[str, object]:
    """Loads one version of the model with `load(quantization=...)`, measures it and frees it."""
    free_memory(device)
    before = allocated_bytes(device)
    model = load(quantization=quantization)
    after = allocated_bytes(device)
    entry: dict[str, object] = {
        "weights": quantization or str(model.embed_tokens.weight.dtype).removeprefix("torch."),
        "weight_bytes": weight_bytes(model),
        # Everything the model holds on the device, including the rotary tables.
        "allocated_bytes": None if before is None or after is None else after - before,
        "results": measure(TextGenerator(model, tokenizer), tokenizer, settings, device),
    }
    del model
    free_memory(device)
    return entry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measures weight memory and decoding speed with and without int8 weights."
    )
    parser.add_argument("--model", required=True, help="checkpoint directory or Hub repo id")
    parser.add_argument("--revision", help="branch, tag or commit of a Hub repo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", help="dtype of the unquantized model")
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
    settings = Settings(
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        batch_sizes=tuple(args.batch_sizes),
        repeats=args.repeats,
        warmup=args.warmup,
    )
    load = partial(load_model, model_dir, device=device, dtype=dtype)
    versions = [
        measure_weights(load, quantization, tokenizer, settings, device)
        for quantization in (None, "int8")
    ]
    result: dict[str, object] = {
        "benchmark": "quantization",
        "engine": "static",
        "model": args.model,
        "revision": args.revision,
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "environment": capture_environment(device, dtype),
        "results": versions,
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = save_result(
        result, args.output or RAW_DIR / f"quantization-{device.type}-{timestamp}.json"
    )

    for version in versions:
        num_bytes, entries = version["weight_bytes"], version["results"]
        assert isinstance(num_bytes, int)
        assert isinstance(entries, list)
        print(f"{version['weights']}: weights {num_bytes / 1e6:.1f} MB")
        print(f"{'batch size':>10}  {'tokens/s':>8}  {'TTFT ms':>8}  {'step ms':>8}")
        for entry in entries:
            summary = entry["median"]
            print(
                f"{entry['batch_size']:>10}  {summary['output_tokens_per_s']:>8.1f}  "
                f"{summary['ttft_s'] * 1e3:>8.1f}  {summary['step_s'] * 1e3:>8.1f}"
            )
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
