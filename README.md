# Kiyal

Kiyal (Kyrgyz *кыял*, "dream") is an LLM inference engine and OpenAI-compatible server, built
from scratch in PyTorch.

## Status

Early development. Implemented so far:

- automatic device and dtype selection (CUDA, MPS or CPU)
- the Llama model: RMSNorm, rotary position embeddings, grouped-query attention and a SwiGLU MLP
- a strict safetensors loader for Hugging Face checkpoints, from a local directory or the Hub
- sampling with temperature, top-k, top-p, min-p, repetition, presence and frequency penalties,
  and per-request seeds, with settings batched per row
- streaming detokenization that never prints half a character, and stop strings matched across
  token boundaries
- a KV cache: the keys and values of every layer are kept in one preallocated tensor, so the
  prompt runs through the model once and each later step runs on a single new token
- `kiyal generate`, which streams text decoded with the KV cache. The loop that recomputes the
  whole sequence at every step is kept as the baseline for tests and benchmarks.

On SmolLM2-135M-Instruct in fp32 on CPU, the model's logits match `transformers` within 1e-4, and
greedy generation for a chat prompt gives the same tokens as `transformers`' `generate()`. Decoding
with the KV cache gives the same tokens as decoding without it. Batching and the server are not
implemented yet.

## Setup

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 if it is missing.

```sh
git clone https://github.com/doublepunch31/kiyal.git
cd kiyal
uv sync --extra cpu
uv run kiyal --help
```

The `cpu` extra installs PyTorch's CPU-only build on Linux and the standard build on macOS, which
runs on Apple Silicon GPUs through MPS. An install option for NVIDIA GPUs will be added with the
first GPU benchmarks.

## Usage

```sh
uv run kiyal generate --model HuggingFaceTB/SmolLM2-135M-Instruct "What is the capital of France?"
```

The prompt is wrapped in the model's chat template and the reply is printed as it is generated.
The model is downloaded into the Hugging Face cache on first use (about 270 MB).

```sh
# Continue plain text instead, greedily, up to the first full stop.
uv run kiyal generate --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --raw --temperature 0 --stop "." "The capital of France is"
```

`uv run kiyal generate --help` lists the sampling options.

## Benchmarks

One request with a 128-token prompt, decoded greedily with and without the KV cache.
SmolLM2-135M-Instruct in fp32 on a laptop CPU (Apple M4), median of 3 runs. These are not GPU
numbers.

![Decoding speed and total time against output length](benchmarks/results/latency-cpu.png)

| output tokens | KV cache tokens/s | naive tokens/s | KV cache total s | naive total s | speedup |
|---|---|---|---|---|---|
| 32 | 81.8 | 15.2 | 0.39 | 2.10 | 5.4 |
| 64 | 82.5 | 13.8 | 0.78 | 4.63 | 6.0 |
| 128 | 87.9 | 12.1 | 1.46 | 10.60 | 7.3 |
| 256 | 89.6 | 10.7 | 2.86 | 24.01 | 8.4 |
| 512 | 89.1 | 7.0 | 5.75 | 73.57 | 12.8 |
| 1024 | 85.3 | 3.9 | 12.01 | 260.45 | 21.7 |
| 2048 | 83.2 | 2.1 | 24.62 | 966.92 | 39.3 |

With the cache, total time doubles when the output doubles. Without it, every step runs the model
on the whole sequence, so total time follows the sum of the sequence lengths: doubling the output
multiplies it by 2.2 for short outputs, where the 128-token prompt is most of the sequence, and by
3.7 from 1024 to 2048 tokens, close to the factor of 4 of quadratic growth.

The result files and the full table are in [benchmarks/results/](benchmarks/results/).

## Development

```sh
uv run ruff format && uv run ruff check   # format and lint
uv run pyright                            # type-check
uv run pytest -m "not slow"               # fast tests
uv run pytest                             # all tests, including those that download weights
```

The slow tests compare against `transformers` on SmolLM2-135M-Instruct and download its weights
(about 270 MB) into the Hugging Face cache on first run. The fast tests use a tiny random model and
download nothing.

Every optimization is tested for equivalence with the simpler path it replaces. The KV cache is
the first step: at every decoding step the cached logits must match the uncached ones within a
tolerance, and greedy generation must give the same tokens. Steps where the two best tokens are
closer than twice the tolerance are near ties that rounding can flip, so the token comparison
stops at the first one.

```sh
# Single-request decoding speed at several output lengths, saved as JSON in benchmarks/raw/.
# The naive engine takes over an hour on a laptop CPU at the default lengths.
uv run python -m benchmarks.latency --model HuggingFaceTB/SmolLM2-135M-Instruct --engine cached
uv run python -m benchmarks.latency --model HuggingFaceTB/SmolLM2-135M-Instruct --engine naive

# Chart (PNG) and table (markdown) from one result file per engine.
uv run python -m benchmarks.plot latency benchmarks/results/latency-*-cpu.json \
    --output benchmarks/results/latency-cpu
```

## License

MIT. See [LICENSE](LICENSE).
