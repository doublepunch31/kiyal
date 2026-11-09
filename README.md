# Kiyal

Kiyal (Kyrgyz *кыял*, "dream") is an LLM inference engine and OpenAI-compatible server, built
from scratch in PyTorch.

## Status

Early development. Implemented so far:

- automatic device and dtype selection (CUDA, MPS or CPU)
- the Llama model: RMSNorm, rotary position embeddings, grouped-query attention and a SwiGLU MLP
- a strict safetensors loader for Hugging Face checkpoints, from a local directory or the Hub

On SmolLM2-135M-Instruct in fp32 on CPU, the model's logits match `transformers` within 1e-4 and
greedy decoding picks the same token at each of 64 steps. Text generation, the KV cache, batching
and the server are not implemented yet.

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

## License

MIT. See [LICENSE](LICENSE).
