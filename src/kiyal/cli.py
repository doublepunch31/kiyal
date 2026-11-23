import argparse
import sys
import time
from collections.abc import Sequence

import uvicorn

from kiyal import __version__
from kiyal.config import EngineConfig
from kiyal.device import DeviceUnavailableError, resolve_device, resolve_dtype
from kiyal.engine.async_engine import AsyncEngine
from kiyal.engine.generator import TextGenerator
from kiyal.engine.llm_engine import LLMEngine
from kiyal.engine.sequence import SamplingParams
from kiyal.model.loader import CheckpointError, load_model, resolve_model_dir
from kiyal.server.app import build_app
from kiyal.tokenizer import Tokenizer, TokenizerError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kiyal",
        description="LLM inference engine and OpenAI-compatible server.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    defaults = SamplingParams()
    generate = commands.add_parser(
        "generate",
        help="generate text for one prompt",
        description="Generates text for one prompt and prints it as it is produced.",
    )
    generate.add_argument("prompt", help="text to continue, or the user message of a chat")
    _add_model_arguments(generate)
    generate.add_argument(
        "--raw",
        action="store_true",
        help="continue the prompt as plain text instead of using the model's chat template",
    )
    generate.add_argument("--system", help="system message placed before the prompt in a chat")
    generate.add_argument(
        "--max-tokens", type=int, default=256, help="most tokens to generate (default: 256)"
    )
    generate.add_argument(
        "--temperature", type=float, default=defaults.temperature, help="0 is greedy"
    )
    generate.add_argument("--top-k", type=int, default=defaults.top_k, help="0 turns it off")
    generate.add_argument("--top-p", type=float, default=defaults.top_p, help="1 turns it off")
    generate.add_argument("--min-p", type=float, default=defaults.min_p, help="0 turns it off")
    generate.add_argument(
        "--repetition-penalty",
        type=float,
        default=defaults.repetition_penalty,
        help="1 turns it off",
    )
    generate.add_argument("--presence-penalty", type=float, default=defaults.presence_penalty)
    generate.add_argument("--frequency-penalty", type=float, default=defaults.frequency_penalty)
    generate.add_argument("--seed", type=int, help="makes sampled output reproducible")
    generate.add_argument(
        "--stop", action="append", default=[], help="string that ends the output; repeatable"
    )
    generate.set_defaults(run=_run_generate)

    engine_defaults = EngineConfig()
    serve = commands.add_parser(
        "serve",
        help="serve a model over an OpenAI-compatible HTTP API",
        description="Serves a model over an OpenAI-compatible HTTP API.",
    )
    _add_model_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1", help="address to listen on")
    serve.add_argument("--port", type=int, default=8000, help="port to listen on")
    serve.add_argument(
        "--served-model-name", help="name clients give in the model field (default: --model)"
    )
    serve.add_argument(
        "--max-num-seqs",
        type=int,
        default=engine_defaults.max_num_seqs,
        help="most requests in the batch at once",
    )
    serve.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=engine_defaults.max_num_batched_tokens,
        help="most prompt tokens that start in one step, and the longest prompt",
    )
    serve.add_argument(
        "--max-model-len",
        type=int,
        default=engine_defaults.max_model_len,
        help="most prompt and output tokens of one request",
    )
    serve.add_argument(
        "--block-size",
        type=int,
        default=engine_defaults.block_size,
        help="tokens per block of the KV cache",
    )
    serve.add_argument(
        "--num-kv-blocks", type=int, help="blocks of the KV cache (default: sized from memory)"
    )
    serve.add_argument(
        "--kv-cache-gb",
        type=float,
        default=engine_defaults.kv_cache_gb,
        help="memory of the KV cache on the CPU and MPS, in GiB",
    )
    serve.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=engine_defaults.gpu_memory_utilization,
        help="share of CUDA memory for the weights, activations and KV cache",
    )
    serve.add_argument(
        "--no-prefix-caching",
        action="store_true",
        help="do not share the KV blocks of common prompt prefixes between requests",
    )
    serve.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="log level of the HTTP server",
    )
    serve.set_defaults(run=_run_serve)
    return parser


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", required=True, help="local checkpoint directory or Hugging Face repo id"
    )
    parser.add_argument("--revision", help="branch, tag or commit of a Hugging Face repo")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:N, mps or cpu")
    parser.add_argument("--dtype", default="auto", help="auto, fp32, fp16 or bf16")


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command-line interface.

    Args:
        argv: Arguments without the program name. Defaults to sys.argv[1:].

    Returns:
        The process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return args.run(args)
    except (CheckpointError, TokenizerError, DeviceUnavailableError, ValueError) as e:
        print(f"kiyal: error: {e}", file=sys.stderr)
        return 1


def _run_generate(args: argparse.Namespace) -> int:
    params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        seed=args.seed,
        stop=tuple(args.stop),
    )
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model_dir = resolve_model_dir(args.model, revision=args.revision)
    tokenizer = Tokenizer.from_dir(model_dir)
    model = load_model(model_dir, device=device, dtype=dtype)
    prompt_token_ids = _encode_prompt(tokenizer, args.prompt, system=args.system, raw=args.raw)
    generator = TextGenerator(model, tokenizer)

    num_tokens = 0
    finish_reason = None
    start = time.perf_counter()
    try:
        for step in generator.generate(prompt_token_ids, params):
            # Flushing after every piece is what makes the output appear token by token.
            print(step.text, end="", flush=True)
            num_tokens += 1
            finish_reason = step.finish_reason
    except KeyboardInterrupt:
        print()
        return 130
    elapsed = time.perf_counter() - start
    # Flushed so that the summary on stderr cannot appear before the end of the text.
    print(flush=True)
    reason = "none" if finish_reason is None else finish_reason.value
    print(
        f"{num_tokens} tokens in {elapsed:.2f} s ({num_tokens / elapsed:.1f} tokens/s), "
        f"finish reason: {reason}",
        file=sys.stderr,
    )
    return 0


def _run_serve(args: argparse.Namespace) -> int:
    config = EngineConfig(
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        num_kv_blocks=args.num_kv_blocks,
        enable_prefix_caching=not args.no_prefix_caching,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_gb=args.kv_cache_gb,
    )
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model_dir = resolve_model_dir(args.model, revision=args.revision)
    model = load_model(model_dir, device=device, dtype=dtype)
    engine = LLMEngine(model, Tokenizer.from_dir(model_dir), config)
    print(
        f"kiyal: {engine.num_kv_blocks} KV blocks of {config.block_size} tokens, "
        f"up to {engine.max_model_len} tokens per request, on {device} in "
        f"{str(dtype).removeprefix('torch.')}",
        file=sys.stderr,
    )
    # A second tokenizer for the server, which encodes prompts on another thread than the
    # one where the engine decodes outputs.
    app = build_app(
        AsyncEngine(engine), Tokenizer.from_dir(model_dir), args.served_model_name or args.model
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def _encode_prompt(
    tokenizer: Tokenizer, prompt: str, *, system: str | None, raw: bool
) -> list[int]:
    # A chat model answers a prompt wrapped in its chat template; given the bare text it
    # would continue the text instead.
    if raw or not tokenizer.has_chat_template:
        if system is not None:
            raise ValueError("--system needs the model's chat template")
        return tokenizer.encode(prompt)
    messages = [{"role": "user", "content": prompt}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})
    return tokenizer.encode_chat(messages)
