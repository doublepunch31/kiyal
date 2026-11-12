import argparse
import sys
import time
from collections.abc import Sequence

from kiyal import __version__
from kiyal.device import DeviceUnavailableError, resolve_device, resolve_dtype
from kiyal.engine.naive import NaiveGenerator
from kiyal.engine.sequence import SamplingParams
from kiyal.model.loader import CheckpointError, load_model, resolve_model_dir
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
    generate.add_argument(
        "--model", required=True, help="local checkpoint directory or Hugging Face repo id"
    )
    generate.add_argument("--revision", help="branch, tag or commit of a Hugging Face repo")
    generate.add_argument("--device", default="auto", help="auto, cuda, cuda:N, mps or cpu")
    generate.add_argument("--dtype", default="auto", help="auto, fp32, fp16 or bf16")
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
    return parser


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
    generator = NaiveGenerator(model, tokenizer)

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
