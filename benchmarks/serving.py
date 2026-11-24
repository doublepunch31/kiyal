"""Online serving: requests arrive at random times, at increasing request rates.

At each rate, the requests of a seeded workload (prompt and output lengths as in the
batching benchmark) arrive as a Poisson process: the gaps between arrivals are exponential
with a mean of 1 / rate. The same seed draws the same gaps at every rate, scaled by 1 /
rate, so the patterns differ only in speed. Each request is a streamed greedy completion
of token ids that ignores EOS, so it generates exactly its output length, and the client
records when each chunk of its response arrives.

Each server runs as its own process, started, warmed up, measured at every rate and
stopped in turn:
- hf: transformers' generate(), one request at a time (benchmarks/hf_baseline.py)
- static: Kiyal's static batching, `--batch-size` requests per batch
- continuous: Kiyal's engine with one KV row of the longest request per sequence, the M5
  layout (block size = max_model_len)
- paged: Kiyal's engine with blocks of 16 tokens
The continuous and paged servers get the same KV memory, `--kv-tokens` tokens. Prefix
caching is off for both, so that they differ in the KV layout alone.

Run from the repository root:

    uv run python -m benchmarks.serving --model HuggingFaceTB/SmolLM2-135M-Instruct
"""

import argparse
import asyncio
import contextlib
import itertools
import json
import random
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from benchmarks.batching import Workload, build_workload
from benchmarks.common import (
    RAW_DIR,
    REPO_ROOT,
    capture_environment,
    median,
    percentile,
    save_result,
)
from kiyal.device import DTYPES, resolve_device, resolve_dtype
from kiyal.model.loader import resolve_model_dir
from kiyal.tokenizer import Tokenizer

SERVERS = ("hf", "static", "continuous", "paged")
DEFAULT_RATES = (1.0, 2.0, 4.0, 8.0, 16.0)
# The name every server is started with, and that every request gives.
SERVED_MODEL_NAME = "model"
PAGED_BLOCK_SIZE = 16


@dataclass(frozen=True, slots=True)
class Settings:
    """What a serving run measures, and how often.

    Attributes:
        model: Checkpoint directory or Hub repo id.
        revision: Branch, tag or commit of a Hub repo.
        device: Device the servers run on, as `kiyal serve --device` takes it.
        dtype: Data type of the servers, as `kiyal serve --dtype` takes it.
        servers: Servers to measure, in order.
        rates: Request rates in requests per second, in order.
        repeats: Timed runs per server and rate.
        warmup_requests: Requests sent all at once to each server before it is measured.
        kv_tokens: KV memory of the continuous and paged servers, in tokens.
        batch_size: Requests per batch of the static server.
        max_num_seqs: Most requests in the paged server's batch. Its memory limits it
            sooner when this is large.
    """

    model: str
    revision: str | None
    device: str
    dtype: str
    servers: tuple[str, ...]
    rates: tuple[float, ...]
    repeats: int
    warmup_requests: int
    kv_tokens: int
    batch_size: int
    max_num_seqs: int


@dataclass(frozen=True, slots=True)
class RequestRecord:
    """What the client saw of one request. Times are in seconds from the start of the run.

    Attributes:
        sent: When the request was sent.
        chunk_times: When each chunk with output text, or with the finish reason, arrived.
            A chunk holds one token, or several when the detokenizer held text back.
        prompt_tokens: Prompt tokens, as the server counted them.
        output_tokens: Generated tokens, as the server counted them.
    """

    sent: float
    chunk_times: list[float]
    prompt_tokens: int
    output_tokens: int


def poisson_arrivals(num_requests: int, rate: float, seed: int) -> list[float]:
    """Returns arrival times in seconds of a Poisson process with `rate` arrivals per
    second. The first request arrives at 0.

    Raises:
        ValueError: If the rate is not above 0.
    """
    if not rate > 0:
        raise ValueError(f"rate must be above 0, got {rate}")
    generator = random.Random(seed)
    times = [0.0]
    for _ in range(num_requests - 1):
        times.append(times[-1] + generator.expovariate(rate))
    return times


def server_command(server: str, settings: Settings, max_model_len: int, port: int) -> list[str]:
    """Returns the command that starts a server on a local port.

    Raises:
        ValueError: If the server name is unknown.
    """
    common = [
        *["--model", settings.model, "--device", settings.device, "--dtype", settings.dtype],
        *["--max-model-len", str(max_model_len), "--port", str(port)],
        *["--served-model-name", SERVED_MODEL_NAME, "--log-level", "warning"],
    ]
    if settings.revision is not None:
        common += ["--revision", settings.revision]
    if server in ("hf", "static"):
        batch_size = 1 if server == "hf" else settings.batch_size
        return [
            *[sys.executable, "-m", "benchmarks.hf_baseline", "--engine", server],
            *["--batch-size", str(batch_size), *common],
        ]
    if server == "continuous":
        # One block of the longest request per sequence: a KV row each, as in M5.
        rows = settings.kv_tokens // max_model_len
        engine = [
            *["--block-size", str(max_model_len), "--num-kv-blocks", str(rows)],
            *["--max-num-seqs", str(rows)],
        ]
    elif server == "paged":
        engine = [
            *["--block-size", str(PAGED_BLOCK_SIZE)],
            *["--num-kv-blocks", str(settings.kv_tokens // PAGED_BLOCK_SIZE)],
            *["--max-num-seqs", str(settings.max_num_seqs)],
        ]
    else:
        raise ValueError(f"unknown server {server!r}; expected one of {', '.join(SERVERS)}")
    # The console script next to the interpreter, so that it runs in the same environment.
    kiyal = str(Path(sys.executable).with_name("kiyal"))
    return [kiyal, "serve", *common, *engine, "--no-prefix-caching"]


@contextlib.contextmanager
def running_server(
    command: list[str], base_url: str, log_path: Path, timeout_s: float = 900
) -> Iterator[None]:
    """Starts a server process, waits until it is healthy, and stops it at the end.

    The server's output goes to `log_path`. The timeout covers downloading the model.

    Raises:
        RuntimeError: If the server exits before it is healthy.
        TimeoutError: If it is not healthy within the timeout.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            _wait_until_healthy(process, base_url, timeout_s, log_path)
            yield
        finally:
            # Ctrl-C: uvicorn shuts down cleanly, which stops the engine.
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _wait_until_healthy(
    process: subprocess.Popen[bytes], base_url: str, timeout_s: float, log_path: Path
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"the server exited with {process.returncode}; see {log_path}")
        with contextlib.suppress(httpx.TransportError):
            if httpx.get(f"{base_url}/health", timeout=1).status_code == 200:
                return
        time.sleep(0.5)
    raise TimeoutError(f"the server was not healthy after {timeout_s:.0f} s; see {log_path}")


def free_port() -> int:
    """Returns a local port that was free a moment ago."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def send_request(
    client: httpx.AsyncClient, start: float, arrival: float, prompt: list[int], max_tokens: int
) -> RequestRecord:
    """Sends one request at its arrival time and records when its chunks arrive.

    Raises:
        RuntimeError: If the server answers with an error.
    """
    await asyncio.sleep(max(0.0, start + arrival - time.perf_counter()))
    sent = time.perf_counter() - start
    body = {
        "model": SERVED_MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    chunk_times: list[float] = []
    usage: dict[str, int] | None = None
    async with client.stream("POST", "/v1/completions", json=body) as response:
        if response.status_code != 200:
            message = (await response.aread()).decode()
            raise RuntimeError(f"the server answered {response.status_code}: {message}")
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(f"the server failed mid-stream: {chunk['error']['message']}")
            if chunk["choices"]:
                chunk_times.append(time.perf_counter() - start)
            if chunk.get("usage"):
                usage = chunk["usage"]
    if usage is None or not chunk_times:
        raise RuntimeError("the stream ended without its output or its usage")
    return RequestRecord(sent, chunk_times, usage["prompt_tokens"], usage["completion_tokens"])


async def run_workload(
    base_url: str, workload: Workload, arrivals: Sequence[float]
) -> list[RequestRecord]:
    """Sends every request of the workload at its arrival time and waits for all of them."""
    # No limit on connections: a request must never wait for one on the client side.
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(base_url=base_url, timeout=None, limits=limits) as client:
        start = time.perf_counter()
        return await asyncio.gather(
            *(
                send_request(client, start, arrival, prompt, max_tokens)
                for arrival, prompt, max_tokens in zip(
                    arrivals, workload.prompts, workload.output_tokens, strict=True
                )
            )
        )


def summarize(records: Sequence[RequestRecord]) -> dict[str, float]:
    """Computes the serving metrics of one run.

    TTFT runs from sending a request to its first chunk, E2E to its last chunk. TPOT is a
    request's time after its first token divided by its other tokens. ITL is the gap
    between two chunks of a request. The run lasts from the first request sent to the last
    chunk received.
    """
    ttft = [r.chunk_times[0] - r.sent for r in records]
    e2e = [r.chunk_times[-1] - r.sent for r in records]
    tpot = [
        (r.chunk_times[-1] - r.chunk_times[0]) / (r.output_tokens - 1)
        for r in records
        if r.output_tokens > 1
    ]
    itl = [b - a for r in records for a, b in itertools.pairwise(r.chunk_times)]
    duration = max(r.chunk_times[-1] for r in records) - min(r.sent for r in records)
    output_tokens = sum(r.output_tokens for r in records)
    prompt_tokens = sum(r.prompt_tokens for r in records)
    summary = {
        "duration_s": duration,
        "requests_per_s": len(records) / duration,
        "output_tokens_per_s": output_tokens / duration,
        "total_tokens_per_s": (prompt_tokens + output_tokens) / duration,
    }
    for name, values in (("ttft", ttft), ("tpot", tpot), ("itl", itl), ("e2e", e2e)):
        summary[f"{name}_p50_s"] = percentile(values, 50)
        summary[f"{name}_p99_s"] = percentile(values, 99)
    return summary


def measure_server(
    server: str, settings: Settings, workload: Workload, seed: int, log_path: Path
) -> dict[str, object]:
    """Starts a server, warms it up, measures it at every rate, and stops it."""
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    command = server_command(server, settings, workload.max_len, port)
    rates: list[dict[str, object]] = []
    with running_server(command, base_url, log_path):
        count = min(settings.warmup_requests, len(workload.prompts))
        warmup = Workload(workload.prompts[:count], workload.output_tokens[:count])
        asyncio.run(run_workload(base_url, warmup, [0.0] * count))
        for rate in settings.rates:
            arrivals = poisson_arrivals(len(workload.prompts), rate, seed)
            runs = [
                summarize(asyncio.run(run_workload(base_url, workload, arrivals)))
                for _ in range(settings.repeats)
            ]
            middle = {metric: median([run[metric] for run in runs]) for metric in runs[0]}
            rates.append({"rate": rate, "runs": runs, "median": middle})
            print(
                f"{server:>10}  {rate:>6g}/s  {middle['output_tokens_per_s']:>8.1f} tokens/s  "
                f"p99 TTFT {middle['ttft_p99_s']:>7.2f} s  p50 TPOT "
                f"{middle['tpot_p50_s'] * 1000:>6.1f} ms",
                flush=True,
            )
    return {"command": command, "rates": rates}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measures servers under Poisson arrivals.")
    parser.add_argument("--model", required=True, help="checkpoint directory or Hub repo id")
    parser.add_argument("--revision", help="branch, tag or commit of a Hub repo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--servers", nargs="+", choices=SERVERS, default=SERVERS)
    parser.add_argument("--num-requests", type=int, default=64, help="requests per run")
    parser.add_argument(
        "--prompt-tokens", type=int, nargs=2, default=(32, 256), metavar=("MIN", "MAX")
    )
    parser.add_argument(
        "--output-tokens", type=int, nargs=2, default=(16, 256), metavar=("MIN", "MAX")
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of the lengths and arrivals")
    parser.add_argument("--rates", type=float, nargs="+", default=DEFAULT_RATES)
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per server and rate")
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument(
        "--kv-tokens", type=int, default=8192, help="KV memory of the continuous and paged servers"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="requests per batch of the static server"
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=64, help="most requests in the paged batch"
    )
    parser.add_argument("--output", type=Path, help="result file (default: benchmarks/raw/)")
    args = parser.parse_args(argv)

    # Resolved here, so every server runs on the device and in the dtype recorded below.
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    dtype_name = next(name for name, value in DTYPES.items() if value == dtype)
    tokenizer = Tokenizer.from_dir(resolve_model_dir(args.model, revision=args.revision))
    workload = build_workload(
        tokenizer,
        args.num_requests,
        tuple(args.prompt_tokens),
        tuple(args.output_tokens),
        args.seed,
    )
    settings = Settings(
        model=args.model,
        revision=args.revision,
        device=str(device),
        dtype=dtype_name,
        servers=tuple(args.servers),
        rates=tuple(args.rates),
        repeats=args.repeats,
        warmup_requests=args.warmup_requests,
        kv_tokens=args.kv_tokens,
        batch_size=args.batch_size,
        max_num_seqs=args.max_num_seqs,
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    servers: dict[str, object] = {}
    for server in settings.servers:
        log_path = RAW_DIR / f"serving-{server}-{device.type}-{timestamp}.log"
        servers[server] = measure_server(server, settings, workload, args.seed, log_path)
    result: dict[str, object] = {
        "benchmark": "serving",
        "model": args.model,
        "revision": args.revision,
        "num_requests": args.num_requests,
        "prompt_tokens": list(args.prompt_tokens),
        "output_tokens": list(args.output_tokens),
        "seed": args.seed,
        "rates": list(settings.rates),
        "repeats": settings.repeats,
        "warmup_requests": settings.warmup_requests,
        "max_model_len": workload.max_len,
        "kv_tokens": settings.kv_tokens,
        "batch_size": settings.batch_size,
        "max_num_seqs": settings.max_num_seqs,
        "environment": capture_environment(device, dtype),
        "servers": servers,
    }
    path = save_result(result, args.output or RAW_DIR / f"serving-{device.type}-{timestamp}.json")
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
