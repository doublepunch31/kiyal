import time

import httpx
from openai import OpenAI
from prometheus_client.parser import text_string_to_metric_families

from tests.server.conftest import MODEL_NAME, Server

type Samples = dict[tuple[str, tuple[tuple[str, str], ...]], float]


def scrape(server: Server) -> Samples:
    response = httpx.get(f"{server.base_url}/metrics")
    assert response.status_code == 200
    samples: Samples = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            samples[sample.name, tuple(sorted(sample.labels.items()))] = sample.value
    return samples


def value(samples: Samples, name: str, **labels: str) -> float:
    return samples[name, tuple(sorted(labels.items()))]


def growth(before: Samples, after: Samples, name: str, **labels: str) -> float:
    return value(after, name, **labels) - before.get((name, tuple(sorted(labels.items()))), 0.0)


def test_requests_are_counted_with_their_tokens_and_latencies(
    server: Server, client: OpenAI
) -> None:
    before = scrape(server)
    usages = [
        client.completions.create(
            model=MODEL_NAME, prompt="the cat", max_tokens=n, extra_body={"ignore_eos": True}
        ).usage
        for n in (3, 8)
    ]
    usages += [
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "the cat"}],
            max_tokens=5,
            extra_body={"ignore_eos": True},
        ).usage
    ]
    assert all(usage is not None for usage in usages)
    prompt_tokens = sum(usage.prompt_tokens for usage in usages if usage)
    output_tokens = sum(usage.completion_tokens for usage in usages if usage)
    after = scrape(server)
    assert growth(before, after, "kiyal_prompt_tokens_total") == prompt_tokens
    assert growth(before, after, "kiyal_generation_tokens_total") == output_tokens == 16
    assert growth(before, after, "kiyal_requests_finished_total", finish_reason="length") == 3
    assert growth(before, after, "kiyal_time_to_first_token_seconds_count") == 3
    assert growth(before, after, "kiyal_e2e_request_latency_seconds_count") == 3
    # One gap between each two consecutive tokens of a request.
    assert growth(before, after, "kiyal_inter_token_latency_seconds_count") == 16 - 3
    assert growth(before, after, "kiyal_e2e_request_latency_seconds_sum") > 0


def test_engine_state_is_reported(server: Server) -> None:
    body = {
        "model": MODEL_NAME,
        "prompt": "the cat",
        "max_tokens": 2000,
        "ignore_eos": True,
        "stream": True,
    }
    before = scrape(server)
    with httpx.stream("POST", f"{server.base_url}/v1/completions", json=body) as response:
        lines = response.iter_lines()
        next(lines)
        running = scrape(server)
    assert value(running, "kiyal_num_requests_running") == 1
    assert value(running, "kiyal_num_requests_waiting") == 0
    assert 0 < value(running, "kiyal_kv_cache_usage_ratio") < 1
    assert value(running, "kiyal_num_preemptions_total") == 0
    # The client left before the request ended.
    for _ in range(500):
        after = scrape(server)
        if growth(before, after, "kiyal_requests_finished_total", finish_reason="abort") == 1:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("the abort was not counted")
    assert value(after, "kiyal_num_requests_running") == 0


def test_prefix_cache_hits_are_counted(server: Server, client: OpenAI) -> None:
    # 40 tokens: the second request finds the first's two full blocks of 16 in the cache.
    prompt = [5 + i for i in range(40)]
    before = scrape(server)
    for _ in range(2):
        client.completions.create(model=MODEL_NAME, prompt=prompt, max_tokens=1)
    after = scrape(server)
    assert growth(before, after, "kiyal_prefix_cache_queries_total") == 80
    assert growth(before, after, "kiyal_prefix_cache_hits_total") == 32


def test_metrics_use_the_prometheus_text_format(server: Server) -> None:
    response = httpx.get(f"{server.base_url}/metrics")
    assert response.headers["content-type"].startswith("text/plain; version=")
    assert "# TYPE kiyal_time_to_first_token_seconds histogram" in response.text
