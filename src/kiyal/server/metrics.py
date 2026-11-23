from collections import abc
from typing import Final

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from prometheus_client.metrics_core import CounterMetricFamily, GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

from kiyal.engine.async_engine import EngineStats

# Upper bounds of the histogram buckets, in seconds: about 2.5x apart, from what a step on a
# GPU takes up to what a long queue can add.
TTFT_BUCKETS: Final = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
                       10.0, 25.0, 50.0, 100.0)  # fmt: skip
ITL_BUCKETS: Final = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
E2E_BUCKETS: Final = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0,
                      250.0, 500.0)  # fmt: skip


class Metrics:
    """The Prometheus metrics of one server.

    Latencies and token counts are recorded by the server for each request, on the event
    loop's thread, as the outputs reach it, so they include the time spent waiting for a
    place in the batch. The engine's state is read from its latest stats whenever the
    metrics are collected.

    The metrics live in a registry of their own rather than prometheus_client's global
    one, so that several servers can exist in one process, as in tests.
    """

    def __init__(self, stats: abc.Callable[[], EngineStats]) -> None:
        """Creates the metrics.

        Args:
            stats: Returns the engine's latest stats.
        """
        self.registry = CollectorRegistry()
        self.time_to_first_token = Histogram(
            "kiyal_time_to_first_token_seconds",
            "Time from the arrival of a request to its first output token.",
            buckets=TTFT_BUCKETS,
            registry=self.registry,
        )
        self.inter_token_latency = Histogram(
            "kiyal_inter_token_latency_seconds",
            "Time between two consecutive output tokens of a request.",
            buckets=ITL_BUCKETS,
            registry=self.registry,
        )
        self.e2e_request_latency = Histogram(
            "kiyal_e2e_request_latency_seconds",
            "Time from the arrival of a request to its last output token.",
            buckets=E2E_BUCKETS,
            registry=self.registry,
        )
        self.prompt_tokens = Counter(
            "kiyal_prompt_tokens",
            "Prompt tokens of the requests that started.",
            registry=self.registry,
        )
        self.generation_tokens = Counter(
            "kiyal_generation_tokens",
            "Output tokens generated.",
            registry=self.registry,
        )
        self.requests_finished = Counter(
            "kiyal_requests_finished",
            "Requests that ended, by finish reason: stop, length, or abort when the request "
            "ended before its last token because the client left or the engine stopped.",
            ["finish_reason"],
            registry=self.registry,
        )
        self.registry.register(_EngineCollector(stats))

    def render(self) -> bytes:
        """The metrics in Prometheus' text format."""
        return generate_latest(self.registry)


class _EngineCollector(Collector):
    """Reports the engine's state as it is when the metrics are collected.

    The engine owns these numbers, so they are read rather than mirrored into
    prometheus_client's own counters, which only go up by increments.
    """

    def __init__(self, stats: abc.Callable[[], EngineStats]) -> None:
        self._stats = stats

    def collect(self) -> abc.Iterable[Metric]:
        stats = self._stats()
        yield GaugeMetricFamily(
            "kiyal_num_requests_running",
            "Requests in the batch.",
            value=stats.num_running,
        )
        yield GaugeMetricFamily(
            "kiyal_num_requests_waiting",
            "Requests waiting to start, including preempted ones.",
            value=stats.num_waiting,
        )
        yield GaugeMetricFamily(
            "kiyal_kv_cache_usage_ratio",
            "Share of the KV cache blocks that requests hold. Cached blocks that no request "
            "holds count as free.",
            value=1 - stats.num_free_kv_blocks / stats.num_kv_blocks,
        )
        yield CounterMetricFamily(
            "kiyal_num_preemptions",
            "Times a request gave up its KV blocks to make room, to be computed again later.",
            value=stats.num_preemptions,
        )
        yield CounterMetricFamily(
            "kiyal_prefix_cache_queries",
            "Prompt tokens of the requests that started, looked up in the prefix cache.",
            value=stats.num_prefix_query_tokens,
        )
        yield CounterMetricFamily(
            "kiyal_prefix_cache_hits",
            "Prompt tokens found in the prefix cache.",
            value=stats.num_prefix_hit_tokens,
        )
