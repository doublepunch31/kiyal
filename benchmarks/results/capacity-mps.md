SmolLM2-135M-Instruct, mps float16, Apple M4, 64 requests at once, prompts 32-256 and outputs 16-256 tokens, greedy, median of 3 runs

| KV tokens | KV MB | contiguous requests | paged most requests | contiguous mean requests | paged mean requests | paged preemptions | contiguous tokens/s | paged tokens/s | throughput x | contiguous p99 E2E s | paged p99 E2E s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 45 | 4 | 13 | 4.0 | 7.8 | 55 | 249.9 | 400.1 | 1.60 | 39.37 | 24.35 |
| 4096 | 90 | 8 | 26 | 7.7 | 15.3 | 40 | 433.6 | 507.0 | 1.17 | 22.63 | 19.35 |
| 8192 | 180 | 16 | 54 | 14.9 | 25.5 | 32 | 579.2 | 664.7 | 1.15 | 16.98 | 14.79 |
