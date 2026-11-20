SmolLM2-135M-Instruct, cpu float32, Apple M4, 64 requests at once, prompts 32-256 and outputs 16-256 tokens, greedy, median of 3 runs

| KV tokens | KV MB | contiguous requests | paged most requests | contiguous mean requests | paged mean requests | paged preemptions | contiguous tokens/s | paged tokens/s | throughput x | contiguous p99 E2E s | paged p99 E2E s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 90 | 4 | 13 | 4.0 | 7.8 | 55 | 66.7 | 92.0 | 1.38 | 147.72 | 106.66 |
| 4096 | 180 | 8 | 26 | 7.7 | 15.3 | 40 | 99.0 | 114.8 | 1.16 | 99.35 | 85.73 |
| 8192 | 360 | 16 | 54 | 14.9 | 25.5 | 32 | 153.8 | 179.4 | 1.17 | 63.99 | 54.86 |
