SmolLM2-135M-Instruct, cpu float32, Apple M4, 64 requests at once, prompts 32-256 and outputs 16-256 tokens, greedy, median of 3 runs

| batch size | static tokens/s | continuous tokens/s | static p50 E2E s | continuous p50 E2E s | static p99 E2E s | continuous p99 E2E s | static p50 TTFT s | continuous p50 TTFT s | throughput x | static decode use |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 92.5 | 117.3 | 54.18 | 47.75 | 106.40 | 83.87 | 44.54 | 31.84 | 1.27 | 67% |
| 16 | 175.0 | 210.5 | 29.63 | 30.17 | 55.11 | 46.75 | 22.06 | 16.27 | 1.20 | 62% |
| 32 | 269.4 | 308.3 | 20.53 | 18.78 | 36.39 | 31.86 | 12.04 | 2.83 | 1.14 | 61% |
