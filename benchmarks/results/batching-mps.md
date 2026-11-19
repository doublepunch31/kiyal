SmolLM2-135M-Instruct, mps float16, Apple M4, 64 requests at once, prompts 32-256 and outputs 16-256 tokens, greedy, median of 3 runs

| batch size | static tokens/s | continuous tokens/s | static p50 E2E s | continuous p50 E2E s | static p99 E2E s | continuous p99 E2E s | static p50 TTFT s | continuous p50 TTFT s | throughput x | static decode use |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 254.0 | 371.3 | 20.51 | 15.50 | 38.77 | 26.44 | 17.05 | 11.53 | 1.46 | 67% |
| 16 | 362.4 | 530.5 | 13.69 | 11.57 | 26.77 | 18.53 | 10.74 | 6.43 | 1.46 | 62% |
| 32 | 516.0 | 709.2 | 10.94 | 8.99 | 19.02 | 13.82 | 6.67 | 1.58 | 1.37 | 61% |
