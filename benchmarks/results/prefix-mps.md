SmolLM2-135M-Instruct, mps float16, Apple M4, 16 requests of one system prompt and 16-64 other tokens, 16 output tokens, greedy, median of 3 runs

| system tokens | one at a time, off p50 ms | one at a time, on p50 ms | x | at once, off p50 ms | at once, on p50 ms | x | at once, off p99 ms | at once, on p99 ms | at once, off tokens/s | at once, on tokens/s | hit rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 | 54.1 | 23.2 | 2.3 | 528 | 208 | 2.5 | 659 | 208 | 253.4 | 461.6 | 77% |
| 500 | 91.5 | 24.4 | 3.7 | 1120 | 342 | 3.3 | 1288 | 342 | 151.8 | 343.3 | 86% |
| 1000 | 177.2 | 26.3 | 6.7 | 1527 | 835 | 1.8 | 2687 | 835 | 78.5 | 191.6 | 89% |
| 2000 | 403.7 | 30.8 | 13.1 | 3575 | 2776 | 1.3 | 6341 | 2776 | 36.2 | 74.6 | 92% |
