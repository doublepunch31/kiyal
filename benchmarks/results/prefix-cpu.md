SmolLM2-135M-Instruct, cpu float32, Apple M4, 16 requests of one system prompt and 16-64 other tokens, 16 output tokens, greedy, median of 3 runs

| system tokens | one at a time, off p50 ms | one at a time, on p50 ms | x | at once, off p50 ms | at once, on p50 ms | x | at once, off p99 ms | at once, on p99 ms | at once, off tokens/s | at once, on tokens/s | hit rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 | 103.1 | 39.0 | 2.6 | 1093 | 546 | 2.0 | 1360 | 546 | 123.7 | 209.7 | 77% |
| 500 | 191.6 | 43.9 | 4.4 | 2521 | 1030 | 2.4 | 2873 | 1030 | 68.8 | 132.8 | 86% |
| 1000 | 484.1 | 63.1 | 7.7 | 3754 | 2768 | 1.4 | 6625 | 2768 | 31.3 | 61.4 | 89% |
| 2000 | 1224.4 | 96.9 | 12.6 | 15117 | 16318 | 0.9 | 27407 | 16318 | 7.8 | 12.8 | 92% |
