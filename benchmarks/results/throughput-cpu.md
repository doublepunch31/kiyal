SmolLM2-135M-Instruct, cpu float32, Apple M4, 128-token prompts, 128 output tokens, greedy, median of 3 runs

| batch size | output tokens/s | x batch 1 | step ms | x batch 1 | TTFT ms |
|---|---|---|---|---|---|
| 1 | 92.1 | 1.00 | 10.5 | 1.00 | 53 |
| 2 | 62.7 | 0.68 | 31.4 | 2.99 | 83 |
| 4 | 127.5 | 1.38 | 30.5 | 2.90 | 138 |
| 8 | 186.6 | 2.03 | 41.1 | 3.90 | 271 |
| 16 | 369.1 | 4.01 | 39.5 | 3.75 | 531 |
| 32 | 604.0 | 6.56 | 44.9 | 4.27 | 1077 |
| 64 | 816.3 | 8.86 | 62.3 | 5.92 | 2127 |
