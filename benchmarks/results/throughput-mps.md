SmolLM2-135M-Instruct, mps float16, Apple M4, 128-token prompts, 128 output tokens, greedy, median of 3 runs

| batch size | output tokens/s | x batch 1 | step ms | x batch 1 | TTFT ms |
|---|---|---|---|---|---|
| 1 | 108.4 | 1.00 | 9.0 | 1.00 | 31 |
| 2 | 173.6 | 1.60 | 11.3 | 1.25 | 43 |
| 4 | 331.6 | 3.06 | 11.6 | 1.28 | 73 |
| 8 | 583.7 | 5.38 | 12.7 | 1.41 | 134 |
| 16 | 812.8 | 7.50 | 17.8 | 1.96 | 264 |
| 32 | 1295.9 | 11.95 | 20.9 | 2.31 | 511 |
| 64 | 1829.1 | 16.87 | 27.3 | 3.02 | 1012 |
