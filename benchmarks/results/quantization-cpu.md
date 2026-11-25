SmolLM2-135M-Instruct, cpu, Apple M4; static batching, 128-token prompts, 128 greedy output tokens, median of 3 runs; WikiText-2 test, 304,985 tokens in windows of 1024, stride 512

| weights | weight MB | x | allocated MB | WikiText-2 perplexity | x | top-1 agreement |
|---|---|---|---|---|---|---|
| float32 | 538.1 | 1.00 | - | 16.594 | 1.0000 | - |
| int8 | 220.2 | 0.41 | - | 16.622 | 1.0017 | 97.2% |

| batch size | float32 tokens/s | int8 tokens/s | x | float32 step ms | int8 step ms | float32 TTFT ms | int8 TTFT ms |
|---|---|---|---|---|---|---|---|
| 1 | 65.6 | 26.0 | 0.40 | 14.9 | 37.9 | 62 | 116 |
| 4 | 90.0 | 45.2 | 0.50 | 43.7 | 87.1 | 143 | 234 |
| 16 | 339.6 | 172.9 | 0.51 | 42.2 | 87.2 | 644 | 769 |
| 64 | 686.4 | 485.8 | 0.71 | 70.7 | 108.5 | 2946 | 3080 |
