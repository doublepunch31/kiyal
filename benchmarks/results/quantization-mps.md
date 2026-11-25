SmolLM2-135M-Instruct, mps, Apple M4; static batching, 128-token prompts, 128 greedy output tokens, median of 3 runs; WikiText-2 test, 304,985 tokens in windows of 1024, stride 512

| weights | weight MB | x | allocated MB | WikiText-2 perplexity | x | top-1 agreement |
|---|---|---|---|---|---|---|
| float16 | 269.0 | 1.00 | 271.1 | 16.596 | 1.0000 | - |
| int8 | 163.2 | 0.61 | 165.3 | 16.624 | 1.0017 | 97.2% |

| batch size | float16 tokens/s | int8 tokens/s | x | float16 step ms | int8 step ms | float16 TTFT ms | int8 TTFT ms |
|---|---|---|---|---|---|---|---|
| 1 | 105.6 | 29.0 | 0.27 | 9.3 | 34.3 | 31 | 59 |
| 4 | 317.5 | 98.3 | 0.31 | 11.8 | 40.2 | 73 | 111 |
| 16 | 782.8 | 226.0 | 0.29 | 18.5 | 68.0 | 268 | 419 |
| 64 | 1539.9 | 708.3 | 0.46 | 30.7 | 74.0 | 1422 | 2125 |
