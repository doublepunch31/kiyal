SmolLM2-135M-Instruct, cpu float32, Apple M4, 128-token prompt, greedy, median of 3 runs

| output tokens | cached tokens/s | cached total s | cached x previous | naive tokens/s | naive total s | naive x previous | speedup |
|---|---|---|---|---|---|---|---|
| 32 | 81.8 | 0.39 | - | 15.2 | 2.10 | - | 5.4 |
| 64 | 82.5 | 0.78 | 1.98 | 13.8 | 4.63 | 2.20 | 6.0 |
| 128 | 87.9 | 1.46 | 1.88 | 12.1 | 10.60 | 2.29 | 7.3 |
| 256 | 89.6 | 2.86 | 1.96 | 10.7 | 24.01 | 2.26 | 8.4 |
| 512 | 89.1 | 5.75 | 2.01 | 7.0 | 73.57 | 3.06 | 12.8 |
| 1024 | 85.3 | 12.01 | 2.09 | 3.9 | 260.45 | 3.54 | 21.7 |
| 2048 | 83.2 | 24.62 | 2.05 | 2.1 | 966.92 | 3.71 | 39.3 |
