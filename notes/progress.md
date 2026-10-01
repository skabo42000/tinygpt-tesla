# Progress

Where we are in the course, and the real numbers from our own runs.

**Hardware:** Intel i5-1235U (10 cores), 7.7 GB RAM, no NVIDIA GPU, so we train on the CPU.

**Dataset:** *The Inventions, Researches and Writings of Nikola Tesla* (Project Gutenberg #39272), cleaned into `data/input.txt` (976,605 characters). Recreate it with `uv run python -m tinygpt.download_tesla`. We switched from Tiny Shakespeare on 2026-09-30, so Karpathy's Shakespeare numbers are only rough guides for us.

| # | Lesson | Status | Our numbers |
|---|---|---|---|
| 0 | Setup | done | PyTorch 2.14.0 (CPU); 1000² matmul 6.4 ms, 2000² 54 ms (8.5×); 85 unique chars |
| 1 | Tensors & shapes | done | wrap-up: (4,8,32) @ (32,85) -> (4,8,85) = one score per vocab char |
| 2 | How a model learns | done | found w=2.000, b=1.000 from random start in 100 steps; lr 0.01 too slow, 0.1 good, 1.1 blew up (b = -4218) |
| 3 | Scores → probabilities → loss | done | know-nothing loss = ln(85) = 4.4427 (our step-0 target) |
| 4 | Dataset + character tokenizer | done | V=85, 976,605 tokens; lossless round trip; frequency-only loss 3.05 |
| 5 | Batches | done | train 878,944 / val 97,661 chars; batch (4, 8) = 32 examples |
| 6 | Bigram model | | |
| 7 | The averaging trick | | |
| 8 | One self-attention head | | |
| 9 | Multi-head attention + feed-forward | | |
| 10 | Full GPT block | | |
| 11 | Clean package + tests | | |
| 12 | Training run | | |
| 13 | Experiments | | |
| 14 | BPE tokenizer from scratch | | |
| 15 | Swap BPE into the GPT | | |
| 16 | Modern upgrades (optional) | | |
| 17 | Capstone (optional) | | |
