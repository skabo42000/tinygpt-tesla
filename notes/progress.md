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
| 6 | Bigram model | done | start 4.99 -> val 2.41 in 5000 steps (~20 s); counting bigram 2.406; 7,225 params |
| 7 | The averaging trick | done | loop = matmul = masked softmax (allclose True) |
| 8 | One self-attention head | done | val 2.31 (bigram 2.41), 9,621 params, ~45 s; start loss 4.48; head looks 71% at itself, 25% at previous char |
| 9 | Multi-head attention + feed-forward | done | val 1.90 (one head 2.31), 17,973 params, ~70 s; heads: self / 1-back / self / spread-out |
| 10 | Full GPT block | done | 214,357 params (4 layers, 4 heads, C=64, T=64); val 1.51 after 2,500 steps (~6.5 min); a 5,000-step test reached 1.40 (~20 min); train/val gap ~0.11 |
| 11 | Clean package + tests | done | src/tinygpt/{tokenizer,data,model}.py; 8/8 tests pass; params 214,357 by hand = PyTorch (feed-forward 62%); broken mask leaks 0.034 vs 0.0 |
| 12 | Training run | done | 830,037 params (4 layers, C=128, T=128), 3,000 steps in 37 min; val 1.274 (train 1.122, gap 0.15); still improving at the end |
| 13 | Experiments | done | 800 steps, C=64: baseline 1.83; no positions +0.11; no residual +1.23 (3.06); no scaling -0.03 (no effect at hs=16); no mask 0.09 but garbage text; 20K chars: no dropout val 2.16->2.52 (overfit), dropout 0.3 val 2.09 |
| 14 | BPE tokenizer from scratch | done | 512 tokens learned in 11 s; book 976,605 chars -> 440,673 tokens (2.22 chars/token); first merge ' t'; saved checkpoints/tesla_bpe_512.json; 13/13 tests |
| 15 | Swap BPE into the GPT | done | same settings, 43 min, 939,776 params: BPE 1.69 bits/char vs character 1.84 (on the same val text); 95% vs 96% real words; writes 2.4x faster; BPE train/val gap larger (~31 passes over the data vs ~14) |
| 16 | Modern upgrades (optional) | | |
| 17 | Capstone (optional) | | |
