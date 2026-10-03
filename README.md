# TinyGPT Tesla: a GPT language model built from scratch

A GPT-style language model written from the ground up in **PyTorch** (tokenizer, self-attention, transformer
blocks, training loop and text generation) and trained on a **laptop CPU** on
*The Inventions, Researches and Writings of Nikola Tesla* (1894).

Give it the start of a sentence and it continues in Tesla's voice, one word piece at a time:

> **The alternating current** impulse is equal, and a plate to the insulating brushes was sufficient to
> envelocity, and the speed to work of the condenser, the same with the bulb, a certain great phosphorescence
> when the potential is not sufficient to give at the sound; the loss is illustrated in Fig. 1246. The plates C
> are the condenser should be small, but there is no matter at comparatively points on the other.

It learned how Tesla's writing *sounds* (vocabulary, rhythm, his habit of pointing at figures), not what he
knew: with under a million parameters and one book of training text, it writes plausible-sounding prose, not facts.

**Live demo:** _link added after deployment_

---

## What this project shows

- **Every part of a GPT implemented by hand:** character and byte-pair-encoding (BPE) tokenizers, causal
  self-attention, multi-head attention, feed-forward layers, residual connections, LayerNorm, dropout,
  temperature and top-k sampling.
- **Measured progress:** 17 lesson notebooks build the model one idea at a time, and every step is
  measured on held-out text, from random guessing (6.41 bits per character) down to 1.62.
- **Experiments (ablations):** removing one component at a time shows what each is worth, e.g. without
  residual connections a 4-layer model learns no more than plain character frequencies.
- **Production habits:** a tested Python package (30 automated tests), a training script with checkpoints,
  resume and early stopping, and a streaming web app with input validation, rate limiting and security headers.

## Results

All numbers come from real runs on an Intel i5-1235U laptop CPU (no GPU), measured as **bits per character**
on the same held-out 10% of the book, so models with different tokenizers compare fairly. Lower is better.

| Model | Bits / char | Notes |
|---|---:|---|
| Knows nothing (uniform over 85 characters) | 6.41 | starting point |
| Character frequencies only | 4.39 | no context at all |
| Bigram (previous character) | 3.47 | 7K parameters |
| One self-attention head | 3.33 | 32-character context |
| 4 heads + feed-forward | 2.74 | 18K parameters |
| First full GPT (4 blocks) | 2.18 | 214K parameters, 6.5 min |
| Character-level GPT | 1.84 | 830K parameters, 37 min |
| BPE GPT (512 word pieces) | 1.69 | same budget, writes 2.4× faster |
| **BPE GPT + modern training upgrades** | **1.62** | **874K parameters, 32 min** (the deployed model) |

## How it works

```
text ──BPE tokenizer──► token ids (B, T)
     ──► token embedding + position embedding                     (B, T, C)
     ──► 4 × Block:  x = x + MultiHeadAttention(LayerNorm(x))     tokens look at earlier tokens
                     x = x + FeedForward(LayerNorm(x))            each token "thinks" on its own
     ──► LayerNorm ──► Linear (tied to the embedding) ──► scores for the next token (B, T, V)
     ──► softmax + temperature / top-k ──► sample the next token ──► repeat
```

Deployed model: 4 blocks × 4 heads, 128-dimensional embeddings, 128-token context (~280 characters),
512-token BPE vocabulary, 874,240 parameters.

Training: AdamW, learning-rate warm-up + cosine decay, weight decay, gradient clipping, dropout 0.1,
3,000 steps of 32 × 128 tokens, best checkpoint kept by validation loss.

## The build, step by step

Each notebook in [`notebooks/`](notebooks/) is one lesson, with explanations, tensor-shape diagrams and checks.
[`notes/glossary.md`](notes/glossary.md) explains every term; [`notes/progress.md`](notes/progress.md) records every measured result.

| # | Lesson | # | Lesson |
|---|---|---|---|
| 0 | Setup and the Tesla dataset | 9 | Multi-head attention + feed-forward |
| 1 | Tensors and shapes | 10 | The full GPT: residuals, LayerNorm, dropout |
| 2 | How a model learns (gradient descent) | 11 | Package + automated tests |
| 3 | Softmax, cross-entropy, sampling | 12 | Training script, checkpoints, temperature, top-k |
| 4 | Character tokenizer | 13 | Ablation experiments and overfitting |
| 5 | Batches and train/validation split | 14 | Byte-pair encoding tokenizer from scratch |
| 6 | Bigram language model | 15 | Characters vs word pieces, compared fairly |
| 7 | The averaging trick (causal masking) | 16 | Modern upgrades (fast attention, tied weights, LR schedule…) |
| 8 | One self-attention head | | |

## Tech stack

- **Python 3.12**, **PyTorch 2.14** (CPU), NumPy, `regex`
- **uv** for environments and dependencies; **pytest** for tests
- **Jupyter** notebooks (VS Code) for the lessons; **matplotlib** for plots; `tiktoken` only to compare with GPT-2/GPT-4 tokenizers
- **FastAPI** + **Uvicorn** for the web demo (streaming responses, no external services or API keys)
- **Render** (free tier) for hosting

## Project structure

```
src/tinygpt/
├─ tokenizer.py        character tokenizer
├─ bpe.py              byte-pair encoding tokenizer (GPT-2 style pre-split)
├─ model.py            GPTConfig, attention (hand-written and fast), Block, GPT, generation
├─ data.py             loading text, train/validation split, batches
├─ train.py            training script: checkpoints, resume, LR schedule, clipping, keep-awake
├─ sample.py           write text from a saved model
├─ export.py           slim copy of a checkpoint for serving
├─ download_tesla.py   fetch + clean the book from Project Gutenberg
├─ web.py              FastAPI web demo
└─ static/index.html   the demo page
models/tesla_modern.pt the trained model used by the demo (3.4 MB)
tests/                 30 tests: shapes, causality, parameter counts, tokenizers, web API
notebooks/             lessons 0–16
```

## Running it yourself

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                         # install everything
uv run pytest                                   # run the 30 tests
uv run uvicorn tinygpt.web:app --port 8000      # web demo at http://localhost:8000

uv run python -m tinygpt.sample --checkpoint models/tesla_modern.pt --prompt "The induction coil " --temperature 0.8
```

To train your own model:

```bash
uv run python -m tinygpt.download_tesla                       # data/input.txt (~977K characters)
uv run python -m tinygpt.train                                # character-level model (~40 min on a laptop CPU)
uv run python -m tinygpt.bpe                                  # learn the 512-token BPE tokenizer (~15 s)
uv run python -m tinygpt.train --tokenizer bpe --modern --name tesla_modern   # the deployed model (~30 min)
uv run python -m tinygpt.train --resume --name tesla_modern   # continue after Ctrl+C
uv run python -m tinygpt.export                               # slim copy into models/
```

> On the original Windows machine the project folder sits inside OneDrive, so the environment is kept outside it
> (`UV_PROJECT_ENVIRONMENT`, set for VS Code in `.vscode/settings.json`). Elsewhere, plain `uv sync` is all you need.

## Deployment

The demo runs on Render's free tier (settings in [`render.yaml`](render.yaml)). [`requirements.txt`](requirements.txt)
installs the **CPU-only** PyTorch build, which keeps the server install under 700 MB instead of several GB of GPU libraries.
The free server sleeps after 15 minutes without visitors, so the first request afterwards takes up to a minute.

The web API: `GET /health`, `GET /api/info` (model details), `POST /api/generate`
(`{"prompt": "...", "max_tokens": 1–250, "temperature": 0.1–1.5}`, streams plain text).
Limits: prompts up to 200 characters, 10 requests per minute per visitor, one generation at a time.

## Limitations

- **Small model, small data:** under 1M parameters trained on one ~1 MB book. It produces Tesla-flavoured text with
  invented words and no reliable meaning. It is a demonstration of how GPTs work, not a useful text generator.
- **Overfitting:** with word-piece tokens the model sees the book ~31 times in training; the train/validation gap grows,
  and more text would help more than a bigger model.
- **CPU only:** everything was trained on a laptop CPU; the same code runs on a GPU by changing the device.

## Credits

- Architecture and lesson sequence inspired by Andrej Karpathy's *Let's build GPT* and *Let's build the GPT Tokenizer*.
- Training text: *The Inventions, Researches and Writings of Nikola Tesla* by Thomas Commerford Martin (1894),
  public domain, via [Project Gutenberg](https://www.gutenberg.org/ebooks/39272).
- Built as a guided, hands-on course with Claude (Anthropic) as pair-programmer and tutor.
