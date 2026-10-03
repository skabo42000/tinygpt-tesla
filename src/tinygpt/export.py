"""Save a slim copy of a trained checkpoint for the web app (Lesson 18).

Keeps the model weights, its settings and its tokenizer; drops the optimizer state
(only needed to resume training), so the file is about 3x smaller and safe to commit.

Run:  uv run python -m tinygpt.export
"""
import argparse
import sys

import torch

from tinygpt.data import PROJECT_ROOT
from tinygpt.train import CHECKPOINT_DIR

MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL = MODELS_DIR / "tesla_modern.pt"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Export a trained checkpoint for serving")
    p.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "tesla_modern_best.pt"))
    p.add_argument("--out", default=str(DEFAULT_MODEL))
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    slim = {key: ckpt[key] for key in ("model", "config", "tokenizer", "step", "best_val", "minutes", "history")}
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(slim, args.out)
    print(f"Saved {args.out} (step {slim['step']}, validation loss {slim['best_val']:.3f})")


if __name__ == "__main__":
    main()
