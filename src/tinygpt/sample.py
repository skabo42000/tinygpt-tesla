"""Write text with a trained GPT (Lesson 12).

Run:  uv run python -m tinygpt.sample --prompt "The alternating current" --temperature 0.8 --top-k 10
"""
import argparse
import sys

import torch

from tinygpt.train import CHECKPOINT_DIR, load_model


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Generate text from a saved checkpoint")
    p.add_argument("--checkpoint", default=str(CHECKPOINT_DIR / "tesla_best.pt"))
    p.add_argument("--prompt", default="\n", help="text to start from")
    p.add_argument("--tokens", type=int, default=500, help="how many characters to write")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    model, tokenizer, _ = load_model(args.checkpoint)
    unknown = sorted(set(args.prompt) - set(getattr(tokenizer, "chars", args.prompt)))   # BPE knows every character
    if unknown:
        sys.exit(f"The prompt contains characters the model doesn't know: {unknown}")

    idx = torch.tensor([tokenizer.encode(args.prompt)])
    out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
