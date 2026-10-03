"""Train the GPT on Tesla's text and save checkpoints (Lessons 12 and 15).

Run:  uv run python -m tinygpt.train              (fresh run, ~40 min on the laptop CPU)
      uv run python -m tinygpt.train --resume     (continue after stopping with Ctrl+C)
      uv run python -m tinygpt.train --tokenizer bpe --name tesla_bpe   (word-piece tokens, Lesson 15)
      uv run python -m tinygpt.train --tokenizer bpe --modern --name tesla_modern   (Lesson 16 upgrades)
      uv run python -m tinygpt.train --help       (all settings)
"""
import argparse
import math
import sys
import time
from dataclasses import asdict

import torch

from tinygpt.bpe import BPETokenizer
from tinygpt.data import PROJECT_ROOT, get_batch, load_text, split_data
from tinygpt.model import GPT, GPTConfig
from tinygpt.tokenizer import CharTokenizer

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_BPE = CHECKPOINT_DIR / "tesla_bpe_512.json"


def tokenizer_state(tokenizer):
    """What we store in a checkpoint so the exact tokenizer can be rebuilt."""
    if isinstance(tokenizer, BPETokenizer):
        return {"type": "bpe", "pattern": tokenizer.pattern,
                "merges": [[a, b, i] for (a, b), i in tokenizer.merges.items()]}
    return {"type": "char", "chars": tokenizer.chars}


def tokenizer_from_state(state):
    if state["type"] == "bpe":
        return BPETokenizer({(a, b): i for a, b, i in state["merges"]}, state["pattern"])
    return CharTokenizer(state["chars"])


def parse_args():
    p = argparse.ArgumentParser(description="Train the GPT on data/input.txt")
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--eval-interval", type=int, default=250, help="measure + save every N steps")
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--name", default="tesla", help="checkpoint name: checkpoints/<name>.pt")
    p.add_argument("--resume", action="store_true", help="continue from checkpoints/<name>.pt")
    p.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    p.add_argument("--bpe-path", default=str(DEFAULT_BPE), help="BPE tokenizer saved in Lesson 14")
    p.add_argument("--modern", action="store_true",
                   help="Lesson 16 upgrades: fast attention, tied weights, GPT-2 init, "
                        "warm-up + cosine learning rate, weight decay, gradient clipping")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    return p.parse_args()


def get_lr(step, args):
    """Lesson 16: climb linearly for warmup_steps, then glide down a cosine curve to min_lr."""
    if not args.modern:
        return args.lr
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))) * (args.lr - args.min_lr)


def make_optimizer(model, args):
    if not args.modern:
        return torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Weight decay gently pulls the big weight grids toward 0 (less memorizing);
    # biases and LayerNorm knobs are left alone.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95))


@torch.no_grad()
def estimate_loss(model, splits, batch_size, block_size, eval_batches):
    model.eval()
    result = {}
    for name, data in splits.items():
        losses = [model(*get_batch(data, batch_size, block_size))[1].item() for _ in range(eval_batches)]
        result[name] = sum(losses) / len(losses)
    model.train()
    return result


def save_checkpoint(path, model, optimizer, tokenizer, step, history, best_val, minutes, modern=False):
    torch.save(
        {
            "modern": modern,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(model.config),
            "tokenizer": tokenizer_state(tokenizer),
            "step": step,
            "history": history,
            "best_val": best_val,
            "minutes": minutes,
        },
        path,
    )


def keep_awake():
    """Ask Windows not to go to sleep while training runs (released automatically when it ends).
    The screen may still turn off; closing the laptop lid can still force sleep."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def load_model(path):
    """Load a trained model + its tokenizer from a checkpoint file."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    if "tokenizer" in ckpt:
        tokenizer = tokenizer_from_state(ckpt["tokenizer"])
    else:                                       # Lesson 12 checkpoints stored just the characters
        tokenizer = CharTokenizer(ckpt["chars"])
    return model, tokenizer, ckpt


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    keep_awake()
    torch.manual_seed(args.seed)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    last_path = CHECKPOINT_DIR / f"{args.name}.pt"         # latest state, for --resume
    best_path = CHECKPOINT_DIR / f"{args.name}_best.pt"    # lowest validation loss so far

    ckpt = torch.load(last_path, map_location="cpu", weights_only=False) if args.resume else None
    text = load_text()
    if ckpt is not None:
        tokenizer = tokenizer_from_state(ckpt["tokenizer"])   # resume with the exact same tokenizer
    elif args.tokenizer == "bpe":
        tokenizer = BPETokenizer.load(args.bpe_path)
    else:
        tokenizer = CharTokenizer.from_text(text)
    train_data, val_data = split_data(tokenizer.encode(text))
    splits = {"train": train_data, "val": val_data}

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=args.block_size, n_embd=args.n_embd,
        n_head=args.n_head, n_layer=args.n_layer, dropout=args.dropout,
        fast_attention=args.modern, tie_weights=args.modern, gpt2_init=args.modern,
    )
    model = GPT(config)
    optimizer = make_optimizer(model, args)
    start_step, history, best_val, minutes_before = 0, [], float("inf"), 0.0

    if ckpt is not None:
        args.modern = ckpt.get("modern", False)       # resume the same way it started
        config = GPTConfig(**ckpt["config"])        # the saved model's sizes win over the command line
        model = GPT(config)
        model.load_state_dict(ckpt["model"])
        optimizer = make_optimizer(model, args)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, history = ckpt["step"], ckpt["history"]
        best_val, minutes_before = ckpt["best_val"], ckpt["minutes"]
        print(f"Resuming from step {start_step} (best val so far {best_val:.3f})")

    print(f"{model.num_params():,} parameters | {config}")
    print(f"Training to step {args.max_steps}; progress every {args.eval_interval} steps. Ctrl+C to stop.\n")
    start = time.time()
    minutes = lambda: minutes_before + (time.time() - start) / 60

    grad_norms = []    # size of the gradient each step (Lesson 16), averaged into the history
    try:
        for step in range(start_step, args.max_steps + 1):
            is_eval_step = step % args.eval_interval == 0 or step == args.max_steps
            if is_eval_step and not (args.resume and step == start_step):
                losses = estimate_loss(model, splits, args.batch_size, config.block_size, args.eval_batches)
                avg_norm = sum(grad_norms) / len(grad_norms) if grad_norms else None
                history.append({"step": step, **losses, "minutes": round(minutes(), 2),
                                "lr": get_lr(step, args), "grad_norm": avg_norm})
                grad_norms = []
                marker = ""
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    save_checkpoint(best_path, model, optimizer, tokenizer, step, history, best_val,
                                    minutes(), args.modern)
                    marker = "  (best so far, saved)"
                save_checkpoint(last_path, model, optimizer, tokenizer, step, history, best_val,
                                minutes(), args.modern)
                print(f"step {step:5d}/{args.max_steps}:  train {losses['train']:.3f}   "
                      f"val {losses['val']:.3f}   {minutes():5.1f} min{marker}", flush=True)
            if step == args.max_steps:
                break

            for group in optimizer.param_groups:
                group["lr"] = get_lr(step, args)
            xb, yb = get_batch(train_data, args.batch_size, config.block_size)
            _, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            # With --modern, shrink any gradient bigger than grad_clip (a rare bad batch can't
            # throw the model off). Without it, just measure the size (max_norm = infinity).
            limit = args.grad_clip if args.modern else float("inf")
            grad_norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), limit).item())
            optimizer.step()
    except KeyboardInterrupt:
        print(f"\nStopped. Progress up to the last saved step is in {last_path.name}; "
              f"continue with:  uv run python -m tinygpt.train --resume")
        return

    print(f"\nDone in {minutes():.1f} min. Best validation loss: {best_val:.3f}  ->  {best_path}")
    model.eval()
    start_text = torch.tensor([tokenizer.encode("\n")])
    print("\n--- sample ---")
    print(tokenizer.decode(model.generate(start_text, 300, temperature=0.8)[0].tolist()))


if __name__ == "__main__":
    main()
