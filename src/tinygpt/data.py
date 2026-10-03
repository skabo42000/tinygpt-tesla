"""Loading the text and cutting it into training batches (Lessons 4-5)."""
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEXT = PROJECT_ROOT / "data" / "input.txt"


def load_text(path=DEFAULT_TEXT):
    return Path(path).read_text(encoding="utf-8")


def split_data(ids, train_fraction=0.9):
    """Turn token ids into a tensor and split it into training and validation parts."""
    data = torch.tensor(ids, dtype=torch.long)
    n = int(train_fraction * len(data))
    return data[:n], data[n:]


def get_batch(data, batch_size, block_size, device="cpu"):
    """Pick random chunks of length block_size; targets are the same chunks shifted by one."""
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])          # (B, T)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])  # (B, T)
    return x.to(device), y.to(device)
