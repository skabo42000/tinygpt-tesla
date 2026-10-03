"""The GPT model (Lessons 8-10), with all its size settings in one GPTConfig."""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 85    # V: how many different tokens
    block_size: int = 64    # T: how far back the model can look
    n_embd: int = 64        # C: numbers per token
    n_head: int = 4         # attention heads per block
    n_layer: int = 4        # how many blocks are stacked
    dropout: float = 0.1    # fraction of numbers switched off during training


class Head(nn.Module):
    """One self-attention head: query, key, value, causal mask (Lesson 8)."""

    def __init__(self, config, head_size):
        super().__init__()
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)            # (B, T, hs) each
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5            # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))   # no peeking at the future
        wei = F.softmax(wei, dim=-1)
        self.last_wei = wei.detach()                                   # kept for plotting
        wei = self.dropout(wei)
        return wei @ v                                                 # (B, T, hs)


class MultiHeadAttention(nn.Module):
    """Several heads side by side, joined and mixed by a projection (Lessons 9-10)."""

    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, C)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Each token thinks on its own: widen x4, ReLU, shrink back (Lesson 9)."""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One round of "talk, then think", with residual connections and LayerNorm (Lesson 10)."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # talk
        x = x + self.ffwd(self.ln2(x))   # think
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)    # (B, T, C)
        x = self.blocks(x)                           # (B, T, C)
        logits = self.lm_head(self.ln_f(x))          # (B, T, V)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Write max_new_tokens more tokens (Lesson 12).

        temperature < 1: safer, more repetitive; > 1: more adventurous, more mistakes.
        top_k: only the k most likely tokens may be picked at each step.
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.config.block_size :])
            logits = logits[:, -1, :] / temperature                     # (B, V)
            if top_k is not None:
                kth_best = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth_best, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
