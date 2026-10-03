"""The GPT model (Lessons 8-10), with all its size settings in one GPTConfig.

Lesson 16 adds three optional upgrades (all off by default, so older checkpoints load unchanged).
"""
import math
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
    fast_attention: bool = False   # Lesson 16: all heads in one go with PyTorch's built-in attention
    tie_weights: bool = False      # Lesson 16: output layer shares the token embedding table
    gpt2_init: bool = False        # Lesson 16: GPT-2's starting values (small, scaled by depth)


class CausalSelfAttention(nn.Module):
    """The same multi-head attention as below, computed for all heads at once (Lesson 16).

    q, k, v for every head come from one Linear layer; the heads become an extra batch dimension;
    F.scaled_dot_product_attention does q @ k^T / sqrt(hs), the causal mask, softmax and @ v.
    """

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.proj.is_residual_output = True
        self.attn_dropout = config.dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)                          # (B, T, C) each
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_dropout if self.training else 0.0
        )                                                               # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)               # join the heads: (B, T, C)
        return self.dropout(self.proj(y))


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
        self.proj.is_residual_output = True
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
        self.net[2].is_residual_output = True

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One round of "talk, then think", with residual connections and LayerNorm (Lesson 10)."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config) if config.fast_attention else MultiHeadAttention(config)
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
        if config.tie_weights:
            # Reading a token and writing it use the same table: (V, C) serves both directions.
            self.lm_head.weight = self.tok_emb.weight
        if config.gpt2_init:
            self.apply(self._gpt2_init)

    def _gpt2_init(self, module):
        """Small starting values (std 0.02); layers that add onto the residual path start even
        smaller, so that stacking many blocks doesn't make the residual stream grow."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = 0.02
            if getattr(module, "is_residual_output", False):
                std /= math.sqrt(2 * self.config.n_layer)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())   # a shared (tied) table counts once

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
