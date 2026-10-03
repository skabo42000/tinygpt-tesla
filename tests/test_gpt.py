"""Sanity checks for the tokenizer, batching and GPT model. Run: uv run pytest"""
import math

import pytest
import torch

from tinygpt.data import get_batch
from tinygpt.model import GPT, GPTConfig
from tinygpt.tokenizer import CharTokenizer

SMALL = GPTConfig(vocab_size=10, block_size=8, n_embd=32, n_head=2, n_layer=2, dropout=0.0)


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(0)


def count_params_by_hand(c):
    """The formula from Lesson 11: embeddings + blocks + final LayerNorm + output layer."""
    C, V, T = c.n_embd, c.vocab_size, c.block_size
    per_block = (
        2 * C                         # ln1
        + 3 * C * C                   # query, key, value (no bias)
        + C * C + C                   # proj
        + 2 * C                       # ln2
        + C * 4 * C + 4 * C           # feed-forward widen
        + 4 * C * C + C               # feed-forward shrink
    )
    return V * C + T * C + c.n_layer * per_block + 2 * C + C * V + V


def test_tokenizer_round_trip():
    text = "Nikola Tesla, 1893:\nalternating current."
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text
    with pytest.raises(KeyError):
        tok.encode("ć")


def test_batch_targets_are_shifted_by_one():
    data = torch.arange(100)
    x, y = get_batch(data, batch_size=4, block_size=8)
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y, x + 1)


def test_forward_shapes():
    model = GPT(SMALL)
    idx = torch.randint(SMALL.vocab_size, (3, SMALL.block_size))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, SMALL.block_size, SMALL.vocab_size)
    assert loss.shape == ()


def test_initial_loss_is_close_to_know_nothing():
    model = GPT(GPTConfig(vocab_size=85, block_size=16, n_embd=32, n_head=2, n_layer=2, dropout=0.0))
    idx = torch.randint(85, (8, 16))
    _, loss = model(idx, torch.randint(85, (8, 16)))
    assert abs(loss.item() - math.log(85)) < 0.5


def test_no_peeking_at_the_future():
    model = GPT(SMALL).eval()
    idx = torch.randint(SMALL.vocab_size, (1, SMALL.block_size))
    changed = idx.clone()
    changed[0, 5] = (changed[0, 5] + 1) % SMALL.vocab_size   # change the token at position 5

    before, _ = model(idx)
    after, _ = model(changed)
    assert torch.allclose(before[:, :5], after[:, :5]), "positions 0-4 must not see position 5"
    assert not torch.allclose(before[:, 5:], after[:, 5:]), "positions 5+ should see the change"


def test_can_memorize_one_batch():
    model = GPT(SMALL)
    idx = torch.randint(SMALL.vocab_size, (4, SMALL.block_size))
    targets = torch.randint(SMALL.vocab_size, (4, SMALL.block_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(300):
        _, loss = model(idx, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.1


def test_parameter_count_matches_formula():
    for config in [SMALL, GPTConfig()]:
        assert GPT(config).num_params() == count_params_by_hand(config)


def test_top_k_one_always_picks_the_most_likely_token():
    model = GPT(SMALL).eval()
    start = torch.zeros((1, 1), dtype=torch.long)
    first = model.generate(start, max_new_tokens=10, top_k=1)
    second = model.generate(start, max_new_tokens=10, top_k=1)
    assert torch.equal(first, second), "top_k=1 leaves only one choice, so no randomness"


def test_generate_adds_tokens_and_handles_long_input():
    model = GPT(SMALL).eval()
    start = torch.zeros((1, SMALL.block_size + 5), dtype=torch.long)   # longer than the context
    out = model.generate(start, max_new_tokens=12)
    assert out.shape == (1, SMALL.block_size + 5 + 12)
    assert out.max().item() < SMALL.vocab_size
