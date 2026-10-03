"""Byte-Pair Encoding tokenizer, the kind GPT-2/GPT-4 use (Lesson 14).

Start from the 256 possible byte values, then repeatedly merge the most common
neighbouring pair into a new token. Text is first split into word-like chunks
(GPT-2's pattern) so merges never glue words and punctuation together.
"""
import json
from collections import Counter

import regex

# GPT-2's split pattern: contractions, words (with their leading space), numbers, punctuation, spaces.
GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges=None, pattern=GPT2_PATTERN):
        self.merges = dict(merges or {})                  # (id, id) -> new id, in the order learned
        self.pattern = pattern
        self.splitter = regex.compile(pattern)
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> the bytes it stands for
        for (a, b), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    @property
    def vocab_size(self):
        return len(self.vocab)

    @classmethod
    def train(cls, text, vocab_size, pattern=GPT2_PATTERN, verbose=False):
        """Learn vocab_size - 256 merges from the text."""
        chunk_counts = Counter(regex.findall(pattern, text))
        words = {tuple(chunk.encode("utf-8")): n for chunk, n in chunk_counts.items()}
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        for new_id in range(256, vocab_size):
            pair_counts = Counter()
            for word, n in words.items():
                for pair in zip(word, word[1:]):
                    pair_counts[pair] += n
            if not pair_counts:
                break
            pair = max(pair_counts, key=pair_counts.get)
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            new_words = Counter()
            for word, n in words.items():
                new_words[tuple(merge(list(word), pair, new_id))] += n
            words = new_words
            if verbose:
                print(f"merge {new_id - 255:4d}: {vocab[pair[0]]!r} + {vocab[pair[1]]!r} -> {vocab[new_id]!r}"
                      f"  ({pair_counts[pair]:,} times)")
        return cls(merges, pattern)

    def encode(self, text):
        ids = []
        for chunk in self.splitter.findall(text):
            chunk_ids = list(chunk.encode("utf-8"))
            while len(chunk_ids) >= 2:
                # apply the earliest-learned merge that's possible here
                pair = min(zip(chunk_ids, chunk_ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                chunk_ids = merge(chunk_ids, pair, self.merges[pair])
            ids.extend(chunk_ids)
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_strings(self, ids):
        """Each token as readable text, for display."""
        return [self.vocab[i].decode("utf-8", errors="replace") for i in ids]

    def save(self, path):
        data = {"pattern": self.pattern, "merges": [[a, b, i] for (a, b), i in self.merges.items()]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls({(a, b): i for a, b, i in data["merges"]}, data["pattern"])
