"""Checks for the BPE tokenizer (Lesson 14). Run: uv run pytest"""
from tinygpt.bpe import BPETokenizer, merge

TRAIN_TEXT = "the theory of the thermal current; the current in the coil. " * 20


def test_merge_replaces_every_pair():
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]


def test_round_trip_any_language():
    tok = BPETokenizer.train(TRAIN_TEXT, vocab_size=300)
    for text in [TRAIN_TEXT, "Nikola Tesla, Smiljan: ćevapi i đak", "Никола Тесла ⚡", ""]:
        assert tok.decode(tok.encode(text)) == text


def test_merges_make_text_shorter():
    tok = BPETokenizer.train(TRAIN_TEXT, vocab_size=300)
    assert 256 < tok.vocab_size <= 300      # stops early once every word is a single token
    assert len(tok.encode(TRAIN_TEXT)) < len(TRAIN_TEXT.encode("utf-8")) / 3


def test_save_and_load(tmp_path):
    tok = BPETokenizer.train(TRAIN_TEXT, vocab_size=280)
    tok.save(tmp_path / "bpe.json")
    loaded = BPETokenizer.load(tmp_path / "bpe.json")
    assert loaded.encode(TRAIN_TEXT) == tok.encode(TRAIN_TEXT)
