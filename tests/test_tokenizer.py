"""src.data.tokenizer: a prefix of a longer BPE run equals a run stopped there; digits split; byte map."""
import random

from src.data import tokenizer as tk


def _corpus(n: int = 3000) -> list[str]:
    rng = random.Random(0)
    words = ("the council passed ordinance mayor veto Regierung Erlass veröffentlicht Zeitung Straße "
             "railroad company station Reichstag Abgeordneten 1939 1934").split()
    return [" ".join(rng.choice(words) for _ in range(rng.randint(5, 30))) for _ in range(n)]


def test_prefix_equals_short_run():
    texts = _corpus()
    long_run = tk.train_bpe(iter(texts), 600)
    short_run = tk.train_bpe(iter(texts), 450)
    cut = tk.truncate(long_run, 450)
    assert cut.get_vocab() == short_run.get_vocab()
    for t in texts[:200]:
        assert cut.encode(t).ids == short_run.encode(t).ids


def test_digits_are_single_tokens_and_text_round_trips():
    tok = tk.train_bpe(iter(_corpus()), 500)
    ids = tok.encode("Am 1. September 1939 begann").ids
    pieces = [tok.id_to_token(i) for i in ids]
    assert all(len(p) == 1 for p in pieces if any(ch.isdigit() for ch in p))
    assert tok.decode(ids) == "Am 1. September 1939 begann"


def test_byte_decoder_covers_the_alphabet():
    dec = tk.byte_decoder()
    assert len(dec) == 256 and tk.token_text("Ġthe", dec) == " the" and tk.token_text("Ã¤", dec) == "ä"
