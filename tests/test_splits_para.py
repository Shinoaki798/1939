"""Split membership (src.data.splits) and paragraph-level dedup helpers (src.data.para_dedup)."""

import numpy as np

from src.data import para_dedup, splits
from src.data.dedup import comparison_tokens, shingles


def test_split_rules():
    assert splits.parent_id("sciam_ia_sim_x_1901_c004") == "sciam_ia_sim_x_1901"
    assert splits.bucket("book_c001") == splits.bucket("book_c017")           # chunks share a split
    assert splits.split_of("anything", "embargo") == "embargo"
    ids = [f"doc{i}" for i in range(20000)]
    pre = [splits.split_of(a, "pre") for a in ids]
    post = [splits.split_of(a, "post") for a in ids]
    assert abs(pre.count("holdout") / len(ids) - 0.02) < 0.005
    assert abs(pre.count("val") / len(ids) - 0.02) < 0.005
    assert set(post) == {"holdout", "post"}                                      # no val or train after the cutoff
    assert all(p == "holdout" for a, p, q in zip(ids, pre, post) if q == "holdout")  # same bucket either side
    assert splits.test_set("1935-04-01", "holdout") == "A"
    assert splits.test_set("1939-03-01", "holdout") is None                      # 1939 H1 is held out, not Test-A
    assert splits.test_set("1942-01-01", "holdout") == "B"
    assert splits.test_set("1950-01-01", "holdout") == "C"
    assert splits.test_set("1935-04-01", "train") is None


def test_blocks_cover_text_at_line_boundaries():
    lines = [f"line {i} " + "word " * 9 for i in range(50)]                     # 11 words per line
    text = "\n".join(lines) + "\nshort tail"
    spans = para_dedup.blocks(text)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))                  # contiguous, no gaps
    assert all(text[e - 1] == "\n" for _, e in spans[:-1])                      # cut after a line break
    assert all(len(text[s:e].split()) >= para_dedup.BLOCK_WORDS for s, e in spans[:-1])
    assert len(text[spans[-1][0]:].split()) >= para_dedup.MIN_TAIL


def test_containment_and_bloom():
    a = "the council passed the ordinance over the mayor's veto by a vote of six to two on monday evening"
    ref = np.unique(shingles(comparison_tokens(a + " and adjourned")))
    assert para_dedup._containment(np.unique(shingles(comparison_tokens(a))), ref) == 1.0
    other = "steam engines and iron bridges were the wonder of the age in every county of the state"
    assert para_dedup._containment(np.unique(shingles(comparison_tokens(other))), ref) == 0.0
    bloom = para_dedup.Bloom(10_000)
    h = np.unique(shingles(comparison_tokens(a)))
    assert not bloom.contains(h).any()
    bloom.add(h)
    assert bloom.contains(h).all()
    probe = np.arange(1, 5001, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    assert bloom.contains(probe).mean() < 0.01                                  # false positives stay rare
