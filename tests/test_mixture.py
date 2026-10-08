"""src.data.mixture: repetition only from 1934, caps, recency order, science tiers, determinism."""
import numpy as np

from src.data import mixture as mx

CAPS = {"german": 0.25, "books": 0.12, "legal_of_english": 0.10}


def _units(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    year = rng.integers(1930, 1940, n).astype(np.int16)
    cls = rng.choice([mx.FREE, mx.BOOKS, mx.LEGAL, mx.GERMAN], n, p=[0.5, 0.05, 0.2, 0.25]).astype(np.int8)
    tok = rng.integers(100, 5000, n).astype(np.float64)
    _, u = mx.uniform(np.array([f"doc{i}" for i in range(n)], dtype=object), 20261008)
    return year, cls, tok, mx.race_keys(u, year, 5, 1939)


def _shares(tok, cls, count):
    seen = tok * count
    tot, en = seen.sum(), seen[cls != mx.GERMAN].sum()
    return seen[cls == mx.GERMAN].sum() / tot, seen[cls == mx.BOOKS].sum() / tot, seen[cls == mx.LEGAL].sum() / en, tot


def test_1930s_repeats_only_from_1934_and_respects_caps():
    year, cls, tok, key = _units(20000)
    second = year >= 1934
    count = mx.plan(tok, cls, key, second, target=1e12, caps=CAPS, legal_absolute=0.05 * tok.sum())
    assert count.max() == 2 and not (count[~second] == 2).any()
    german, books, legal, tot = _shares(tok, cls, count)
    assert german <= 0.25 + 1e-6 and books <= 0.12 + 1e-6 and legal <= 0.10 + 1e-6
    assert (tok * count)[cls == mx.LEGAL].sum() <= 0.05 * tok.sum() + 1
    # uncapped English is taken in full, twice from 1934 on
    free = cls == mx.FREE
    assert (count[free] == np.where(second[free], 2, 1)).all()
    # a capped class gives up second epochs before first ones
    g = cls == mx.GERMAN
    assert (count[g] >= 1).all() and (count[g] == 2).sum() < (g & second).sum()


def test_sampled_period_hits_target_and_prefers_recent_years():
    rng = np.random.default_rng(1)
    n = 50000
    year = rng.integers(1920, 1930, n).astype(np.int16)
    cls = np.full(n, mx.FREE, dtype=np.int8)
    tok = np.full(n, 1000.0)
    _, u = mx.uniform(np.array([f"p{i}" for i in range(n)], dtype=object), 7)
    count = mx.plan(tok, cls, mx.race_keys(u, year, 5, 1939), np.zeros(n, dtype=bool), target=0.3 * tok.sum(), caps=CAPS)
    assert count.max() == 1
    assert abs((tok * count).sum() - 0.3 * tok.sum()) <= 1000
    rate = [count[year == y].mean() for y in (1920, 1925, 1929)]
    assert rate[0] < rate[1] < rate[2]


def test_science_tiers_fill_in_order():
    src = np.array(["pnas_ia"] * 100 + ["sciam_ia"] * 100 + ["gutenberg_sci_en"] * 100, dtype=object)
    _, u = mx.uniform(np.array([f"s{i}" for i in range(300)], dtype=object), 3)
    tiers = [["pnas_ia", "nature_ia"], ["*"], ["gutenberg_sci_en"]]
    key = mx.science_key(src, u, tiers)
    tok = np.full(300, 10.0)
    count = mx.plan(tok, np.zeros(300, dtype=np.int8), key, np.ones(300, dtype=bool), target=1500.0)
    assert (count[:100] == 1).all() and count[100:200].sum() == 50 and count[200:].sum() == 0
    count = mx.plan(tok, np.zeros(300, dtype=np.int8), key, np.ones(300, dtype=bool), target=3500.0)
    assert (count >= 1).all() and (count == 2).sum() == 50 and count[100:].max() == 1   # 2nd epochs: tier 0 first


def test_uniform_is_deterministic_and_seeded():
    ids = np.array(["a", "b", "c"], dtype=object)
    h1, u1 = mx.uniform(ids, 1)
    h2, u2 = mx.uniform(ids, 1)
    h3, _ = mx.uniform(ids, 2)
    assert (h1 == h2).all() and (u1 == u2).all() and not (h1 == h3).all()
    assert ((u1 > 0) & (u1 < 1)).all()
