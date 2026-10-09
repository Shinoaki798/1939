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


def _meta(n: int = 30000, seed: int = 5):
    """A synthetic candidate pool: periods, languages, categories, a little science."""
    rng = np.random.default_rng(seed)
    year = rng.integers(1900, 1940, n).astype(np.int16)
    period_name = np.where(year < 1920, "1900-19", np.where(year < 1930, "1920-29", "1930-39.06"))
    lang = (rng.random(n) < 0.25).astype(np.int8)                     # 1 = de
    cat_name = np.where(lang == 1, "newspaper", rng.choice(["newspaper", "books", "legal", "legislative"], n,
                                                             p=[0.7, 0.1, 0.15, 0.05]))
    sci = rng.random(n) < 0.08
    vocab = {"source": {"s": 0}, "bucket": {"general": 0, "science": 1},
             "category": {c: i for i, c in enumerate(["newspaper", "books", "legal", "legislative", "science"])},
             "period": {p: i for i, p in enumerate(["1930-39.06", "1920-29", "1900-19", "<1900"])}}
    ids = np.array([f"d{i}" for i in range(n)], dtype=object)
    h, u = mx.uniform(ids, 1)
    h = h ^ (lang.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))
    return {"h": h, "u": u, "tokens": rng.integers(200, 3000, n).astype(np.float64), "lang": lang,
            "source": np.zeros(n, dtype=np.int16), "bucket": sci.astype(np.int16),
            "category": np.array([vocab["category"]["science" if s else c] for s, c in zip(sci, cat_name)], dtype=np.int16),
            "period": np.array([vocab["period"][p] for p in period_name], dtype=np.int16), "year": year, "vocab": vocab}


def test_twin_replaces_german_period_by_period():
    meta = _meta()
    tot = meta["tokens"].sum()
    mcfg = {"seed": 1, "half_life_years": 5, "reference_year": 1939, "repeat_from_year": 1934, "max_epochs": 2,
            "caps": {"german": 0.25, "books": 0.12, "legal_of_english": 0.10, "science_of_language": 0.10,
                     "legal_absolute": {"1930-39.06": 0.02 * tot}},
            "science_tiers": {"en": [["*"]], "de": [["*"]]},
            "profiles": {"t": {"budget": 0.55 * tot, "science": {"en": 0.03 * tot, "de": 0.01 * tot},
                               "periods": {"1930-39.06": 0.5 * tot, "1920-29": 0.08 * tot, "1900-19": 0.03 * tot}}}}
    count, twin, s = mx.draw(meta, mcfg, "t")
    de, sci = meta["lang"] == 1, meta["bucket"] == 1
    assert twin.max() <= 2 and twin[de].sum() == 0 and count[de].sum() > 0
    for per, code in meta["vocab"]["period"].items():
        if per == "<1900":
            continue
        m = (meta["period"] == code) & ~sci
        main_seen, twin_seen = (meta["tokens"] * count)[m].sum(), (meta["tokens"] * twin)[m].sum()
        short = s["twin"]["shortfall"][per]
        assert abs((main_seen - twin_seen) - short) < 1 and -3000 <= short, per
        en = m & ~de
        assert (twin[en] >= count[en]).all(), per                            # English only grows
    y = meta["year"]
    extra = (twin == 2) & (count == 1)
    assert extra.any() and (y[extra] < 1934).all() and (y[extra] >= 1930).all()
    legal = meta["category"] == meta["vocab"]["category"]["legal"]
    assert not (legal & (meta["period"] == 0) & (twin == 2)).any()
    # this pool has less 1930-33 English than 1930s German: every candidate gets its second epoch, the rest
    # is a recorded shortfall (never a third epoch)
    cand = (meta["period"] == 0) & ~de & ~sci & (y < 1934) & ~legal & (count == 1)
    assert s["twin"]["shortfall"]["1930-39.06"] > 3000 and (twin[cand] == 2).sum() >= 0.99 * cand.sum()
    for per in ("1920-29", "1900-19"):
        assert abs(s["twin"]["shortfall"][per]) <= 3000, per
    assert abs((meta["tokens"] * twin)[sci].sum() - (meta["tokens"] * count)[sci].sum()) <= 3000
