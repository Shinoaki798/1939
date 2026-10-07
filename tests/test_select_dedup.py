"""Selection rules and MinHash dedup (src.data.select, src.data.dedup)."""

import numpy as np
import pyarrow as pa

from src.data import dedup, select


def test_normalize_german_typography():
    assert select.normalize("Muͤnchen und Vorzuͤge", "de") == "München und Vorzüge"
    assert select.normalize("Waſſer Aͤrzte", "de") == "Wasser Ärzte"
    assert select.normalize("Muͤnchen ſ", "en") == "Muͤnchen ſ"                 # English text untouched
    assert select.normalize("über", "en") == "über"                        # NFC for every language


def test_date_classes():
    assert select.date_class("1939-06-30", False) == "pre"
    assert select.date_class("1939-07-01", False) == "embargo"
    assert select.date_class("1939-08-31", False) == "embargo"
    assert select.date_class("1939-09-01", False) == "post"
    assert select.date_class("1955-12-31", False) == "post"
    assert select.date_class("1956-01-01", False) is None
    assert select.date_class("1899-12-31", False) is None                        # general text before 1900
    assert select.date_class("1899-12-31", True) == "pre"                         # science: any year


def test_unit_hash_is_fixed_and_uniform():
    a = select.unit_hash("ca_sn123_1930-01-01_ed1_seq1", "pre1920")
    assert a == select.unit_hash("ca_sn123_1930-01-01_ed1_seq1", "pre1920")
    assert a != select.unit_hash("ca_sn123_1930-01-01_ed1_seq1", "split")
    xs = [select.unit_hash(str(i), "t") for i in range(4000)]
    assert 0 <= min(xs) and max(xs) < 1 and abs(np.mean(xs) - 0.5) < 0.03


def test_doc_lang():
    assert select.doc_lang("chronicling_america", "en", 0.6, None, "", 1930) == "en"
    assert select.doc_lang("chronicling_america", "en", 0.1, None, "", 1930) is None
    assert select.doc_lang("ddb_newspapers_de", "de", 0.0, None, "", 1930) == "de"
    assert select.doc_lang("jfm", "und", 0.0, None, "", 1930) is None
    titles = {"1930": {"sn1"}}
    assert select.doc_lang("american_stories", "en", 0.6, titles, "sn1", 1930) == "en"
    assert select.doc_lang("american_stories", "en", 0.6, titles, "sn2", 1930) is None


WORDS = ("the of and to in a is that for it was on with as by at he his be from this have had not are but "
         "which or were they river bridge engine coal iron steam council mayor ordinance farmers bank").split()


def text(n, seed):
    r = np.random.default_rng(seed)
    return " ".join(WORDS[i] for i in r.integers(0, len(WORDS), n))


def test_minhash_similarity():
    a = dedup.comparison_tokens(text(400, 1))
    b = list(a)
    b[200] = "zeppelin"                                   # one changed word spoils 5 of ~396 shingles
    c = dedup.comparison_tokens(text(400, 2))
    assert dedup.minhash(["too", "short"]) is None
    assert (dedup.minhash(a) == dedup.minhash(a)).all()
    assert dedup.jaccard(dedup.minhash(a), dedup.minhash(b)) > 0.9
    assert dedup.jaccard(dedup.minhash(a), dedup.minhash(c)) < 0.2


def test_match_keeps_earliest_and_finds_exact():
    base = text(300, 3)
    edited = base.replace("bridge", "brldge", 1)          # OCR-style variant
    docs = [("late_copy", "1931-05-01", False, edited), ("original", "1930-05-01", False, base),
            ("other", "1930-01-01", False, text(300, 4)), ("dup_a", "1932-01-01", True, "Exact same notice " * 3),
            ("dup_b", "1932-01-01", False, "exact same notice " * 3)]
    toks = [dedup.comparison_tokens(d[3]) for d in docs]
    sigs = np.stack([dedup.minhash(t) if dedup.minhash(t) is not None else np.zeros(dedup.NUM_PERM, np.uint32)
                     for t in toks])
    idx = pa.table({"article_id": [d[0] for d in docs], "source": ["s"] * len(docs), "date": [d[1] for d in docs],
                    "page_level": [d[2] for d in docs], "n_words": [len(t) for t in toks],
                    "exact": [dedup.exact_hash(t) for t in toks],
                    "has_sig": [dedup.minhash(t) is not None for t in toks]})
    thr = np.full(len(docs), dedup.THRESHOLD_OCR)
    dropped, kept, jd, reason, uni, _ = dedup.match(sigs, idx, thr)
    got = {docs[d][0]: (docs[k][0], r) for d, k, r in zip(dropped, kept, reason)}
    assert got["late_copy"] == ("original", "near")
    assert got["dup_a"] == ("dup_b", "exact")             # same date: article-level beats page-level
    assert "original" not in got and "other" not in got


def test_components_chain():
    rank = np.array([2, 0, 1, 3])
    lab = dedup.components(4, np.array([0, 2]), np.array([2, 1]), rank)
    assert lab.tolist() == [0, 0, 0, 3]


def test_threshold_by_source_type():
    base = dedup.comparison_tokens(text(300, 5))
    noisy = list(base)
    for i in range(0, len(noisy), 28):                     # ~4 % of words misread: J ~ 0.69
        noisy[i] = noisy[i] + "x"
    toks = [base, noisy]
    sigs = np.stack([dedup.minhash(t) for t in toks])
    j = float(dedup.jaccard(sigs[0], sigs[1]))
    assert dedup.THRESHOLD_OCR <= j < dedup.THRESHOLD_KEYED
    idx = pa.table({"article_id": ["a", "b"], "source": ["s", "s"], "date": ["1930-01-01", "1930-02-01"],
                    "page_level": [False, False], "n_words": [300, 300],
                    "exact": [dedup.exact_hash(t) for t in toks], "has_sig": [True, True]})
    d_ocr = dedup.match(sigs, idx, np.array([0.6, 0.6]))[0]
    d_mixed = dedup.match(sigs, idx, np.array([0.8, 0.6]))[0]      # a pair with an OCR side uses 0.60
    d_keyed = dedup.match(sigs, idx, np.array([0.8, 0.8]))[0]
    assert d_ocr.tolist() == [1] and d_mixed.tolist() == [1] and d_keyed.tolist() == []
