"""src.eval.crosslingual: passages start at the hit line, mismatches never share the topic, target positions and
the copy-free token mask are right, a uniform model shows no priming, and the topic bootstrap brackets the value."""
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.eval.crosslingual import (assign_mismatch, contexts, effects, letter_share, passage, per_item, scored,
                                   target_nll)


class CharTok:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[ord(ch) for ch in text])

    def decode(self, ids):
        return "".join(map(chr, ids))


def test_passage_starts_at_the_line_of_the_first_hit():
    text = "Weather today.\nA big EARTHQUAKE hit Chile.\nMore later."
    ids = passage(text, r"\bearthquakes?\b", CharTok(), 12)
    assert CharTok().decode(ids) == "A big EARTHQ"
    assert passage("nothing here", r"\bearthquake\b", CharTok(), 12) == []
    long = "word " * 100 + "the earthquake struck"                  # one long line, as on a newspaper page
    out = CharTok().decode(passage(long, r"\bearthquake\b", CharTok(), 300))
    assert out.startswith("word ") and out.endswith("struck") and len(out) <= 220


def test_letter_share_flags_ocr_debris():
    assert letter_share("Der Reichstag trat heute zusammen.") > 0.9
    assert letter_share("V - -»»»>»» >»» « ^»^»»^/»»» ^«") < 0.5


def test_mismatch_is_another_topic_and_passes_the_check():
    topics = ["a", "a", "b", "b", "c"]
    mis = assign_mismatch(topics, lambda i, j: True, seed=0)
    assert all(topics[i] != topics[j] for i, j in enumerate(mis))
    assert len(set(mis)) == len(mis)                                 # each item lends its contexts once
    mis = assign_mismatch(topics, lambda i, j: topics[j] == "c" or topics[i] == "c", seed=0)
    assert all(topics[j] == "c" for i, j in enumerate(mis) if topics[i] != "c")


class Uniform(torch.nn.Module):
    output_weight = torch.zeros(60, 4)

    def forward(self, x):
        return torch.zeros(*x.shape, 4)


def test_target_positions_and_uniform_model():
    ctxs = [np.array([5, 6, 7]), np.array([8])]
    targets = [np.array([9, 10]), np.array([11, 12, 13])]
    out = target_nll(Uniform(), ctxs, targets, "cpu", batch_rows=2)
    assert [len(o) for o in out] == [2, 3]
    assert np.allclose(np.concatenate(out), math.log(60))


def items_frame():
    return pd.DataFrame({"topic": ["a", "b"], "target": [np.array([1, 2, 3]), np.array([4, 5, 6])],
                         "de_ctx": [np.array([2, 20]), np.array([30])], "en_ctx": [np.array([21]), np.array([5])],
                         "mis": [1, 0]})


def test_novel_mask_drops_tokens_seen_in_any_context():
    items = items_frame()
    ctx = contexts(items)
    m = scored(items, ctx, "novel")
    assert m[0].tolist() == [True, False, True]       # 2 is in its German context
    assert m[1].tolist() == [True, False, True]       # 5 is in its English context
    assert all(x.all() for x in scored(items, ctx, "all"))


def test_no_priming_gives_zero_effect_and_ci_brackets_it():
    items = items_frame()
    same = [np.full(3, 2.0), np.full(3, 2.0)]
    nll = {k: same for k in ("de_match", "de_mis", "en_match", "en_mis")}
    d = per_item(nll, scored(items, contexts(items), "all"), [np.ones(3), np.ones(3)])
    e = effects(d, items["topic"].to_numpy(), 200, 0)
    assert e["de"] == 0 and e["en"] == 0
    nll = dict(nll, de_mis=[np.full(3, 3.0), np.full(3, 2.5)], en_mis=[np.full(3, 4.0), np.full(3, 4.0)])
    d = per_item(nll, scored(items, contexts(items), "all"), [np.ones(3), np.ones(3)])
    e = effects(d, items["topic"].to_numpy(), 200, 0)
    assert e["de"] == round(4.5 / math.log(2) / 6, 5) and e["en"] == round(12 / math.log(2) / 6, 5)
    assert e["de_ci"][0] <= e["de"] <= e["de_ci"][1]
    assert abs(e["ratio"] - 4.5 / 12) < 1e-3
