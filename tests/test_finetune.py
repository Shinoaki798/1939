"""src.train.finetune: packed rows score only answers and their closing <|endoftext|>, examples never cross a
row, decade balancing keeps the first run's order when off, replay rows never repeat, and an interrupted and
resumed fine-tuning run (with and without replay) ends exactly where an uninterrupted one does."""
import json

import numpy as np
import torch

from src.eval.finetune_eval import em_f1, paired
from src.model.build import build
from src.train.finetune import balanced, epochs, finetune, pack, replay_rows

MODEL = {"type": "transformer", "vocab_size": 50, "context": 16, "n_layer": 2, "d_model": 16, "n_head": 2}
RUN = {"seed": 1939, "model": MODEL, "lr": 1e-3, "betas": [0.9, 0.95], "weight_decay": 0.1, "warmup_steps": 2,
       "min_lr_ratio": 0.1, "grad_clip": 1.0, "batch_rows": 2, "micro_batch_rows": 1, "dev_every_steps": 3,
       "checkpoint_every_steps": 3, "log_gpu_temp_every_steps": 100}


def test_pack_scores_answers_and_keeps_examples_whole():
    exs = [([5, 6, 7], [8, 9]), ([10, 11], [12]), (list(range(1, 20)), [3])]       # the last is too long
    x, y, skipped = pack(exs, 8)
    assert skipped == 1 and x.shape == (2, 8)
    assert x[0].tolist() == [0, 5, 6, 7, 8, 9, 0, 0]                 # 6 + 4 tokens do not fit one row of 8
    assert y[0].tolist() == [-1, -1, -1, 8, 9, 0, -1, -1]            # answer 8 9 and its <|endoftext|>
    assert x[1].tolist()[:4] == [0, 10, 11, 12] and y[1].tolist()[:4] == [-1, -1, 12, 0]


def _data():
    rng = np.random.default_rng(0)
    exs = [(rng.integers(1, 50, 4).tolist(), rng.integers(1, 50, 2).tolist()) for _ in range(30)]
    x, y, info = epochs(exs, 16, 2, 1939)
    dx, dy, _ = pack(exs[:4], 16)
    return x, y, (dx, dy), info


def test_resume_continues_exactly(tmp_path):
    torch.use_deterministic_algorithms(True)
    x, y, dev, info = _data()
    assert [i["epoch"] for i in info] == [0, 1] and not np.array_equal(x[:info[0]["rows"]], x[info[0]["rows"]:])
    torch.manual_seed(0)
    base = build(MODEL).state_dict()
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    out_a = finetune(RUN, a, base, x, y, dev, "cpu", resume=False)
    finetune(RUN, b, base, x, y, dev, "cpu", resume=False, stop_after=3)
    out_b = finetune(RUN, b, None, x, y, dev, "cpu", resume=True)
    assert out_a["step"] == out_b["step"] == len(x) // 2
    la = [json.loads(v) for v in (a / "log.jsonl").read_text().splitlines()]
    lb = [json.loads(v) for v in (b / "log.jsonl").read_text().splitlines()]
    assert la[-1]["dev_loss"] == lb[-1]["dev_loss"] and la[-1]["dev_loss"] < la[0]["dev_loss"]


def test_balanced_epochs_and_first_run_order():
    exs = [([1], [2])] * 10
    decade = np.array([1920] * 6 + [1930] * 2 + [1910] + [-1])
    for e in range(2):                                           # off: the first run's order, unchanged
        rng = np.random.default_rng(1939 + e)
        assert (rng.permutation(np.arange(10)) == np.random.default_rng(1939 + e).permutation(10)).all()
    picks = [np.sort(balanced(decade, np.random.default_rng(s))) for s in (0, 1)]
    for k in picks:
        assert (decade[k] == 1920).sum() == 2 and set(k[decade[k] != 1920]) == {6, 7, 8, 9}
    _, _, info = epochs(exs, 16, 2, 1939, decade)
    assert [i["examples"] for i in info] == [6, 6]


def test_replay_rows_fixed_and_without_repeats():
    r = replay_rows(1000, 20, 16, 1939)
    assert r.shape == (20, 16) and len(np.unique(r)) == 320 and (r == replay_rows(1000, 20, 16, 1939)).all()


class _Stream:
    def __init__(self, n, C):
        self.t, self.C = torch.from_numpy(np.random.default_rng(1).integers(0, 50, n * C + 1)), C

    def batch(self, row, n):
        t = self.t[row * self.C:(row + n) * self.C + 1]
        return t[:-1].view(n, self.C), t[1:].view(n, self.C)


def test_resume_with_replay_continues_exactly(tmp_path):
    torch.use_deterministic_algorithms(True)
    x, y, dev, _ = _data()
    run = {**RUN, "replay": {"stream": "main", "rows_per_step": 2, "weight": 1.0}}
    rep = (_Stream(100, 16), replay_rows(100, len(x) // 2, 2, 1939))
    torch.manual_seed(0)
    base = build(MODEL).state_dict()
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    val = {"en": (dev[0], np.where(dev[1] >= 0, dev[1], 1).astype(np.int32))}
    finetune(run, a, base, x, y, dev, "cpu", resume=False, replay=rep, val=val)
    finetune(run, b, base, x, y, dev, "cpu", resume=False, stop_after=3, replay=rep, val=val)
    finetune(run, b, None, x, y, dev, "cpu", resume=True, replay=rep, val=val)
    la = [json.loads(v) for v in (a / "log.jsonl").read_text().splitlines()]
    lb = [json.loads(v) for v in (b / "log.jsonl").read_text().splitlines()]
    assert la[-1]["dev_loss"] == lb[-1]["dev_loss"] and la[-1]["val_loss_en"] == lb[-1]["val_loss_en"]
    assert "replay_loss" in la[-1] and "val_loss_en" in la[0]


def test_compare_two_runs_on_the_same_items(tmp_path, monkeypatch):
    import pytest

    import src.eval.finetune_eval as fe
    monkeypatch.setattr(fe, "repo_path", lambda p: tmp_path / p)
    d = tmp_path / "reports" / "finetune"
    d.mkdir(parents=True)
    common = {"s_before": np.array([2.0, 4.0]), "s_weight": np.array([1.0, 1.0]), "t_f1_before": np.array([0.0, 1.0])}
    np.savez(d / "a_items.npz", **common, s_after=np.array([2.0, 2.0]), t_f1_after=np.array([1.0, 1.0]))
    np.savez(d / "b_items.npz", **common, s_after=np.array([1.0, 1.0]), t_f1_after=np.array([0.0, 1.0]))
    r = fe.compare("a", "b", 100, 0)["scores"]
    assert r["s"]["base"] == 3.0 and r["s"]["a"] == 2.0 and r["s"]["b"] == 1.0 and r["s"]["diff"] == 1.0
    assert r["t_f1"]["diff"] == 0.5 and r["t_f1"]["base"] == 0.5
    np.savez(d / "b_items.npz", **{**common, "s_before": np.array([2.0, 5.0])}, s_after=np.array([1.0, 1.0]),
             t_f1_after=np.array([0.0, 1.0]))
    with pytest.raises(SystemExit):
        fe.compare("a", "b", 100, 0)


def test_em_f1_and_paired_difference():
    assert em_f1("The Reginald Denny.", "reginald denny") == (1.0, 1.0)
    assert em_f1("Denny", "Reginald Denny")[1] == 2 / 3
    r = paired(np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0]), 100, 0)
    assert r["before"] == 1.0 and r["after"] == 0.5 and r["diff"] == 0.5 and r["diff_ci"] == [0.5, 0.5]
    r = paired(np.array([1.0, 0.0]), np.array([1.0, 1.0]), None, 100, 0)          # exact match 0.5 -> 1.0
    assert r["before"] == 0.5 and r["after"] == 1.0 and r["diff"] == -0.5
