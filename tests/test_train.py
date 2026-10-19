"""src.train.train: an interrupted and resumed run ends exactly where an uninterrupted one does; Val bpb."""
import json
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.train.train import Stream, ValSlice, lr_at, train

RUN = {"seed": 1939, "lr": 3e-3, "betas": [0.9, 0.95], "weight_decay": 0.1, "warmup_steps": 2, "min_lr_ratio": 0.1,
       "grad_clip": 1.0, "batch_tokens": 32, "micro_batch_rows": 1, "tokens": 8 * 32, "checkpoint_every_steps": 3,
       "eval_every_steps": 4, "log_gpu_temp_every_steps": 100, "stream": "main"}


def _stream(tmp_path):
    root = tmp_path / "main"
    root.mkdir()
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 50, 1000).astype("<u2")
    toks[:500].tofile(root / "a.bin")
    toks[500:].tofile(root / "b.bin")
    man = {"config": {"context": 16}, "rows": 999 // 16, "tokens": 1000,
           "shards": [{"file": "a.bin", "first_token": 0}, {"file": "b.bin", "first_token": 500}]}
    (root / "MANIFEST.json").write_text(json.dumps(man), encoding="utf-8")
    return Stream(root)


def _params(run_dir):
    sd = torch.load(sorted((run_dir / "checkpoints").glob("step_*.pt"))[-1], weights_only=False)["model"]
    return {k: v.clone() for k, v in sd.items()}


@pytest.mark.parametrize("model", [{"type": "lstm", "vocab_size": 50, "hidden": 8, "n_layer": 2},
                                   {"type": "transformer", "vocab_size": 50, "context": 16, "n_layer": 2, "d_model": 16,
                                    "n_head": 2}], ids=["lstm", "transformer"])
def test_resume_continues_exactly(tmp_path, model):
    torch.use_deterministic_algorithms(True)
    stream = _stream(tmp_path)
    run = RUN | {"model": model}
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    out_a = train(run, a, stream, [], "cpu", resume=False)
    train(run, b, stream, [], "cpu", resume=False, stop_after=3)        # "killed" after the step-3 checkpoint
    out_b = train(run, b, stream, [], "cpu", resume=True)
    assert out_a["step"] == out_b["step"] == 8
    pa_, pb = _params(a), _params(b)
    assert all(torch.equal(pa_[k], pb[k]) for k in pa_)
    la = [json.loads(x) for x in (a / "log.jsonl").read_text().splitlines()]
    lb = [json.loads(x) for x in (b / "log.jsonl").read_text().splitlines()]
    assert la[-1]["loss"] == lb[-1]["loss"]


def test_lr_schedule_warms_up_then_decays_to_the_floor():
    run = {"lr": 1.0, "warmup_steps": 10, "min_lr_ratio": 0.1}
    assert lr_at(0, 100, run) == pytest.approx(0.1) and lr_at(9, 100, run) == pytest.approx(1.0)
    assert lr_at(100, 100, run) == pytest.approx(0.1)
    assert lr_at(55, 100, run) == pytest.approx(0.55)


def test_val_bpb_of_a_uniform_model(tmp_path):
    root = tmp_path / "val"
    root.mkdir()
    docs = [np.array([5, 6, 7], "<u2"), np.array([8, 9], "<u2"), np.array([1, 2, 3, 4], "<u2")]
    buf, off = [np.zeros(1, "<u2")], []
    pos = 1
    for d in docs:
        off.append(pos)
        buf += [d, np.zeros(1, "<u2")]
        pos += len(d) + 1
    np.concatenate(buf).tofile(root / "en.bin")                         # 13 tokens -> 3 rows of 4
    pq.write_table(pa.table({"article_id": ["x", "y", "z"], "offset": off, "n_tokens": [3, 2, 4],
                             "n_bytes": [10, 6, 20]}), root / "en.idx.parquet")
    v = ValSlice(root, "en", 4)
    w = torch.randn(50, 8)
    r = v.evaluate(lambda x: torch.zeros(*x.shape, 8), w, "cpu", rows_per_batch=2)
    assert r["documents"] == 3                                          # every document ends inside 3 rows
    assert r["bpb"] == pytest.approx(9 * math.log2(50) / 36)
    assert r["loss"] == pytest.approx(math.log(50))
