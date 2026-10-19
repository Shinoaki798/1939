"""src.data.stream: halves, copies apart, determinism, twin order, chunk order, packing with <|endoftext|>."""
import json

import numpy as np
import pandas as pd

from src.data import stream as st
from src.train.train import Stream

SEED = 20261019


def _parents(n, two_share=0.3, seed=0):
    rng = np.random.default_rng(seed)
    parents = np.array([f"p{i}" for i in range(n)], dtype=object)
    count = np.where(rng.random(n) < two_share, 2, 1).astype(np.int64)
    return parents, count


def test_twice_seen_parents_have_one_copy_per_half_and_halves_are_balanced():
    parents, count = _parents(20000)
    o = st.order_copies(parents, count, SEED)
    assert len(o) == count.sum()
    halves = o.groupby("parent")["half"].agg(["min", "max", "size"])
    two = halves["size"] == 2
    assert (halves.loc[two, "min"] == 0).all() and (halves.loc[two, "max"] == 1).all()
    assert (o["half"].diff().fillna(0) >= 0).all()                  # all of half 0, then all of half 1
    share = (o["half"] == 0).mean()
    assert abs(share - 0.5) < 0.02
    once = o[o["parent"].isin(halves.index[~two])]
    assert abs((once["half"] == 0).mean() - 0.5) < 0.02


def test_order_is_deterministic_and_seeded():
    parents, count = _parents(5000)
    a, b = st.order_copies(parents, count, SEED), st.order_copies(parents, count, SEED)
    assert a.equals(b)
    c = st.order_copies(parents, count, SEED + 1)
    assert not a["parent"].equals(c["parent"])


def test_shared_parents_keep_their_relative_order_in_the_twin():
    parents, count = _parents(4000)
    main = st.order_copies(parents, count, SEED)
    # twin: drop a third of the parents (German), add new ones, give some a second copy
    keep = np.arange(len(parents)) % 3 != 0
    extra = np.array([f"en{i}" for i in range(1500)], dtype=object)
    tw_par = np.concatenate([parents[keep], extra])
    tw_cnt = np.concatenate([np.where(np.arange(keep.sum()) % 7 == 0, 2, count[keep]), np.ones(1500, np.int64)])
    twin = st.order_copies(tw_par, tw_cnt, SEED)
    m = pd.DataFrame({"pid": parents[main["parent"]], "copy": main["copy"], "half": main["half"]})
    t = pd.DataFrame({"pid": tw_par[twin["parent"]], "copy": twin["copy"], "half": twin["half"]})
    m0 = m[(m["copy"] == 0) & m["pid"].isin(set(parents[keep]))]
    t0 = t[(t["copy"] == 0) & t["pid"].isin(set(parents[keep]))]
    assert list(m0["pid"]) == list(t0["pid"])


def test_chunks_stay_adjacent_and_in_order():
    rows = pd.DataFrame({"article_id": ["b_c000", "b_c001", "b_c002", "a", "c_c000", "c_c010", "c_c002"]})
    rows["parent_id"] = rows["article_id"].str.replace(r"_c\d+$", "", regex=True)
    rows["chunk"] = st.chunk_number(rows["article_id"])
    code, parents = pd.factorize(rows["parent_id"])
    rows["parent_code"] = code
    rows = rows.sort_values(["parent_code", "chunk"], kind="stable").reset_index(drop=True)
    copies = st.order_copies(np.asarray(parents, dtype=object), np.array([2, 1, 1]), SEED)
    ex = st.expand(rows[["parent_code"]], copies)
    seq = rows["article_id"].to_numpy()[ex["row"].to_numpy()].tolist()
    assert len(seq) == 10 and seq.count("b_c001") == 2
    for p, chunks in (("b", ["b_c000", "b_c001", "b_c002"]), ("c", ["c_c000", "c_c002", "c_c010"])):
        i = seq.index(chunks[0])
        assert seq[i:i + len(chunks)] == chunks


def test_write_shards_packs_documents_with_eos_and_stream_reads_rows(tmp_path):
    troot = tmp_path / "tok"
    (troot / "en").mkdir(parents=True)
    rng = np.random.default_rng(1)
    docs = [rng.integers(1, 1000, rng.integers(1, 40)).astype("<u2") for _ in range(60)]
    files, rows = [], []
    for f in range(3):                                   # three tokenized files, 20 documents each
        part = docs[f * 20:(f + 1) * 20]
        np.concatenate(part).tofile(troot / "en" / f"f{f}.bin")
        files.append(f"en/f{f}.bin")
        off = 0
        for d in part:
            rows.append({"file": f, "offset": off, "n_tokens": len(d)})
            off += len(d)
    order = pd.DataFrame(rows).iloc[rng.permutation(60)].reset_index(drop=True)
    scfg = {"rows_per_shard": 4, "context": 16, "eos_id": 0}
    out = tmp_path / "shards"
    recs = st.write_shards(order, files, troot, out, scfg)
    want = []
    for k in range(60):
        r = order.iloc[k]
        f = int(r["file"])
        local = [x for x in rows if x["file"] == f]
        j = [x["offset"] for x in local].index(int(r["offset"]))
        want.append(np.concatenate([docs[f * 20 + j], np.zeros(1, "<u2")]))
    want = np.concatenate(want)
    got = np.concatenate([np.fromfile(out / rec["file"], dtype="<u2") for rec in recs])
    assert np.array_equal(got, want)
    assert all(rec["tokens"] == 64 for rec in recs[:-1])
    man = {"config": scfg, "rows": (len(want) - 1) // 16, "tokens": len(want), "shards": recs}
    (out / "MANIFEST.json").write_text(json.dumps(man), encoding="utf-8")
    s = Stream(out)
    x, y = s.batch(3, 5)                                 # rows 3..7 cross shard boundaries
    assert np.array_equal(x.numpy().ravel(), want[48:128]) and np.array_equal(y.numpy().ravel(), want[49:129])
