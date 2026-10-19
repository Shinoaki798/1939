"""Training stream: the mixture's documents in their final order, packed into rows and written once
(TASKS Phase 4 prerequisite; decisions 2026-10-19, HANDOFF §12; settings in config/stream.yaml).

Order. The unit is the parent document: the chunks of a book or a long opinion stay adjacent, in chunk
order. A seeded 64-bit hash of the parent id decides everything:
  - the stream has two halves; a parent seen twice puts one copy in each half, a parent seen once goes to
    the half its hash picks, so both halves have the same make-up and two copies are never close;
  - within a half, copies are sorted by a key hashed from (parent, copy).
The twin uses the same seed, so the documents it shares with the main run keep their relative order.
Every document is followed by <|endoftext|>; the stream is cut into rows of `context` tokens with no
padding and no document mask. Row r is the input stream[r*C : r*C+C], its targets stream[r*C+1 : r*C+C+1].

Outputs
  data/shards/<stream>/<sha12>.bin     uint16 little-endian, rows_per_shard x context tokens (last one shorter)
  data/shards/<stream>/order.parquet   the documents in stream order: article_id, copy, half, start, n_tokens
  data/shards/<stream>/MANIFEST.json   shards in order with sha256, totals per half x lang x period x
                                       source, the gap between the two copies of a parent, input manifests
  data/shards/val/<lang>.bin, <lang>.idx.parquet   fixed Val slices (config `val`), documents back to back,
                                       each followed by <|endoftext|>, the slice starting with one

    python -m src.data.stream [--stream main,twin] [--val] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from src.data.download import load_config, repo_path
from src.data.mixture import uniform

COPY_SALT = np.uint64(0xD1B54A32D192ED03)


def splitmix(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finaliser on a uint64 array (wrapping arithmetic)."""
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def order_copies(parents: np.ndarray, count: np.ndarray, seed: int) -> pd.DataFrame:
    """Stream order of the copies of each parent: columns parent (index into `parents`), copy, half."""
    h, _ = uniform(parents, seed)
    idx = np.repeat(np.arange(len(parents)), count)
    copy = (np.arange(len(idx)) - np.repeat(np.cumsum(count) - count, count)).astype(np.int8)
    half = ((h[idx] & np.uint64(1)).astype(np.int8) ^ copy).astype(np.int8)
    key = splitmix(h[idx] ^ (copy.astype(np.uint64) * COPY_SALT))
    o = np.lexsort((key, half))
    return pd.DataFrame({"parent": idx[o], "copy": copy[o], "half": half[o]})


def chunk_number(article_ids: pd.Series) -> np.ndarray:
    """Chunk index from an `_c<digits>` suffix (0 when there is none)."""
    n = article_ids.str.extract(r"_c(\d+)$", expand=False)
    return n.fillna("0").astype(np.int64).to_numpy()


def expand(rows: pd.DataFrame, copies: pd.DataFrame) -> pd.DataFrame:
    """Stream order of document rows. rows: one row per document with column parent_code (index into the
    parents given to order_copies), sorted by parent_code then chunk number. Returns the row positions
    (column `row`) with copy and half, in stream order."""
    code = rows["parent_code"].to_numpy()
    starts = np.searchsorted(code, np.arange(code.max() + 1), side="left")
    ends = np.searchsorted(code, np.arange(code.max() + 1), side="right")
    p = copies["parent"].to_numpy()
    lens = (ends - starts)[p]
    first = np.repeat(starts[p], lens)
    within = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    return pd.DataFrame({"row": first + within, "copy": np.repeat(copies["copy"].to_numpy(), lens),
                         "half": np.repeat(copies["half"].to_numpy(), lens)})


def read_index(troot: Path, tman: dict, langs: list[str], split: str, workers: int = 16) -> pd.DataFrame:
    """Tokenized documents of one split: article id hash, file id, offset, n_tokens, n_bytes, source, lang."""
    items = [(k, o) for k, o in sorted(tman["outputs"].items()) if k.split("/")[0] in langs]

    def one(i: int) -> pd.DataFrame | None:
        k, o = items[i]
        t = pq.read_table(troot / o["index"])
        t = t.filter(pc.equal(t.column("split"), split))
        if t.num_rows == 0:
            return None
        d = pd.DataFrame({"aid": t.column("article_id").to_pandas(), "offset": t.column("offset").to_numpy(),
                          "n_tokens": t.column("n_tokens").to_numpy(), "n_bytes": t.column("n_bytes").to_numpy()})
        d["file"] = np.int32(i)
        return d
    with ThreadPoolExecutor(workers) as ex:
        parts = [d for d in ex.map(one, range(len(items))) if d is not None]
    d = pd.concat(parts, ignore_index=True)
    d.attrs["files"] = [o["bin"] for _, o in items]
    d.attrs["sources"] = [k.split("/")[1] for k, _ in items]
    d.attrs["langs"] = [k.split("/")[0] for k, _ in items]
    return d


def id_hash(ids) -> np.ndarray:
    return pd.util.hash_array(np.asarray(ids, dtype=object), hash_key="1939stream000000")


def locate(mix: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    """Join mixture rows to their tokenized location by article id; every row must be found once."""
    left = pd.DataFrame({"h": id_hash(mix["article_id"])})
    right = pd.DataFrame({"h": id_hash(idx["aid"]), "file": idx["file"].to_numpy(), "offset": idx["offset"].to_numpy(),
                          "n_tokens": idx["n_tokens"].to_numpy(), "n_bytes": idx["n_bytes"].to_numpy()})
    if right["h"].duplicated().any():
        raise SystemExit("article id hash collision or duplicate article id in the tokenized index")
    j = left.merge(right, on="h", how="left", validate="many_to_one")
    if j["file"].isna().any():
        raise SystemExit(f"{int(j['file'].isna().sum()):,} mixture rows not found among tokenized train documents")
    return j.drop(columns="h").astype({"file": np.int32, "offset": np.int64, "n_tokens": np.int64,
                                       "n_bytes": np.int64})


def write_shards(order: pd.DataFrame, files: list[str], troot: Path, out: Path, scfg: dict,
                 dry_run: bool = False) -> list[dict]:
    """Pack the ordered documents (columns file, offset, n_tokens) with <|endoftext|> after each and write
    shards of rows_per_shard x context tokens. Returns one record per shard."""
    shard_tokens = int(scfg["rows_per_shard"]) * int(scfg["context"])
    n = order["n_tokens"].to_numpy()
    start = np.concatenate([[0], np.cumsum(n + 1)[:-1]])
    total = int(n.sum() + len(n))
    order["start"] = start
    recs = []
    out.mkdir(parents=True, exist_ok=True)
    maps: dict[int, np.ndarray] = {}
    for s in range((total + shard_tokens - 1) // shard_tokens):
        a, b = s * shard_tokens, min((s + 1) * shard_tokens, total)
        if dry_run and s > 0:
            recs.append({"shard": s, "tokens": b - a})
            continue
        buf = np.full(b - a, int(scfg["eos_id"]), dtype="<u2")
        lo = int(np.searchsorted(start + n, a, side="right"))
        hi = int(np.searchsorted(start, b, side="left"))
        part = order.iloc[lo:hi].sort_values(["file", "offset"], kind="stable")
        for f, o, k, st in zip(part["file"].to_numpy(), part["offset"].to_numpy(), part["n_tokens"].to_numpy(),
                               part["start"].to_numpy()):
            if f not in maps:
                if len(maps) > 256:                      # each map holds a file descriptor
                    maps.clear()
                maps[f] = np.memmap(troot / files[f], dtype="<u2", mode="r")
            s0, s1 = max(st, a), min(st + k, b)            # the document's tokens that fall in this shard
            buf[s0 - a:s1 - a] = maps[f][o + (s0 - st):o + (s1 - st)]
        digest = hashlib.sha256(buf.tobytes()).hexdigest()
        rec = {"shard": s, "file": f"{digest[:12]}.bin", "sha256": digest, "tokens": b - a, "first_token": a}
        if not dry_run:
            tmp = out / f".tmp_{s:04d}.bin"
            buf.tofile(tmp)
            tmp.replace(out / rec["file"])
        recs.append(rec)
        print(f"  shard {s}: {b - a:,} tokens, {hi - lo:,} documents -> {rec['file']}", flush=True)
    return recs


def summarize(order: pd.DataFrame) -> dict:
    """Tokens per half x lang x period x source, and the distance between the two copies of a parent."""
    t = order.assign(tok=order["n_tokens"] + 1)
    by = t.groupby(["half", "lang", "period", "source"], observed=True)["tok"].sum()
    cells = [{"half": int(h), "lang": l, "period": p, "source": s, "tokens": int(v)} for (h, l, p, s), v in by.items()]
    two = t[t.groupby("parent_id")["copy"].transform("max") == 1]
    gap = (two.groupby("parent_id")["start"].max() - two.groupby("parent_id")["start"].min()).to_numpy()
    q = np.percentile(gap, [0, 1, 50]) if len(gap) else [0, 0, 0]
    return {"cells": cells, "tokens_by_half": {int(k): int(v) for k, v in t.groupby("half")["tok"].sum().items()},
            "copies_gap_tokens": {"parents_seen_twice": int(len(gap)), "min": int(q[0]), "p1": int(q[1]),
                                  "median": int(q[2])}}


def build_stream(name: str, cfg: dict, scfg: dict, dry_run: bool) -> dict:
    mroot, troot = repo_path(cfg["mixture"]), repo_path(cfg["tokenized"])
    spec = scfg["streams"][name]
    mman_path = mroot / spec["manifest"]
    mman = json.loads(mman_path.read_text(encoding="utf-8"))
    tman_path = troot / "MANIFEST.json"
    tman = json.loads(tman_path.read_text(encoding="utf-8"))
    cols = ["article_id", "parent_id", "source", "lang", "period", "year", "count"]
    parts = []
    for lang in spec["langs"]:
        f = mroot / mman["outputs"][lang]["file"]
        if not f.exists():
            f = mman_path.parent / name / mman["outputs"][lang]["file"]
        parts.append(pq.read_table(f, columns=cols).to_pandas())
    mix = pd.concat(parts, ignore_index=True)
    mix = mix[mix["count"] > 0].reset_index(drop=True)
    print(f"{name}: {len(mix):,} documents, {int(mix['count'].sum()):,} copies", flush=True)
    idx = read_index(troot, tman, spec["langs"], "train")
    loc = locate(mix, idx)
    mix = pd.concat([mix, loc], axis=1)
    mix["chunk"] = chunk_number(mix["article_id"])
    code, parents = pd.factorize(mix["parent_id"])
    parents = np.asarray(parents, dtype=object)
    mix["parent_code"] = code
    mix = mix.sort_values(["parent_code", "chunk"], kind="stable").reset_index(drop=True)
    pcount = mix.groupby("parent_code")["count"].max().to_numpy()
    if (mix.groupby("parent_code")["count"].nunique() > 1).any():
        raise SystemExit("chunks of one parent have different seen counts")
    copies = order_copies(parents, pcount.astype(np.int64), int(scfg["seed"]))
    ex = expand(mix[["parent_code"]], copies)
    order = mix.iloc[ex["row"].to_numpy()].reset_index(drop=True)
    order["copy"], order["half"] = ex["copy"].to_numpy(), ex["half"].to_numpy()
    out = repo_path(cfg["shards"]) / name
    recs = write_shards(order, idx.attrs["files"], troot, out, scfg, dry_run)
    summary = summarize(order)
    manifest = {"stage": "shards", "stream": name,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "config": scfg | {"val": None}, "dtype": "uint16 little-endian",
                "inputs": {"mixture_manifest": spec["manifest"],
                           "mixture_manifest_sha256": hashlib.sha256(mman_path.read_bytes()).hexdigest(),
                           "tokenized_manifest_sha256": hashlib.sha256(tman_path.read_bytes()).hexdigest(),
                           "tokenizer_sha256": tman["tokenizer"]["sha256"]},
                "documents": int(len(order)), "tokens": int(sum(r["tokens"] for r in recs)),
                "rows": int((sum(r["tokens"] for r in recs) - 1) // int(scfg["context"])),
                "shards": recs, "first_documents": order["article_id"].head(5).tolist()} | summary
    print(json.dumps({k: manifest[k] for k in ("documents", "tokens", "rows", "tokens_by_half", "copies_gap_tokens")}),
          flush=True)
    if not dry_run:
        pq.write_table(pa.Table.from_pandas(order[["article_id", "copy", "half", "start", "n_tokens"]],
                                            preserve_index=False), out / "order.parquet", compression="zstd")
        (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def build_val(cfg: dict, scfg: dict, dry_run: bool) -> dict:
    """Fixed Val slices: documents of the Val split from the configured sources, in seeded hash order,
    until the token target; each slice starts with <|endoftext|> and has one after every document."""
    troot = repo_path(cfg["tokenized"])
    tman = json.loads((troot / "MANIFEST.json").read_text(encoding="utf-8"))
    out = repo_path(cfg["shards"]) / "val"
    res = {}
    for lang, spec in scfg["val"].items():
        idx = read_index(troot, tman, [lang], "val")
        src = np.asarray(idx.attrs["sources"])[idx["file"].to_numpy()]
        d = idx[np.isin(src, spec["sources"])].copy()
        h, _ = uniform(d["aid"].to_numpy(dtype=object), int(scfg["seed"]))
        d = d.iloc[np.argsort(h, kind="stable")]
        keep = int(np.searchsorted(np.cumsum(d["n_tokens"].to_numpy() + 1), float(spec["tokens"]))) + 1
        d = d.iloc[:keep].reset_index(drop=True)
        n = d["n_tokens"].to_numpy()
        d["offset_in_slice"] = np.concatenate([[1], 1 + np.cumsum(n + 1)[:-1]])
        buf = np.full(int(n.sum() + len(n) + 1), int(scfg["eos_id"]), dtype="<u2")
        for f, o, k, s in zip(d["file"], d["offset"], d["n_tokens"], d["offset_in_slice"]):
            buf[s:s + k] = np.memmap(troot / idx.attrs["files"][f], dtype="<u2", mode="r")[o:o + k]
        digest = hashlib.sha256(buf.tobytes()).hexdigest()
        res[lang] = {"file": f"{lang}.bin", "sha256": digest, "documents": int(len(d)), "tokens": int(len(buf)),
                     "bytes": int(d["n_bytes"].sum()), "sources": spec["sources"]}
        print(f"val {lang}: {res[lang]}", flush=True)
        if not dry_run:
            out.mkdir(parents=True, exist_ok=True)
            buf.tofile(out / f"{lang}.bin")
            pq.write_table(pa.table({"article_id": d["aid"].to_numpy(dtype=object), "offset": d["offset_in_slice"].to_numpy(),
                                     "n_tokens": n.astype(np.int32), "n_bytes": d["n_bytes"].to_numpy().astype(np.int32)}),
                           out / f"{lang}.idx.parquet", compression="zstd")
    if not dry_run:
        (out / "MANIFEST.json").write_text(json.dumps({"stage": "val", "created_at": dt.datetime.now(
            dt.timezone.utc).isoformat(timespec="seconds"), "seed": scfg["seed"], "slices": res}, indent=1),
                                            encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stream", default="main,twin")
    ap.add_argument("--val", action="store_true", help="also write the fixed Val slices")
    ap.add_argument("--dry-run", action="store_true", help="order and count; build the first shard in memory only")
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    scfg = yaml.safe_load(repo_path("config/stream.yaml").read_text(encoding="utf-8"))
    for name in [s for s in args.stream.split(",") if s]:
        build_stream(name, cfg, scfg, args.dry_run)
    if args.val:
        build_val(cfg, scfg, args.dry_run)


if __name__ == "__main__":
    main()
