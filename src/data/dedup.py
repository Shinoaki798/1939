"""Exact and near-duplicate removal, per language pool (HANDOFF §5.2; §12 2026-10-07).

Input: every selected parquet of one language, data/selected/<source>/<lang>/*.parquet. Output under
data/dedup/<lang>/: drops.parquet (one row per removed document: article_id, source, date, reason,
kept_id, jaccard) and MANIFEST.json (parameters, counts per source). A document that is not in
drops.parquet survives. MinHash signatures are cached per selected file (sig/<sha12>.npy and
sig/<sha12>.idx.parquet), so a rerun after new data arrives signs only the new files; the matching
step always runs over the whole language pool.

  comparison form  lower case, letters-only tokens (the selected text is already NFC and, for German,
                   typography-normalised). Documents with fewer than SHINGLE tokens match exactly only.
  exact            identical comparison form (blake2b, 64 bit).
  near             MinHash of word SHINGLE-grams with NUM_PERM multiply-shift hashes; LSH with BANDS x
                   ROWS. Inside an LSH bucket every document is compared with the bucket's keeper on the
                   full signature; an estimated Jaccard (share of equal values) >= THRESHOLD links them.
                   Linked documents form clusters (connected components).
  keeper           per cluster the earliest date; ties: article-level before page-level source, then a
                   fixed hash of the id. Everything else in the cluster is dropped.

At THRESHOLD 0.8 with 16 x 8 bands a pair with true Jaccard 0.8 becomes a candidate with probability
~0.95, one at 0.85 ~0.99, one at 0.6 ~0.24 (then rejected on the full signature).

    python -m src.data.dedup --lang en [--workers 8] [--dry-run] [--samples reports/dedup_samples_en.md]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.download import load_config, repo_path

SHINGLE = 5
NUM_PERM = 128
BANDS, ROWS = 16, 8
THRESHOLD = 0.8
SEED = 1939
TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)

_rng = np.random.default_rng(SEED)
PERM_A = _rng.integers(1, 2 ** 63, NUM_PERM, dtype=np.uint64) | np.uint64(1)    # odd multipliers
PERM_B = _rng.integers(0, 2 ** 63, NUM_PERM, dtype=np.uint64)
SHINGLE_MUL = _rng.integers(1, 2 ** 63, SHINGLE, dtype=np.uint64) | np.uint64(1)
BAND_MUL = _rng.integers(1, 2 ** 63, ROWS, dtype=np.uint64) | np.uint64(1)
assert BANDS * ROWS == NUM_PERM

_TOKH: dict[str, int] = {}


def token_hash(t: str) -> int:
    h = _TOKH.get(t)
    if h is None:
        b = t.encode("utf-8")
        h = zlib.crc32(b) | (zlib.crc32(b, 0x9E3779B9) << 32)
        if len(_TOKH) < 2_000_000:
            _TOKH[t] = h
    return h


def comparison_tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def exact_hash(tokens: list[str]) -> int:
    d = hashlib.blake2b(" ".join(tokens).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "big", signed=True)


def minhash(tokens: list[str]) -> np.ndarray | None:
    """uint32[NUM_PERM] signature of the set of word SHINGLE-grams; None if the text is too short."""
    n = len(tokens) - SHINGLE + 1
    if n < 1:
        return None
    th = np.fromiter((token_hash(t) for t in tokens), dtype=np.uint64, count=len(tokens))
    sh = np.zeros(n, dtype=np.uint64)
    for j in range(SHINGLE):
        sh += th[j:j + n] * SHINGLE_MUL[j]
    sig = np.full(NUM_PERM, np.iinfo(np.uint64).max, dtype=np.uint64)
    for s in range(0, n, 4096):
        block = sh[s:s + 4096, None] * PERM_A[None, :] + PERM_B[None, :]
        np.minimum(sig, block.min(axis=0), out=sig)
    return (sig >> np.uint64(32)).astype(np.uint32)


def jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> np.ndarray:
    return (sig_a == sig_b).mean(axis=-1)


# ---------------------------------------------------------------- signing (per selected file)

IDX_SCHEMA = pa.schema([("article_id", pa.string()), ("source", pa.string()), ("date", pa.string()),
                        ("page_level", pa.bool_()), ("n_words", pa.int32()), ("exact", pa.int64()),
                        ("has_sig", pa.bool_())])


def _sign_job(job: tuple) -> dict:
    path, sig_path, idx_path = job
    t0 = time.time()
    pf = pq.ParquetFile(path)
    rows: dict[str, list] = defaultdict(list)
    sigs = []
    words = 0
    for batch in pf.iter_batches(batch_size=2000, columns=["article_id", "source", "date", "page_level", "n_words", "text"]):
        b = batch.to_pydict()
        for i in range(batch.num_rows):
            toks = comparison_tokens(b["text"][i] or "")
            words += len(toks)
            sig = minhash(toks)
            for k in ("article_id", "source", "date", "page_level", "n_words"):
                rows[k].append(b[k][i])
            rows["exact"].append(exact_hash(toks))
            rows["has_sig"].append(sig is not None)
            sigs.append(sig if sig is not None else np.zeros(NUM_PERM, dtype=np.uint32))
    arr = np.stack(sigs) if sigs else np.zeros((0, NUM_PERM), dtype=np.uint32)
    np.save(sig_path, arr)
    pq.write_table(pa.Table.from_pydict(rows, schema=IDX_SCHEMA), idx_path, compression="zstd")
    return {"path": path, "docs": len(sigs), "words": words, "seconds": round(time.time() - t0, 1)}


def selected_files(cfg: dict, lang: str) -> list[tuple[str, str]]:
    """(path, sha256) of every selected parquet of this language, from the per-source MANIFESTs."""
    out = []
    for m in sorted(repo_path(cfg["selected"]).glob("*/MANIFEST.json")):
        man = json.loads(m.read_text(encoding="utf-8"))
        for e in man["files"].values():
            o = e["outputs"].get(lang)
            if o:
                out.append((str(m.parent / o["file"]), o["sha256"]))
    return sorted(set(out))


# ---------------------------------------------------------------- matching

def priority_rank(idx: pa.Table) -> np.ndarray:
    """rank[i] = position of document i in keeper order (earliest date, article-level, id hash)."""
    date = np.array([int(d.replace("-", "")[:8] or 0) for d in idx.column("date").to_pylist()], dtype=np.int64)
    page = idx.column("page_level").to_numpy(zero_copy_only=False).astype(np.int8)
    idh = np.array([zlib.crc32(a.encode("utf-8")) for a in idx.column("article_id").to_pylist()], dtype=np.int64)
    order = np.lexsort((idh, page, date))
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank


def group_pairs(keys: np.ndarray, rank: np.ndarray, valid: np.ndarray | None = None):
    """For equal keys: (member, keeper) pairs, keeper = best-ranked member of the group."""
    idx = np.flatnonzero(valid) if valid is not None else np.arange(len(keys))
    k = keys[idx]
    order = idx[np.argsort(k, kind="stable")]
    ks = keys[order]
    start = np.r_[True, ks[1:] != ks[:-1]]
    gid = np.cumsum(start) - 1
    size = np.bincount(gid)
    multi = size[gid] > 1
    order, gid = order[multi], gid[multi]
    if len(order) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    gid = np.unique(gid, return_inverse=True)[1]
    best = np.full(gid.max() + 1, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(best, gid, rank[order])
    doc_of_rank = np.empty_like(rank)
    doc_of_rank[rank] = np.arange(len(rank))
    keeper = doc_of_rank[best[gid]]
    m = order != keeper
    return order[m], keeper[m]


def components(n: int, u: np.ndarray, v: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """Label of every document = best rank in its connected component (min-label propagation)."""
    lab = rank.copy()
    doc_of_rank = np.empty_like(rank)
    doc_of_rank[rank] = np.arange(n)
    for _ in range(200):
        m = np.minimum(lab[u], lab[v])
        before = lab.copy()
        np.minimum.at(lab, u, m)
        np.minimum.at(lab, v, m)
        lab = np.minimum(lab, lab[doc_of_rank[lab]])          # pointer jumping
        if np.array_equal(lab, before):
            return lab
    raise RuntimeError("label propagation did not converge")


def match(sig: np.ndarray, idx: pa.Table, sample_k: int = 20,
          bins: tuple = ((0.80, 0.85), (0.70, 0.80))):
    n = len(idx)
    rank = priority_rank(idx)
    has_sig = idx.column("has_sig").to_numpy(zero_copy_only=False)
    exact = idx.column("exact").to_numpy()
    eu, ev = group_pairs(exact, rank)
    us, vs, js = [eu], [ev], [np.ones(len(eu))]
    random.seed(SEED)
    samples = {f"{lo:.2f}-{hi:.2f}": [] for lo, hi in bins}
    seen_bins = Counter()
    for band in range(BANDS):
        cols = sig[:, band * ROWS:(band + 1) * ROWS].astype(np.uint64)
        key = np.zeros(n, dtype=np.uint64)
        for j in range(ROWS):
            key += cols[:, j] * BAND_MUL[j]
        u, v = group_pairs(key, rank, valid=has_sig)
        if len(u) == 0:
            continue
        j_est = np.concatenate([jaccard(sig[u[s:s + 500_000]], sig[v[s:s + 500_000]])
                                for s in range(0, len(u), 500_000)])
        for lo, hi in bins:
            name = f"{lo:.2f}-{hi:.2f}"
            sel = np.flatnonzero((j_est >= lo) & (j_est < hi))
            for t in sel[:5000]:                      # reservoir over all bands
                seen_bins[name] += 1
                if len(samples[name]) < sample_k:
                    samples[name].append((int(u[t]), int(v[t]), float(j_est[t])))
                elif random.random() < sample_k / seen_bins[name]:
                    samples[name][random.randrange(sample_k)] = (int(u[t]), int(v[t]), float(j_est[t]))
        ok = j_est >= THRESHOLD
        us.append(u[ok]); vs.append(v[ok]); js.append(j_est[ok])
    u, v = np.concatenate(us), np.concatenate(vs)
    lab = components(n, u, v, rank) if len(u) else rank.copy()
    doc_of_rank = np.empty_like(rank)
    doc_of_rank[rank] = np.arange(n)
    keeper = doc_of_rank[lab]
    dropped = np.flatnonzero(keeper != np.arange(n))
    jd = np.where(exact[dropped] == exact[keeper[dropped]], 1.0,
                  jaccard(sig[dropped], sig[keeper[dropped]]) if len(dropped) else 0.0)
    reason = np.where(exact[dropped] == exact[keeper[dropped]], "exact", "near")
    return dropped, keeper[dropped], jd, reason, samples


# ---------------------------------------------------------------- driver

def write_samples(path: Path, samples: dict, idx: pa.Table, files: list[str]) -> None:
    ids = {i for pairs in samples.values() for a, b, _ in pairs for i in (a, b)}
    aid = idx.column("article_id").to_pylist()
    want = {aid[i] for i in ids}
    text: dict[str, tuple] = {}
    for f in files:
        t = pq.read_table(f, columns=["article_id", "source", "date", "text"],
                          filters=[("article_id", "in", list(want))])
        for a, s, d, x in zip(*(t.column(c).to_pylist() for c in ("article_id", "source", "date", "text"))):
            text[a] = (s, d, x)
    out = ["# Dedup samples near the threshold", "",
           f"THRESHOLD {THRESHOLD}; MinHash {NUM_PERM} perms, {BANDS}x{ROWS} bands, word {SHINGLE}-grams.", ""]
    for name, pairs in samples.items():
        linked = float(name.split("-")[0]) >= THRESHOLD
        out += [f"## Estimated Jaccard {name} ({'linked: duplicates' if linked else 'not linked'})", ""]
        for k, (a, b, j) in enumerate(pairs, 1):
            for tag, i in (("A", a), ("B", b)):
                s, d, x = text.get(aid[i], ("?", "?", ""))
                snippet = " ".join(x.split())[:400].replace("|", "/")
                out.append(f"{k}{tag}. `{aid[i]}` {s} {d} J={j:.2f}: {snippet}")
            out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["en", "de"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="match and count; write no drops/MANIFEST")
    ap.add_argument("--samples", default="", help="write near-threshold example pairs to this markdown file")
    ap.add_argument("--sample-bins", default="0.80-0.85,0.70-0.80", help="Jaccard ranges to sample, lo-hi,...")
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    out_dir = repo_path(cfg["dedup"]) / args.lang
    sig_dir = out_dir / "sig"
    sig_dir.mkdir(parents=True, exist_ok=True)
    files = selected_files(cfg, args.lang)
    if not files:
        sys.exit(f"no selected files for {args.lang}")
    t0 = time.time()
    jobs = [(p, str(sig_dir / f"{sha[:12]}.npy"), str(sig_dir / f"{sha[:12]}.idx.parquet")) for p, sha in files
            if not (sig_dir / f"{sha[:12]}.idx.parquet").exists()]
    signed_words = 0
    if jobs:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
            for k, fut in enumerate(as_completed([ex.submit(_sign_job, j) for j in jobs]), 1):
                r = fut.result()
                signed_words += r["words"]
                if k % 200 == 0 or k == len(jobs):
                    print(f"signed {k}/{len(jobs)} files, {signed_words:,} tokens, {time.time() - t0:.0f}s", flush=True)
    t_sign = time.time() - t0
    shas = [sha[:12] for _, sha in files]
    idx = pa.concat_tables([pq.read_table(sig_dir / f"{s}.idx.parquet") for s in shas])
    sig = np.concatenate([np.load(sig_dir / f"{s}.npy") for s in shas]) if shas else np.zeros((0, NUM_PERM), np.uint32)
    t1 = time.time()
    bins = tuple(tuple(float(x) for x in b.split("-")) for b in args.sample_bins.split(","))
    dropped, kept, jd, reason, samples = match(sig, idx, bins=bins)
    t_match = time.time() - t1
    aid, src = idx.column("article_id").to_pylist(), idx.column("source").to_pylist()
    dates, nw = idx.column("date").to_pylist(), idx.column("n_words").to_pylist()
    per = defaultdict(Counter)
    for i in range(len(idx)):
        per[src[i]]["docs"] += 1
        per[src[i]]["words"] += nw[i]
    for i, r in zip(dropped.tolist(), reason.tolist()):
        per[src[i]][f"dropped_{r}"] += 1
        per[src[i]]["dropped_words"] += nw[i]
    print(f"{args.lang}: {len(idx):,} docs, dropped {len(dropped):,} "
          f"(exact {int((reason == 'exact').sum()):,}, near {int((reason == 'near').sum()):,}); "
          f"sign {t_sign:.0f}s ({signed_words / max(t_sign, 1e-9) / 1e6:.1f}M tokens/s), match {t_match:.0f}s", flush=True)
    for s, c in sorted(per.items()):
        print(f"  {s:24} docs={c['docs']:>9,} dropped={c['dropped_exact'] + c['dropped_near']:>8,} "
              f"({100 * (c['dropped_exact'] + c['dropped_near']) / c['docs']:5.1f} %) "
              f"words dropped {100 * c['dropped_words'] / max(c['words'], 1):5.1f} %", flush=True)
    if args.samples:
        write_samples(Path(args.samples), samples, idx, [p for p, _ in files])
        print(f"samples -> {args.samples}", flush=True)
    if args.dry_run:
        return
    drops = pa.table({"article_id": [aid[i] for i in dropped.tolist()], "source": [src[i] for i in dropped.tolist()],
                      "date": [dates[i] for i in dropped.tolist()], "reason": reason.tolist(),
                      "kept_id": [aid[i] for i in kept.tolist()], "jaccard": jd.astype(np.float32)})
    pq.write_table(drops, out_dir / "drops.parquet", compression="zstd")
    manifest = {"stage": "dedup", "lang": args.lang,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "params": {"SHINGLE": SHINGLE, "NUM_PERM": NUM_PERM, "BANDS": BANDS, "ROWS": ROWS,
                           "THRESHOLD": THRESHOLD, "SEED": SEED, "keeper": "earliest date, article-level, id hash"},
                "inputs": {s: sha for (_, sha), s in zip(files, shas)},
                "docs": len(idx), "dropped": len(dropped),
                "per_source": {s: dict(c) for s, c in sorted(per.items())},
                "seconds": {"sign": round(t_sign), "match": round(t_match)}}
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
