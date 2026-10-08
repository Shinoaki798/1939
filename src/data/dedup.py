"""Exact and near-duplicate removal of whole documents, per language pool (HANDOFF §5.2; §12 2026-10-07).

Input: every selected parquet of one language, data/selected/<source>/<lang>/*.parquet. Output under
data/dedup/<lang>/: drops.parquet (one row per removed document: article_id, source, date, reason,
kept_id, jaccard) and MANIFEST.json (parameters, per-source thresholds, removal per source at the
per-source thresholds and at a uniform 0.80). A document that is not in drops.parquet survives.
MinHash signatures are cached per selected file (sig/<sha12>.npy and sig/<sha12>.idx.parquet), so a
rerun after new data arrives signs only the new files; the matching always covers the whole pool.

  comparison form  lower case, letters-only tokens (the selected text is already NFC and, for German,
                   typography-normalised). Documents with fewer than SHINGLE tokens match exactly only.
  exact            identical comparison form (blake2b, 64 bit).
  near             MinHash of word SHINGLE-grams with NUM_PERM multiply-shift hashes; LSH with BANDS x
                   ROWS. Inside an LSH bucket every document is compared with the bucket's keeper on the
                   full signature; an estimated Jaccard (share of equal values) at or above the pair's
                   threshold links them. Linked documents form clusters (connected components).
  thresholds       by source type (user, 2026-10-07): OCR sources 0.60, keyed sources (`ocr_or_keyed:
                   keyed` in sources.yaml) 0.80; a pair with an OCR side uses 0.60. One OCR error spoils
                   SHINGLE shingles, so true duplicates of noisy pages fall well below 0.80.
  keeper           per cluster the earliest date; ties: article-level before page-level source, then a
                   fixed hash of the id. Everything else in the cluster is dropped.

With 32 x 4 bands a pair at true Jaccard 0.60 becomes a candidate with probability ~0.99, at 0.55 ~0.95,
at 0.30 ~0.23 (then rejected on the full signature). Paragraph-level dedup of page-level sources is a
separate stage (src.data.para_dedup).

    python -m src.data.dedup --lang en [--workers 8] [--dry-run] [--samples reports/dedup_samples_en.md]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
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
import pyarrow.parquet as pq

from src.data.download import load_config, repo_path

SHINGLE = 5
NUM_PERM = 128
BANDS, ROWS = 32, 4
THRESHOLD_OCR, THRESHOLD_KEYED = 0.60, 0.80
SAMPLE_BAND, SAMPLE_K = (0.55, 0.65), 30
SEED = 1939
TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)
IN_RAM_BYTES = 4 << 30          # band keys above this size go to a memory-mapped file

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


def shingles(tokens: list[str]) -> np.ndarray:
    """uint64 hash of every word SHINGLE-gram (empty if the text is too short)."""
    n = len(tokens) - SHINGLE + 1
    if n < 1:
        return np.zeros(0, dtype=np.uint64)
    th = np.fromiter((token_hash(t) for t in tokens), dtype=np.uint64, count=len(tokens))
    sh = np.zeros(n, dtype=np.uint64)
    for j in range(SHINGLE):
        sh += th[j:j + n] * SHINGLE_MUL[j]
    return sh


def minhash(tokens: list[str]) -> np.ndarray | None:
    """uint32[NUM_PERM] signature of the set of word SHINGLE-grams; None if the text is too short."""
    sh = shingles(tokens)
    if len(sh) == 0:
        return None
    sig = np.full(NUM_PERM, np.iinfo(np.uint64).max, dtype=np.uint64)
    for s in range(0, len(sh), 4096):
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


def selected_files(cfg: dict, lang: str) -> tuple[list[tuple[str, str]], dict[str, bool]]:
    """(path, sha256) of every selected parquet of this language, and keyed (True/False) per source."""
    out, keyed = [], {}
    for m in sorted(repo_path(cfg["selected"]).glob("*/MANIFEST.json")):
        man = json.loads(m.read_text(encoding="utf-8"))
        keyed[man["source"]] = bool(man.get("params", {}).get("keyed"))
        for e in man["files"].values():
            o = e["outputs"].get(lang)
            if o:
                out.append((str(m.parent / o["file"]), o["sha256"]))
    return sorted(set(out)), keyed


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
    if len(u) == 0:
        return lab
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


def band_keys(sig: np.ndarray, out_path: Path | None = None) -> np.ndarray:
    """keys[b, i] = hash of rows b*ROWS .. b*ROWS+ROWS-1 of document i's signature, one sequential pass."""
    n = len(sig)
    if out_path is not None and BANDS * n * 8 > IN_RAM_BYTES:
        keys = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint64, shape=(BANDS, n))
    else:
        keys = np.zeros((BANDS, n), dtype=np.uint64)
    for r0 in range(0, n, 1_000_000):
        block = np.asarray(sig[r0:r0 + 1_000_000]).astype(np.uint64)
        for b in range(BANDS):
            k = np.zeros(len(block), dtype=np.uint64)
            for j in range(ROWS):
                k += block[:, b * ROWS + j] * BAND_MUL[j]
            keys[b, r0:r0 + len(block)] = k
    return keys


def match(sig: np.ndarray, idx: pa.Table, thr: np.ndarray, keys: np.ndarray | None = None,
          sample_band: tuple = SAMPLE_BAND, sample_k: int = SAMPLE_K):
    """Clusters at the per-document thresholds `thr` (a pair uses the lower one) and, for the report, at a
    uniform THRESHOLD_KEYED. Returns dropped, their keepers, Jaccard to the keeper, reason, the drop mask
    at the uniform threshold, and per-source samples of OCR pairs in `sample_band`."""
    n = len(idx)
    rank = priority_rank(idx)
    has_sig = idx.column("has_sig").to_numpy(zero_copy_only=False)
    exact = idx.column("exact").to_numpy()
    src = idx.column("source").to_pylist()
    keys = band_keys(sig) if keys is None else keys
    eu, ev = group_pairs(exact, rank)
    pol_u, pol_v, uni_u, uni_v = [eu], [ev], [eu], [ev]
    random.seed(SEED)
    samples: dict[str, list] = defaultdict(list)
    seen: Counter = Counter()
    lo, hi = sample_band
    for band in range(BANDS):
        u, v = group_pairs(np.asarray(keys[band]), rank, valid=has_sig)
        if len(u) == 0:
            continue
        j_est = np.concatenate([jaccard(np.asarray(sig[u[s:s + 500_000]]), np.asarray(sig[v[s:s + 500_000]]))
                                for s in range(0, len(u), 500_000)])
        ok_pol = j_est >= np.minimum(thr[u], thr[v])
        ok_uni = j_est >= THRESHOLD_KEYED
        pol_u.append(u[ok_pol]); pol_v.append(v[ok_pol]); uni_u.append(u[ok_uni]); uni_v.append(v[ok_uni])
        for t in np.flatnonzero((j_est >= lo) & (j_est < hi) & (thr[u] < THRESHOLD_KEYED))[:20000]:
            s = src[u[t]]
            seen[s] += 1
            pair = (int(u[t]), int(v[t]), float(j_est[t]))
            if len(samples[s]) < sample_k:
                samples[s].append(pair)
            elif random.random() < sample_k / seen[s]:
                samples[s][random.randrange(sample_k)] = pair
    doc_of_rank = np.empty_like(rank)
    doc_of_rank[rank] = np.arange(n)
    keeper = doc_of_rank[components(n, np.concatenate(pol_u), np.concatenate(pol_v), rank)]
    keeper_uni = doc_of_rank[components(n, np.concatenate(uni_u), np.concatenate(uni_v), rank)]
    dropped = np.flatnonzero(keeper != np.arange(n))
    k = keeper[dropped]
    same = exact[dropped] == exact[k]
    jd = np.where(same, 1.0, jaccard(np.asarray(sig[dropped]), np.asarray(sig[k])) if len(dropped) else 0.0)
    reason = np.where(same, "exact", "near")
    return dropped, k, jd, reason, keeper_uni != np.arange(n), dict(samples)


# ---------------------------------------------------------------- driver

def write_samples(path: Path, samples: dict, idx: pa.Table, files: list[str]) -> None:
    ids = {i for pairs in samples.values() for a, b, _ in pairs for i in (a, b)}
    aid = idx.column("article_id").to_pylist()
    want = {aid[i] for i in ids}
    text: dict[str, tuple] = {}
    for f in files:
        t = pq.read_table(f, columns=["article_id", "source", "date", "text"], filters=[("article_id", "in", list(want))])
        for a, s, d, x in zip(*(t.column(c).to_pylist() for c in ("article_id", "source", "date", "text"))):
            text[a] = (s, d, x)
    out = ["# Dedup samples, OCR sources, estimated Jaccard 0.55-0.65", "",
           f"Thresholds: OCR {THRESHOLD_OCR}, keyed {THRESHOLD_KEYED}. MinHash {NUM_PERM} perms, {BANDS}x{ROWS} "
           f"bands, word {SHINGLE}-grams. Pairs at or above {THRESHOLD_OCR} are linked (A = the later document).", ""]
    for source, pairs in sorted(samples.items()):
        out += [f"## {source} ({len(pairs)} pairs)", ""]
        for k, (a, b, j) in enumerate(sorted(pairs, key=lambda p: p[2]), 1):
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
    ap.add_argument("--samples", default="", help="write 0.55-0.65 example pairs per OCR source to this file")
    ap.add_argument("--sign-only", action="store_true", help="sign new files and stop (before all data is in)")
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    out_dir = repo_path(cfg["dedup"]) / args.lang
    sig_dir = out_dir / "sig"
    sig_dir.mkdir(parents=True, exist_ok=True)
    files, keyed = selected_files(cfg, args.lang)
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
    if args.sign_only:
        print(f"signed {len(jobs)} new files in {t_sign:.0f}s; {len(files)} files in the pool", flush=True)
        return
    shas = [sha[:12] for _, sha in files]
    idx = pa.concat_tables([pq.read_table(sig_dir / f"{s}.idx.parquet") for s in shas])
    n = len(idx)
    if n * NUM_PERM * 4 > IN_RAM_BYTES:                      # the 5080 pool: one memory-mapped array
        sig = np.lib.format.open_memmap(out_dir / "sig_all.npy", mode="w+", dtype=np.uint32, shape=(n, NUM_PERM))
        r = 0
        for s in shas:
            a = np.load(sig_dir / f"{s}.npy")
            sig[r:r + len(a)] = a
            r += len(a)
        sig.flush()
    else:
        sig = np.concatenate([np.load(sig_dir / f"{s}.npy") for s in shas])
    src = idx.column("source").to_pylist()
    thr_of = {s: THRESHOLD_KEYED if keyed.get(s) else THRESHOLD_OCR for s in set(src)}
    thr = np.array([thr_of[s] for s in src], dtype=np.float64)
    t1 = time.time()
    keys = band_keys(sig, out_dir / "band_keys.npy")
    dropped, kept, jd, reason, uni_mask, samples = match(sig, idx, thr, keys)
    t_match = time.time() - t1
    aid, dates, nw = idx.column("article_id").to_pylist(), idx.column("date").to_pylist(), idx.column("n_words").to_pylist()
    per = defaultdict(Counter)
    for i in range(n):
        c = per[src[i]]
        c["docs"] += 1
        c["words"] += nw[i]
        if uni_mask[i]:
            c["dropped_at_0.80"] += 1
            c["dropped_words_at_0.80"] += nw[i]
    for i, r in zip(dropped.tolist(), reason.tolist()):
        per[src[i]][f"dropped_{r}"] += 1
        per[src[i]]["dropped_words"] += nw[i]
    print(f"{args.lang}: {n:,} docs, dropped {len(dropped):,} (exact {int((reason == 'exact').sum()):,}, near "
          f"{int((reason == 'near').sum()):,}); at a uniform 0.80: {int(uni_mask.sum()):,}; sign {t_sign:.0f}s "
          f"({signed_words / max(t_sign, 1e-9) / 1e6:.1f}M tokens/s), match {t_match:.0f}s", flush=True)
    for s, c in sorted(per.items()):
        d = c["dropped_exact"] + c["dropped_near"]
        print(f"  {s:24} thr={thr_of[s]:.2f} docs={c['docs']:>9,} dropped={d:>8,} ({100 * d / c['docs']:5.1f} %, "
              f"words {100 * c['dropped_words'] / max(c['words'], 1):5.1f} %); at 0.80: "
              f"{100 * c['dropped_at_0.80'] / c['docs']:5.1f} % docs, "
              f"{100 * c['dropped_words_at_0.80'] / max(c['words'], 1):5.1f} % words", flush=True)
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
                "params": {"SHINGLE": SHINGLE, "NUM_PERM": NUM_PERM, "BANDS": BANDS, "ROWS": ROWS, "SEED": SEED,
                           "THRESHOLD_OCR": THRESHOLD_OCR, "THRESHOLD_KEYED": THRESHOLD_KEYED,
                           "pair_threshold": "lower of the two documents' thresholds",
                           "keeper": "earliest date, article-level, id hash"},
                "threshold_per_source": dict(sorted(thr_of.items())),
                "inputs": {s: sha for (_, sha), s in zip(files, shas)},
                "docs": n, "dropped": len(dropped), "dropped_at_uniform_0.80": int(uni_mask.sum()),
                "per_source": {s: dict(c) for s, c in sorted(per.items())},
                "seconds": {"sign": round(t_sign), "match": round(t_match)}}
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
