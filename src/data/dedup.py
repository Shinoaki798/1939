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
  scale            documents under MIN_LSH_WORDS words match exactly only; candidates are verified on 8-bit
                   b-bit signatures held in RAM (Li and König 2010, bias corrected), and clusters are grown
                   band by band in a vectorised union-find, so memory stays O(documents) on the 5080 pool.

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
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.download import load_config, repo_path

SHINGLE = 5
NUM_PERM = 128
BANDS, ROWS = 32, 4
THRESHOLD_OCR, THRESHOLD_KEYED = 0.60, 0.80
SAMPLE_BAND, SAMPLE_K = (0.55, 0.65), 30
SEED = 1939
TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)
MIN_LSH_WORDS = 25              # shorter documents match exactly only (too few shingles to compare)

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
    arr = np.zeros((pf.metadata.num_rows, NUM_PERM), dtype=np.uint32)      # one array: a list of small ones doubles RAM
    r = words = 0
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
            if sig is not None:
                arr[r] = sig
            r += 1
    np.save(sig_path, arr[:r])
    pq.write_table(pa.Table.from_pydict(rows, schema=IDX_SCHEMA), idx_path, compression="zstd")
    return {"path": path, "docs": r, "words": words, "seconds": round(time.time() - t0, 1)}


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

def _chunked(col: pa.ChunkedArray, size: int = 2_000_000):
    for start in range(0, len(col), size):
        yield col.slice(start, size).to_pylist()


def priority_rank(idx: pa.Table) -> np.ndarray:
    """rank[i] = position of document i in keeper order (earliest date, article-level, id hash). Built in
    chunks: a Python list of every id of the 5080 pool would not fit beside the signatures."""
    n = len(idx)
    date = np.concatenate([np.fromiter((int((d or "0").replace("-", "")[:8] or 0) for d in c), dtype=np.int64, count=len(c))
                           for c in _chunked(idx.column("date"))]) if n else np.zeros(0, np.int64)
    page = idx.column("page_level").to_numpy(zero_copy_only=False).astype(np.int8)
    idh = np.concatenate([np.fromiter((zlib.crc32(a.encode("utf-8")) for a in c), dtype=np.int64, count=len(c))
                          for c in _chunked(idx.column("article_id"))]) if n else np.zeros(0, np.int64)
    order = np.lexsort((idh, page, date))
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank


def source_codes(idx: pa.Table) -> tuple[np.ndarray, list[str]]:
    enc = pc.dictionary_encode(idx.column("source")).combine_chunks()
    return enc.indices.to_numpy(zero_copy_only=False).astype(np.int32), enc.dictionary.to_pylist()


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


class Components:
    """Vectorised union-find over rank positions; the root of every set is its best (lowest) rank, i.e.
    the cluster's keeper. Memory is O(documents): edges are applied band by band and then discarded."""

    def __init__(self, rank: np.ndarray):
        self.rank = rank
        self.parent = np.arange(len(rank), dtype=np.int64)

    def _find(self, r: np.ndarray) -> np.ndarray:
        while True:
            p = self.parent[r]
            if np.array_equal(p, r):
                return r
            r = p

    def _compress(self) -> None:
        while True:
            pp = self.parent[self.parent]
            if np.array_equal(pp, self.parent):
                return
            self.parent = pp

    def union(self, u: np.ndarray, v: np.ndarray) -> None:
        a, b = self.rank[u], self.rank[v]
        for _ in range(1000):
            ra, rb = self._find(a), self._find(b)
            m = ra != rb
            if not m.any():
                return
            a, b, ra, rb = a[m], b[m], ra[m], rb[m]
            np.minimum.at(self.parent, np.maximum(ra, rb), np.minimum(ra, rb))   # hook the later root
            self._compress()
        raise RuntimeError("union-find did not converge")

    def keeper(self, doc_of_rank: np.ndarray) -> np.ndarray:
        self._compress()
        return doc_of_rank[self.parent[self.rank]]


def components(n: int, u: np.ndarray, v: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """Label of every document = best rank in its connected component."""
    c = Components(rank)
    if len(u):
        c.union(u, v)
    c._compress()
    return c.parent[rank]


def low_bits(sig: np.ndarray) -> np.ndarray:
    """b-bit minwise hashing (Li and König 2010): the low byte of every value, read in one sequential pass;
    128 bytes per document keeps the 5080 pool in RAM for verification."""
    out = np.empty(sig.shape, dtype=np.uint8)
    for r0 in range(0, len(sig), 1_000_000):
        out[r0:r0 + 1_000_000] = np.asarray(sig[r0:r0 + 1_000_000]) & np.uint32(0xFF)
    return out


def jaccard_b(a8: np.ndarray, b8: np.ndarray) -> np.ndarray:
    """Jaccard from 8-bit values: P(equal) = J + (1 - J) / 256."""
    p = (a8 == b8).mean(axis=-1)
    return np.clip((p - 1 / 256) / (1 - 1 / 256), 0.0, 1.0)


def band_keys(sig: np.ndarray) -> np.ndarray:
    """keys[b, i] = hash of rows b*ROWS .. b*ROWS+ROWS-1 of document i's signature (small pools, tests)."""
    n = len(sig)
    keys = np.zeros((BANDS, n), dtype=np.uint64)
    for r0 in range(0, n, 1_000_000):
        block = np.asarray(sig[r0:r0 + 1_000_000]).astype(np.uint64)
        for b in range(BANDS):
            k = np.zeros(len(block), dtype=np.uint64)
            for j in range(ROWS):
                k += block[:, b * ROWS + j] * BAND_MUL[j]
            keys[b, r0:r0 + len(block)] = k
    return keys


class StreamedBandKeys:
    """Band keys computed from the per-file signature arrays, `per_pass` bands at a time, as the high 32
    bits of the band hash (a rare collision only adds a candidate that verification rejects). Nothing is
    written to disk: rerunning the 5080 pool no longer writes 50 GB of merged signatures and keys."""

    def __init__(self, files: list[Path], n: int, per_pass: int = BANDS // 2):
        self.files, self.n, self.per = files, n, per_pass
        self.lo, self.keys = None, None

    def __getitem__(self, band: int) -> np.ndarray:
        lo = (band // self.per) * self.per
        if lo != self.lo:
            self.keys = None
            keys = np.empty((self.per, self.n), dtype=np.uint32)
            r = 0
            for f in self.files:
                a = np.load(f)
                for b in range(lo, lo + self.per):
                    k = np.zeros(len(a), dtype=np.uint64)
                    for j in range(ROWS):
                        k += a[:, b * ROWS + j].astype(np.uint64) * BAND_MUL[j]
                    keys[b - lo, r:r + len(a)] = (k >> np.uint64(32)).astype(np.uint32)
                r += len(a)
            self.lo, self.keys = lo, keys
        return self.keys[band - lo]


def streamed_low_bits(files: list[Path], n: int) -> np.ndarray:
    out = np.empty((n, NUM_PERM), dtype=np.uint8)
    r = 0
    for f in files:
        a = np.load(f)
        out[r:r + len(a)] = a & np.uint32(0xFF)
        r += len(a)
    return out


def match(sig: np.ndarray, idx: pa.Table, thr: np.ndarray, keys: np.ndarray | None = None,
          sample_band: tuple = SAMPLE_BAND, sample_k: int = SAMPLE_K, sig8: np.ndarray | None = None):
    """Clusters at the per-document thresholds `thr` (a pair uses the lower one) and, for the report, at a
    uniform THRESHOLD_KEYED. Documents shorter than MIN_LSH_WORDS match exactly only. Returns dropped,
    their keepers, Jaccard to the keeper, reason, the drop mask at the uniform threshold, and per-source
    samples of OCR pairs in `sample_band`."""
    n = len(idx)
    t0 = time.time()
    rank = priority_rank(idx)
    doc_of_rank = np.empty_like(rank)
    doc_of_rank[rank] = np.arange(n)
    has_sig = idx.column("has_sig").to_numpy(zero_copy_only=False)
    words = idx.column("n_words").to_numpy(zero_copy_only=False)
    lsh_ok = has_sig & (words >= MIN_LSH_WORDS)
    exact = idx.column("exact").to_numpy()
    codes, names = source_codes(idx)
    keys = band_keys(sig) if keys is None else keys
    sig8 = low_bits(sig) if sig8 is None else sig8
    pol, uni = Components(rank), Components(rank)
    eu, ev = group_pairs(exact, rank)
    pol.union(eu, ev)
    uni.union(eu, ev)
    if n > 1_000_000:
        print(f"  exact: {len(eu):,} pairs; {int(lsh_ok.sum()):,} of {n:,} documents in LSH; {time.time() - t0:.0f}s",
              flush=True)
    random.seed(SEED)
    samples: dict[str, list] = defaultdict(list)
    seen: Counter = Counter()
    lo, hi = sample_band
    for band in range(BANDS):
        u, v = group_pairs(np.asarray(keys[band]), rank, valid=lsh_ok)
        if len(u) == 0:
            continue
        j_est = np.concatenate([jaccard_b(sig8[u[s:s + 2_000_000]], sig8[v[s:s + 2_000_000]])
                                for s in range(0, len(u), 2_000_000)])
        ok_pol = j_est >= np.minimum(thr[u], thr[v])
        ok_uni = j_est >= THRESHOLD_KEYED
        pol.union(u[ok_pol], v[ok_pol])
        uni.union(u[ok_uni], v[ok_uni])
        for t in np.flatnonzero((j_est >= lo) & (j_est < hi) & (thr[u] < THRESHOLD_KEYED))[:20000]:
            s = names[codes[u[t]]]
            seen[s] += 1
            pair = (int(u[t]), int(v[t]), float(j_est[t]))
            if len(samples[s]) < sample_k:
                samples[s].append(pair)
            elif random.random() < sample_k / seen[s]:
                samples[s][random.randrange(sample_k)] = pair
        if n > 1_000_000:
            print(f"  band {band + 1}/{BANDS}: {len(u):,} candidates, {int(ok_pol.sum()):,} linked; "
                  f"{time.time() - t0:.0f}s", flush=True)
    keeper = pol.keeper(doc_of_rank)
    keeper_uni = uni.keeper(doc_of_rank)
    dropped = np.flatnonzero(keeper != np.arange(n))
    k = keeper[dropped]
    same = exact[dropped] == exact[k]
    jd = np.where(same, 1.0, jaccard_b(sig8[dropped], sig8[k]) if len(dropped) else 0.0)
    reason = np.where(same, "exact", "near")
    return dropped, k, jd, reason, keeper_uni != np.arange(n), dict(samples)


# ---------------------------------------------------------------- driver

def write_samples(path: Path, samples: dict, idx: pa.Table, files: list[str]) -> None:
    ids = sorted({i for pairs in samples.values() for a, b, _ in pairs for i in (a, b)})
    aid = dict(zip(ids, idx.column("article_id").take(pa.array(ids, pa.int64())).to_pylist())) if ids else {}
    want = set(aid.values())
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
    # 65M ids exceed the 2 GB offsets of `string`: take() on them needs large_string
    idx = idx.cast(pa.schema([f.with_type(pa.large_string()) if f.type == pa.string() else f for f in idx.schema]))
    n = len(idx)
    sig_files = [sig_dir / f"{s}.npy" for s in shas]
    codes, names = source_codes(idx)
    thr_of = {s: THRESHOLD_KEYED if keyed.get(s) else THRESHOLD_OCR for s in names}
    thr = np.array([thr_of[s] for s in names], dtype=np.float64)[codes]
    t1 = time.time()
    sig8 = streamed_low_bits(sig_files, n)
    print(f"8-bit signatures ready, {time.time() - t1:.0f}s", flush=True)
    dropped, kept, jd, reason, uni_mask, samples = match(None, idx, thr, StreamedBandKeys(sig_files, n), sig8=sig8)
    t_match = time.time() - t1
    nw = idx.column("n_words").to_numpy(zero_copy_only=False).astype(np.float64)
    k = len(names)
    docs, words = np.bincount(codes, minlength=k), np.bincount(codes, weights=nw, minlength=k)
    uni_docs = np.bincount(codes[uni_mask], minlength=k)
    uni_words = np.bincount(codes[uni_mask], weights=nw[uni_mask], minlength=k)
    is_exact = reason == "exact"
    d_exact = np.bincount(codes[dropped[is_exact]], minlength=k)
    d_near = np.bincount(codes[dropped[~is_exact]], minlength=k)
    d_words = np.bincount(codes[dropped], weights=nw[dropped], minlength=k)
    per = {names[j]: {"docs": int(docs[j]), "words": int(words[j]), "dropped_exact": int(d_exact[j]),
                      "dropped_near": int(d_near[j]), "dropped_words": int(d_words[j]),
                      "dropped_at_0.80": int(uni_docs[j]), "dropped_words_at_0.80": int(uni_words[j])}
           for j in range(k)}
    print(f"{args.lang}: {n:,} docs, dropped {len(dropped):,} (exact {int(is_exact.sum()):,}, near "
          f"{int((~is_exact).sum()):,}); at a uniform 0.80: {int(uni_mask.sum()):,}; sign {t_sign:.0f}s "
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
    take = pa.array(dropped, pa.int64())
    s_ = lambda a: a.cast(pa.string())
    drops = pa.table({"article_id": s_(idx.column("article_id").take(take)), "source": s_(idx.column("source").take(take)),
                      "date": s_(idx.column("date").take(take)), "reason": pa.array(reason.tolist(), pa.string()),
                      "kept_id": s_(idx.column("article_id").take(pa.array(kept, pa.int64()))),
                      "jaccard": pa.array(jd.astype(np.float32))})
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
                "per_source": dict(sorted(per.items())),
                "seconds": {"sign": round(t_sign), "match": round(t_match)}}
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
