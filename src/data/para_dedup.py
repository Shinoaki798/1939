"""Paragraph-level dedup of page-level training sources (HANDOFF §12, 2026-10-07, decision 2).

Page-level sources (whole OCR pages or whole books: Chronicling America batch OCR, DDB, Europeana, LoC
and pre-1929 books, Federal Register) carry reprinted articles inside pages, which whole-document MinHash
cannot see. Their text has no paragraph marks (one OCR line per line), so a document is cut at line
boundaries into blocks of at least BLOCK_WORDS words (a tail shorter than MIN_TAIL joins the previous
block); block i of a document is reproducible from its text. Containment of a block = share of its
distinct word-5-gram hashes (src.data.dedup.shingles) found in a reference set. Only TRAINING documents
(src.data.splits) that survived document dedup lose blocks; removed blocks are listed, not cut here.

  heldout (2a, required before freeze): reference = 5-grams of every American Stories article in the
          holdout or Val split that survived document dedup. A block with containment >= CUTOFF is removed,
          whatever the dates (evaluation integrity). Hit counts at several cutoffs go to the MANIFEST.
          German pages cannot match English articles (different language pool), so 2a runs on `en`.
  reprint (2b, may be partial): documents of the language pool in date order (ties: article-level first,
          then id hash); a page-level training block whose containment in the 5-grams of everything
          earlier (rolling window of the current and previous year, a Bloom filter) is >= CUTOFF is
          removed; every other block and every article-level document adds its 5-grams. Article-level
          documents are references only (their duplicates are document dedup's business). Years run in
          parallel: each year job fills its previous-year filter from that year's text itself.

CUTOFF 0.5: on 30 duplicate Chronicling America page pairs (2080 experiment) the median block
containment was 0.82 and 84 % of blocks reached 0.5; against unrelated pages no block reached 0.3.

Output data/para_dedup/<lang>/<pass>_drops.parquet (article_id, source, date, block, start, end,
n_words, containment) with every block at containment >= REPORT_FROM, and MANIFEST_<pass>.json.

    python -m src.data.para_dedup heldout --lang en [--workers 8]
    python -m src.data.para_dedup reprint --lang en [--sources chronicling_america,...]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.dedup import comparison_tokens, selected_files, shingles
from src.data.download import load_config, repo_path
from src.data.splits import split_of

PARA_SOURCES = {"chronicling_america", "ddb_newspapers_de", "europeana_newspapers_de", "loc_pd_books",
                "pre_1929_books", "federal_register"}
REPRINT_DEFAULT = ("chronicling_america", "ddb_newspapers_de", "europeana_newspapers_de", "federal_register")
BLOCK_WORDS, MIN_TAIL = 60, 20
CUTOFF, REPORT_FROM = 0.5, 0.3
CUTOFFS_REPORTED = (0.3, 0.4, 0.5, 0.6, 0.7)
BLOOM_BITS_PER_ITEM, BLOOM_K = 14.4, 10             # false-positive rate ~0.1 %

DROP_SCHEMA = pa.schema([("article_id", pa.string()), ("source", pa.string()), ("date", pa.string()),
                         ("block", pa.int32()), ("start", pa.int32()), ("end", pa.int32()),
                         ("n_words", pa.int32()), ("containment", pa.float32())])
COLS = ["article_id", "source", "date", "date_class", "page_level", "text"]


def blocks(text: str, min_words: int = BLOCK_WORDS, min_tail: int = MIN_TAIL) -> list[tuple[int, int]]:
    """(start, end) character spans of the blocks of a document, cut at line boundaries."""
    spans, start, n, pos = [], 0, 0, 0
    for line in text.splitlines(keepends=True):
        pos += len(line)
        n += len(line.split())
        if n >= min_words:
            spans.append((start, pos))
            start, n = pos, 0
    if start < len(text) and text[start:].strip():
        if spans and n < min_tail:
            spans[-1] = (spans[-1][0], len(text))
        else:
            spans.append((start, len(text)))
    return spans


def block_shingles(text: str, span: tuple[int, int]) -> np.ndarray:
    return np.unique(shingles(comparison_tokens(text[span[0]:span[1]])))


def drops_path(cfg: dict, lang: str) -> str:
    p = repo_path(cfg["dedup"]) / lang / "drops.parquet"
    if not p.exists():
        sys.exit(f"{p} missing: run `python -m src.data.dedup --lang {lang}` (not --dry-run) first")
    return str(p)


_DROPS: dict[str, set[str]] = {}


def dropped_ids(path: str) -> set[str]:
    """Document-dedup drops, loaded once per worker process."""
    if path not in _DROPS:
        _DROPS[path] = set(pq.read_table(path, columns=["article_id"]).column("article_id").to_pylist())
    return _DROPS[path]


def out_dir(cfg: dict, lang: str) -> Path:
    d = repo_path(cfg["data_root"]) / "para_dedup" / lang
    d.mkdir(parents=True, exist_ok=True)
    return d


def _iter_docs(path: str, drops_file: str, sources: set[str] | None = None):
    """(article_id, source, date, split, page_level, text) of surviving documents of a selected file."""
    drops = dropped_ids(drops_file)
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2000, columns=COLS):
        b = batch.to_pydict()
        for i in range(batch.num_rows):
            aid, src = b["article_id"][i], b["source"][i]
            if (sources is not None and src not in sources) or aid in drops:
                continue
            yield aid, src, b["date"][i], split_of(aid, b["date_class"][i]), b["page_level"][i], b["text"][i] or ""


# ---------------------------------------------------------------- 2a: held-out protection

def _heldout_ref_job(job: tuple) -> np.ndarray:
    path, drops = job
    parts = [np.unique(shingles(comparison_tokens(t)))
             for aid, src, date, split, page, t in _iter_docs(path, drops, {"american_stories"})
             if split in ("holdout", "val")]
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.uint64)


_REF: np.ndarray | None = None


def _containment(sh: np.ndarray, ref: np.ndarray) -> float:
    if len(sh) == 0 or len(ref) == 0:
        return 0.0
    pos = np.searchsorted(ref, sh)
    pos[pos == len(ref)] = 0
    return float((ref[pos] == sh).mean())


def _heldout_job(job: tuple) -> dict:
    global _REF
    path, ref_path, drops = job
    if _REF is None:
        _REF = np.load(ref_path, mmap_mode="r")
    rows, stats = defaultdict(list), Counter()
    for aid, src, date, split, page, text in _iter_docs(path, drops, PARA_SOURCES):
        if split != "train":
            continue
        stats[f"{src}|docs"] += 1
        for k, span in enumerate(blocks(text)):
            sh = block_shingles(text, span)
            if len(sh) == 0:
                continue
            c = _containment(sh, _REF)
            stats[f"{src}|blocks"] += 1
            if c < REPORT_FROM:
                continue
            nw = len(text[span[0]:span[1]].split())
            for cut in CUTOFFS_REPORTED:
                if c >= cut:
                    stats[f"{src}|hits@{cut}"] += 1
                    stats[f"{src}|words@{cut}"] += nw
            for key, val in (("article_id", aid), ("source", src), ("date", date), ("block", k),
                             ("start", span[0]), ("end", span[1]), ("n_words", nw), ("containment", c)):
                rows[key].append(val)
    return {"rows": dict(rows), "stats": dict(stats)}


def heldout(cfg: dict, lang: str, workers: int) -> None:
    files, _ = selected_files(cfg, lang)
    drops = drops_path(cfg, lang)
    od = out_dir(cfg, lang)
    t0 = time.time()
    as_files = [p for p, _ in files if "/american_stories/" in p.replace("\\", "/")]
    if not as_files:
        sys.exit(f"no American Stories in the {lang} pool: nothing to protect")
    with ProcessPoolExecutor(max_workers=min(workers, len(as_files))) as ex:
        parts = [f.result() for f in as_completed([ex.submit(_heldout_ref_job, (p, drops)) for p in as_files])]
    ref = np.unique(np.concatenate(parts))
    ref_path = od / "heldout_shingles.npy"
    np.save(ref_path, ref)
    del parts
    print(f"held-out reference: {len(ref):,} distinct 5-grams from {len(as_files)} files, {time.time() - t0:.0f}s",
          flush=True)
    page_files = [p for p, _ in files if any(f"/{s}/" in p.replace("\\", "/") for s in PARA_SOURCES)]
    rows, stats = defaultdict(list), Counter()
    with ProcessPoolExecutor(max_workers=min(workers, max(1, len(page_files)))) as ex:
        for fut in as_completed([ex.submit(_heldout_job, (p, str(ref_path), drops)) for p in page_files]):
            r = fut.result()
            stats.update(r["stats"])
            for k, v in r["rows"].items():
                rows[k].extend(v)
    write_pass(od, "heldout", rows, stats, {"reference_5grams": len(ref), "reference": "American Stories holdout + val",
                                            "seconds": round(time.time() - t0)})


# ---------------------------------------------------------------- 2b: reprints within the pool

class Bloom:
    def __init__(self, n_items: int):
        self.m = max(1 << 20, int(n_items * BLOOM_BITS_PER_ITEM))
        self.bits = np.zeros((self.m + 7) // 8, dtype=np.uint8)

    def _pos(self, h: np.ndarray) -> np.ndarray:
        h1 = (h & np.uint64(0xFFFFFFFF)).astype(np.uint64)
        h2 = ((h >> np.uint64(32)) | np.uint64(1)).astype(np.uint64)
        i = np.arange(BLOOM_K, dtype=np.uint64)
        return ((h1[:, None] + i[None, :] * h2[:, None]) % np.uint64(self.m)).ravel()

    def contains(self, h: np.ndarray) -> np.ndarray:
        p = self._pos(h)
        hit = (self.bits[p >> np.uint64(3)] >> (p & np.uint64(7)).astype(np.uint8)) & 1
        return hit.reshape(len(h), BLOOM_K).all(axis=1)

    def add(self, h: np.ndarray) -> None:
        p = self._pos(h)
        np.bitwise_or.at(self.bits, (p >> np.uint64(3)).astype(np.int64), (1 << (p & np.uint64(7))).astype(np.uint8))


def _year_shard_job(job: tuple) -> dict:
    """Split one selected file into per-year shards (surviving pre-cutoff documents only)."""
    path, drops, shard_dir, sources = job
    by_year: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for aid, src, date, split, page, text in _iter_docs(path, drops):
        if split in ("embargo", "post") or not date[:4].isdigit():
            continue
        y = by_year[int(date[:4])]
        for k, v in (("article_id", aid), ("source", src), ("date", date), ("split", split),
                     ("droppable", src in sources and split == "train"), ("page_level", bool(page)), ("text", text)):
            y[k].append(v)
    words, droppable = {}, set()
    stem = Path(path).stem
    for year, cols in by_year.items():
        d = Path(shard_dir) / str(year)
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(cols), d / f"{stem}.parquet", compression="zstd")
        words[year] = sum(len(t.split()) for t in cols["text"])
        if any(cols["droppable"]):
            droppable.add(year)
    return {"words": words, "droppable_years": droppable}


def _load_year(shard_dir: Path, year: int) -> dict | None:
    files = sorted((shard_dir / str(year)).glob("*.parquet"))
    if not files:
        return None
    t = pa.concat_tables([pq.read_table(f) for f in files])
    return {c: t.column(c).to_pylist() for c in ("article_id", "source", "date", "page_level", "droppable", "text")}


def _reprint_year_job(job: tuple) -> dict:
    """One year: the previous year's text fills one Bloom filter, then this year's documents run in date
    order (ties: article-level first, then id hash) against both. Years are independent, so they run in
    parallel; the previous year is inserted whole (its own removed blocks repeat earlier text anyway)."""
    shard_dir, year, words_prev, words_cur, heldout_path = job
    shard_dir = Path(shard_dir)
    heldout_blocks = set()
    if heldout_path:
        t = pq.read_table(heldout_path, columns=["article_id", "block", "containment"], filters=[("containment", ">=", CUTOFF)])
        heldout_blocks = set(zip(t.column("article_id").to_pylist(), t.column("block").to_pylist()))
    prev = None
    py = _load_year(shard_dir, year - 1)
    if py is not None:
        prev = Bloom(int(words_prev * 1.05))
        for text in py["text"]:
            sh = np.unique(shingles(comparison_tokens(text)))
            if len(sh):
                prev.add(sh)
        del py
    d = _load_year(shard_dir, year)
    cur = Bloom(int(words_cur * 1.05))
    aid, src, date, page, drop_ok, text = (d[c] for c in ("article_id", "source", "date", "page_level", "droppable", "text"))
    order = sorted(range(len(aid)), key=lambda i: (date[i], bool(page[i]), zlib.crc32(aid[i].encode("utf-8"))))
    rows, stats = defaultdict(list), Counter()
    for i in order:
        if not drop_ok[i]:
            sh = np.unique(shingles(comparison_tokens(text[i])))
            if len(sh):
                cur.add(sh)
            continue
        stats[f"{src[i]}|docs"] += 1
        for k, span in enumerate(blocks(text[i])):
            if (aid[i], k) in heldout_blocks:
                continue
            sh = block_shingles(text[i], span)
            if len(sh) == 0:
                continue
            stats[f"{src[i]}|blocks"] += 1
            seen = cur.contains(sh)
            if prev is not None:
                seen |= prev.contains(sh)
            c = float(seen.mean())
            if c >= CUTOFF:
                nw = len(text[i][span[0]:span[1]].split())
                stats[f"{src[i]}|hits@{CUTOFF}"] += 1
                stats[f"{src[i]}|words@{CUTOFF}"] += nw
                for key, val in (("article_id", aid[i]), ("source", src[i]), ("date", date[i]), ("block", k),
                                 ("start", span[0]), ("end", span[1]), ("n_words", nw), ("containment", c)):
                    rows[key].append(val)
            else:
                cur.add(sh)
    return {"year": year, "rows": dict(rows), "stats": dict(stats)}


def reprint(cfg: dict, lang: str, sources: set[str], workers: int) -> None:
    files, _ = selected_files(cfg, lang)
    drops = drops_path(cfg, lang)
    od = out_dir(cfg, lang)
    shard_dir = od / "by_year"            # this pass's own scratch: rebuilt every run, removed at the end
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    t0 = time.time()
    words_by_year: Counter = Counter()
    droppable_years: set[int] = set()
    with ProcessPoolExecutor(max_workers=min(workers, len(files))) as ex:
        for fut in as_completed([ex.submit(_year_shard_job, (p, drops, str(shard_dir), sources)) for p, _ in files]):
            r = fut.result()
            words_by_year.update(r["words"])
            droppable_years.update(r["droppable_years"])
    hp = od / "heldout_drops.parquet"
    print(f"shards written for {len(words_by_year)} years ({len(droppable_years)} with page-level training "
          f"documents), {time.time() - t0:.0f}s", flush=True)
    rows, stats = defaultdict(list), Counter()
    jobs = [(str(shard_dir), y, words_by_year.get(y - 1, 0), words_by_year[y], str(hp) if hp.exists() else "")
            for y in sorted(droppable_years)]
    with ProcessPoolExecutor(max_workers=min(workers, max(1, len(jobs)))) as ex:
        for fut in as_completed([ex.submit(_reprint_year_job, j) for j in jobs]):
            r = fut.result()
            stats.update(r["stats"])
            for k, v in r["rows"].items():
                rows[k].extend(v)
            print(f"{r['year']}: {sum(v for k, v in r['stats'].items() if k.endswith(f'hits@{CUTOFF}')):,} blocks removed, "
                  f"{time.time() - t0:.0f}s", flush=True)
    shutil.rmtree(shard_dir, ignore_errors=True)
    write_pass(od, "reprint", rows, stats, {"sources": sorted(sources), "window": "current + previous year",
                                            "bloom": {"bits_per_item": BLOOM_BITS_PER_ITEM, "k": BLOOM_K},
                                            "seconds": round(time.time() - t0)})


# ---------------------------------------------------------------- output

def write_pass(od: Path, name: str, rows: dict, stats: Counter, extra: dict) -> None:
    pq.write_table(pa.table({k: rows.get(k, []) for k in DROP_SCHEMA.names}, schema=DROP_SCHEMA),
                   od / f"{name}_drops.parquet", compression="zstd")
    per = defaultdict(dict)
    for k, v in stats.items():
        s, key = k.split("|", 1)
        per[s][key] = v
    manifest = {"stage": f"para_dedup:{name}", "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "params": {"BLOCK_WORDS": BLOCK_WORDS, "MIN_TAIL": MIN_TAIL, "CUTOFF": CUTOFF, "REPORT_FROM": REPORT_FROM},
                "per_source": dict(sorted(per.items())), **extra}
    (od / f"MANIFEST_{name}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for s, c in sorted(per.items()):
        hits = {k: v for k, v in c.items() if k.startswith("hits@")}
        print(f"  {s:24} docs={c.get('docs', 0):>9,} blocks={c.get('blocks', 0):>10,} {hits}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["heldout", "reprint"])
    ap.add_argument("--lang", required=True, choices=["en", "de"])
    ap.add_argument("--sources", default=",".join(REPRINT_DEFAULT), help="reprint: page-level sources whose blocks may go")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    if args.command == "heldout":
        heldout(cfg, args.lang, args.workers)
    else:
        reprint(cfg, args.lang, {s for s in args.sources.split(",") if s} & PARA_SOURCES, args.workers)


if __name__ == "__main__":
    main()
