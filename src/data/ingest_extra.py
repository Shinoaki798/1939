"""Ingest the approved training-only corpora into the common article schema.

Input : data/raw/<source>/*.parquet (sha256-verified by download.py)
Output: data/ingested/<source>/<sha256[:12]>.parquet, one per input file, listed in
        data/ingested/<source>/MANIFEST.json

Same columns as src.data.ingest.SCHEMA plus `meta` (JSON of source-specific fields).
These sources are training-only, so rows dated after the cutoff are dropped HERE
(and counted) — they can never be used. Rows without a valid date are dropped too.

    python -m src.data.ingest_extra --source congressional_record --dry-run
    python -m src.data.ingest_extra --source hmd_newspapers --workers 8
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.download import load_config, repo_path
from src.data.ingest import BATCH_ROWS, SCHEMA, guess_lang, sha256_file

EXTRA_SCHEMA = SCHEMA.append(pa.field("meta", pa.string()))


def _row(article_id, source, date, newspaper, state, byline, text, meta) -> dict:
    lang, en_share = guess_lang(text)
    return {
        "article_id": article_id, "source": source, "year": date.year, "date": date.isoformat(),
        "newspaper": newspaper, "lccn": "", "state": state, "page": "", "edition": "",
        "headline": "", "byline": byline, "text": text,
        "n_bytes": len(text.encode("utf-8")), "n_words": len(text.split()),
        "lang": lang, "lang_en_share": en_share, "lccn_language": "",
        "meta": json.dumps(meta, ensure_ascii=False, sort_keys=True),
    }


def rows_congressional_record(path: Path, key: str, cutoff: dt.date, stats: Counter):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=50_000):
        for r in batch.to_pylist():
            stats["rows_in"] += 1
            text = (r.get("speech_text") or "").strip()
            try:
                d = dt.datetime.strptime(str(r.get("date")), "%Y%m%d").date()
            except ValueError:
                stats["dropped_bad_date"] += 1
                continue
            if d > cutoff:
                stats["dropped_after_cutoff"] += 1
                continue
            if not text:
                stats["dropped_empty"] += 1
                continue
            chamber = {"S": "Senate", "H": "House"}.get(r.get("chamber") or "", r.get("chamber") or "")
            meta = {k: r.get(k) for k in ("speech_id", "congress", "chamber", "speaker", "party", "state", "source")}
            yield _row(f"cr_{r.get('speech_id')}", "congressional_record", d,
                       f"Congressional Record ({chamber})", r.get("state") or "", r.get("speaker") or "", text, meta)


def rows_hmd_newspapers(path: Path, key: str, cutoff: dt.date, stats: Counter):
    pf = pq.ParquetFile(path)
    i = 0
    for batch in pf.iter_batches(batch_size=20_000):
        for r in batch.to_pylist():
            stats["rows_in"] += 1
            idx, i = i, i + 1
            text = (r.get("text") or "").strip()
            ts = r.get("date")
            if ts is None:
                stats["dropped_bad_date"] += 1
                continue
            d = ts.date() if isinstance(ts, dt.datetime) else ts
            if d > cutoff:
                stats["dropped_after_cutoff"] += 1
                continue
            if not text:
                stats["dropped_empty"] += 1
                continue
            meta = {k: r.get(k) for k in ("item_type", "ocr_quality_mean", "ocr_quality_sd", "location")}
            yield _row(f"hmd_{key}_{idx}", "hmd_newspapers", d, r.get("title") or "", r.get("location") or "",
                       "", text, meta)


ADAPTERS = {"congressional_record": rows_congressional_record, "hmd_newspapers": rows_hmd_newspapers}


def ingested_dir(cfg: dict, name: str, src: dict) -> Path:
    """English: <ingested_root>/<name>. Foreign (user: stored apart): <foreign_root>/<lang>/ingested/<name>."""
    lang = src.get("lang", "en")
    if lang == "en":
        return repo_path(cfg["ingested_root"]) / name
    return repo_path(cfg["foreign_root"]) / lang / "ingested" / name


def ingest_file(job: tuple) -> dict:
    source, key, path, out_dir, cutoff_iso, dry_run = job
    t0 = time.time()
    stats: Counter = Counter()
    tmp = Path(out_dir) / f".tmp_{key}.parquet"
    writer = None if dry_run else pq.ParquetWriter(tmp, EXTRA_SCHEMA, compression="zstd")
    batch: list[dict] = []
    for row in ADAPTERS[source](Path(path), key, dt.date.fromisoformat(cutoff_iso), stats):
        stats["rows_out"] += 1
        stats["words"] += row["n_words"]
        stats[f"lang_{row['lang']}"] += 1
        if writer is None:
            continue
        batch.append(row)
        if len(batch) >= BATCH_ROWS:
            writer.write_table(pa.Table.from_pylist(batch, schema=EXTRA_SCHEMA))
            batch = []
    result = {"key": key, "stats": dict(stats), "seconds": round(time.time() - t0, 1)}
    if writer is not None:
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=EXTRA_SCHEMA))
        writer.close()
        digest = sha256_file(tmp)
        final = Path(out_dir) / f"{digest[:12]}.parquet"
        os.replace(tmp, final)
        result.update({"file": final.name, "sha256": digest, "file_bytes": final.stat().st_size})
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="count only; write nothing")
    args = ap.parse_args()

    cfg = load_config(Path(args.config) if args.config else repo_path("config/paths.yaml"))
    src = load_config(repo_path(cfg["sources"]))[args.source]
    raw_dir = repo_path(src["dest"])
    out_dir = ingested_dir(cfg, args.source, src)
    raw_manifest = json.loads((raw_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    mpath = out_dir / "MANIFEST.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {
        "stage": "ingested", "source": args.source, "hf_repo": src["hf_repo"], "revision": raw_manifest["revision"],
        "cutoff": str(cfg["cutoff"]), "files": {}}

    jobs = [(args.source, k, str(raw_dir / e["file"]), str(out_dir), str(cfg["cutoff"]), args.dry_run)
            for k, e in sorted(raw_manifest["files"].items())
            if args.dry_run or k not in manifest["files"]]
    if not jobs:
        print("nothing to do")
        return
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    totals: Counter = Counter()
    with Pool(min(args.workers, len(jobs))) as pool:
        for res in pool.imap_unordered(ingest_file, jobs):
            s = res["stats"]
            totals.update(s)
            print(f"{res['key']:>24} in={s.get('rows_in', 0):>9} out={s.get('rows_out', 0):>9} "
                  f"after_cutoff={s.get('dropped_after_cutoff', 0):>7} words={s.get('words', 0):>13,} "
                  f"{res['seconds']:>6}s", flush=True)
            if not args.dry_run:
                res["raw_sha256"] = raw_manifest["files"][res["key"]]["sha256"]
                res["ingested_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
                manifest["files"][res["key"]] = res
                tmp = mpath.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
                os.replace(tmp, mpath)
    langs = {k[5:]: v for k, v in totals.items() if k.startswith("lang_")}
    print(f"TOTAL rows_out={totals['rows_out']:,} words={totals['words']:,} after_cutoff_dropped="
          f"{totals['dropped_after_cutoff']:,} bad_date={totals['dropped_bad_date']:,} langs={langs}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
