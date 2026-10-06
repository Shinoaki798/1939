"""Ingest the approved training-only corpora into the common article schema.

Input : data/raw/<source>/*.parquet (sha256-verified by download.py)
Output: data/ingested/<source>/<sha256[:12]>.parquet, one per input file, listed in
        data/ingested/<source>/MANIFEST.json

Same columns as src.data.ingest.SCHEMA plus `meta` (JSON of source-specific fields).
English extras are training-only, so rows dated after the cutoff are dropped HERE
(and counted) — they can never be used. German sources keep every dated row up to the
end of the study range (1955): the German per-year holdout spans both sides of the
boundary (HANDOFF §12), and the split stage applies the cutoff. Rows without a valid
date are dropped everywhere.

    python -m src.data.ingest_extra --source congressional_record --dry-run
    python -m src.data.ingest_extra --source hmd_newspapers --workers 8
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.download import load_config, repo_path
from src.data.ingest import BATCH_ROWS, SCHEMA, guess_lang, sha256_file

EXTRA_SCHEMA = SCHEMA.append(pa.field("meta", pa.string()))
BATCH_BYTES = 256 << 20   # flush the parquet writer at this much text, whatever the row count


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


GERMAN_LAST_DATE = dt.date(1955, 12, 31)   # end of the study range; the split stage applies the cutoff


def _german_date(value, stats: Counter) -> dt.date | None:
    try:
        d = value if isinstance(value, dt.date) else dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        stats["dropped_bad_date"] += 1
        return None
    if d > GERMAN_LAST_DATE:
        stats["dropped_after_1955"] += 1
        return None
    return d


def rows_ddb_newspapers_de(path: Path, key: str, cutoff: dt.date, stats: Counter):
    """Deutsches Zeitungsportal pages (storytracer/German-PD-Newspapers): one row per page."""
    cols = ["issue_id", "date", "paper", "page", "language", "zdb_id", "provider", "license", "text"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2_000, columns=cols):
        for r in batch.to_pylist():
            stats["rows_in"] += 1
            d = _german_date(r.get("date"), stats)
            text = (r.get("text") or "").strip()
            if d is None:
                continue
            if not text:
                stats["dropped_empty"] += 1
                continue
            meta = {k: r.get(k) for k in ("issue_id", "page", "language", "zdb_id", "provider", "license")}
            yield _row(f"ddb_{r.get('issue_id')}_{r.get('page')}", "ddb_newspapers_de", d, r.get("paper") or "",
                       "", "", text, meta)


def rows_europeana_newspapers_de(path: Path, key: str, cutoff: dt.date, stats: Counter):
    """Europeana Newspapers German decade files (biglam/europeana_newspapers data/de-19x0): one row per page."""
    cols = ["id", "date", "title", "mean_ocr", "std_ocr", "language", "multi_language", "issue_uri", "text"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1_000, columns=cols):
        for r in batch.to_pylist():
            stats["rows_in"] += 1
            d = _german_date(r.get("date"), stats)
            text = (r.get("text") or "").strip()
            if d is None:
                continue
            if not text:
                stats["dropped_empty"] += 1
                continue
            meta = {k: r.get(k) for k in ("mean_ocr", "std_ocr", "language", "multi_language", "issue_uri")}
            yield _row(f"eu_{r.get('id')}", "europeana_newspapers_de", d, r.get("title") or "", "", "", text, meta)


_VB_DATES: dict[str, str] = {}


def rows_voelkischer_beobachter_de(path: Path, key: str, cutoff: dt.date, stats: Counter):
    """Voelkischer Beobachter issues (Internet Archive *_djvu.txt): one row per issue; the issue
    date comes from config/voelkischer_beobachter_de_dates.tsv, never from the identifier."""
    if not _VB_DATES:
        with open(repo_path("config/voelkischer_beobachter_de_dates.tsv"), encoding="utf-8") as f:
            rows = [l.rstrip("\n").split("\t") for l in f if l.strip() and not l.startswith("#")]
        _VB_DATES.update({k: v for k, v in rows[1:]})
    stats["rows_in"] += 1
    d = _german_date(_VB_DATES.get(key, ""), stats)
    if d is None:
        return
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        stats["dropped_empty"] += 1
        return
    yield _row(f"vb_{key}", "voelkischer_beobachter_de", d, "Völkischer Beobachter", "", "", text,
               {"ia_identifier": key, "named_source": "NSDAP party daily"})


BOOK_FIRST_YEAR, BOOK_LAST_YEAR = 1900, 1938   # year-only metadata: keep <= 1938 (1939 cannot be split)


def _book_row(article_id, source, year, title, author, text, meta) -> dict:
    meta = dict(meta, date_precision="year")
    return _row(article_id, source, dt.date(year, 1, 1), title, "", author, text, meta)


def rows_loc_pd_books(path: Path, key: str, cutoff: dt.date, stats: Counter):
    """storytracer/LoC-PD-Books: one row per book; year-only, kept 1900-1938."""
    cols = ["lccn", "title", "author", "year", "page_count", "filename", "text"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=200, columns=cols):
        for r in batch.to_pylist():
            stats["rows_in"] += 1
            y, text = r.get("year"), (r.get("text") or "").strip()
            if not isinstance(y, int) or not BOOK_FIRST_YEAR <= y <= BOOK_LAST_YEAR:
                stats["dropped_year_outside_1900_1938"] += 1
                continue
            if not text:
                stats["dropped_empty"] += 1
                continue
            meta = {k: r.get(k) for k in ("lccn", "page_count", "filename")}
            yield _book_row(f"loc_{r.get('lccn')}_{r.get('filename')}", "loc_pd_books", y, r.get("title") or "",
                            r.get("author") or "", text, meta)


def rows_pre_1929_books(path: Path, key: str, cutoff: dt.date, stats: Counter):
    """common-pile/pre_1929_books (jsonl.gz): one row per book; year-only, kept 1900-1938."""
    import ast
    import gzip
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            stats["rows_in"] += 1
            r = json.loads(line)
            m = r.get("metadata") or {}
            if isinstance(m, str):
                m = ast.literal_eval(m)
            try:
                y = int(float(m.get("year")))
            except (TypeError, ValueError, OverflowError):
                stats["dropped_bad_date"] += 1
                continue
            text = (r.get("text") or "").strip()
            if not BOOK_FIRST_YEAR <= y <= BOOK_LAST_YEAR:
                stats["dropped_year_outside_1900_1938"] += 1
                continue
            if not text:
                stats["dropped_empty"] += 1
                continue
            meta = {k: m.get(k) for k in ("htid", "language", "place")}
            yield _book_row(f"p1929_{r.get('id')}", "pre_1929_books", y, m.get("title") or "",
                            m.get("author") or "", text, meta)


CA_FIRST, CA_LAST = dt.date(1930, 1, 1), dt.date(1939, 6, 30)   # option C window (HANDOFF §12)
_CA_PAGE = re.compile(r"(?:^|/)(sn\d+|[a-z]{1,3}\d+)/(\d{4})/(\d{2})/(\d{2})/ed-(\d+)/seq-(\d+)/ocr\.txt$")


def rows_chronicling_america(path: Path, key: str, cutoff: dt.date, stats: Counter):
    """Chronicling America batch OCR tarball: one row per page (ocr.txt), window 1930-01-01..1939-06-30.
    Page-level OCR, training only; pages of titles already in American Stories are removed at dedup
    on (lccn, date, page)."""
    import tarfile
    with tarfile.open(path, "r|bz2") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith("ocr.txt"):
                continue
            mm = _CA_PAGE.search(m.name)
            if not mm:
                stats["skipped_unparsed_name"] += 1
                continue
            stats["rows_in"] += 1
            lccn, y, mo, d, ed, seq = mm.groups()
            try:
                date = dt.date(int(y), int(mo), int(d))
            except ValueError:
                stats["dropped_bad_date"] += 1
                continue
            if not CA_FIRST <= date <= CA_LAST:
                stats["outside_window"] += 1
                continue
            text = tf.extractfile(m).read().decode("utf-8", "replace").strip()
            if not text:
                stats["dropped_empty"] += 1
                continue
            row = _row(f"ca_{lccn}_{date.isoformat()}_ed{ed}_seq{seq}", "chronicling_america", date, lccn, "", "",
                       text, {"batch": key, "lccn": lccn, "edition": ed, "seq": seq})
            row["lccn"], row["page"], row["edition"] = lccn, f"p{seq}", ed
            yield row


def _fr_page_text(page) -> str:
    """Column-aware text of one Federal Register page: text blocks in the left half first, then the right
    half, each top to bottom; full-width blocks (headers, tables) stay in vertical order."""
    w = page.rect.width
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    def col(b):
        x0, x1 = b[0], b[2]
        return 0 if x1 <= w * 0.55 else (1 if x0 >= w * 0.45 else -1)
    full = [b for b in blocks if col(b) == -1]
    if len(full) > len(blocks) / 2:                 # mostly full-width: plain reading order
        ordered = sorted(blocks, key=lambda b: (b[1], b[0]))
    else:
        ordered = sorted(blocks, key=lambda b: (max(col(b), 0), b[1], b[0]))
    return "\n".join(b[4].strip() for b in ordered)


def rows_federal_register(path: Path, key: str, cutoff: dt.date, stats: Counter):
    """Federal Register daily issue PDF (OCR text layer): one row per issue."""
    import fitz   # pymupdf
    stats["rows_in"] += 1
    try:
        date = dt.date.fromisoformat(key[len("FR-"):])
    except ValueError:
        stats["dropped_bad_date"] += 1
        return
    if date > cutoff:
        stats["dropped_after_cutoff"] += 1
        return
    with fitz.open(path) as doc:
        text = "\n\n".join(_fr_page_text(p) for p in doc).strip()
        n_pages = doc.page_count
    if not text:
        stats["dropped_empty"] += 1
        return
    yield _row(f"fr_{key}", "federal_register", date, "Federal Register", "", "", text,
               {"issue": key, "pages": n_pages, "genre": "legal/regulatory"})


def _cap_date(value: str, cutoff: dt.date) -> tuple[dt.date | None, str]:
    """CAP decision_date is YYYY-MM-DD, YYYY-MM or YYYY. A partial date is kept only if its whole span
    lies on or before the cutoff; it is stored as the first day of the span."""
    parts = (value or "").strip().split("-")
    try:
        nums = [int(p) for p in parts]
        if len(nums) == 3:
            return dt.date(*nums), "day"
        if len(nums) == 2:
            first = dt.date(nums[0], nums[1], 1)
            last = (dt.date(nums[0] + (nums[1] == 12), nums[1] % 12 + 1, 1) - dt.timedelta(days=1))
            return (first if last <= cutoff else None), "month"
        if len(nums) == 1:
            return (dt.date(nums[0], 1, 1) if dt.date(nums[0], 12, 31) <= cutoff else None), "year"
    except ValueError:
        pass
    return None, "bad"


def rows_caselaw_access_project(path: Path, key: str, cutoff: dt.date, stats: Counter):
    """Caselaw Access Project volume zip (static.case.law): one row per case, opinion text only
    (head matter, attorneys and parties left out), kept by decision_date <= cutoff."""
    import zipfile
    with zipfile.ZipFile(path) as z:
        for name in sorted(n for n in z.namelist() if n.startswith("json/") and n.endswith(".json")):
            c = json.loads(z.read(name))
            stats["rows_in"] += 1
            date, precision = _cap_date(c.get("decision_date"), cutoff)
            if date is None:
                stats["dropped_bad_date" if precision == "bad" else "dropped_after_cutoff"] += 1
                continue
            if date > cutoff:
                stats["dropped_after_cutoff"] += 1
                continue
            ops = (c.get("casebody") or {}).get("opinions") or []
            text = "\n\n".join((o.get("text") or "").strip() for o in ops if (o.get("text") or "").strip())
            if not text:
                stats["dropped_empty"] += 1
                continue
            court = c.get("court") or {}
            juris = c.get("jurisdiction") or {}
            meta = {"volume": key, "case_id": c.get("id"), "name": c.get("name_abbreviation"),
                    "citations": [x.get("cite") for x in c.get("citations") or []],
                    "opinion_types": [o.get("type") for o in ops], "date_precision": precision,
                    "genre": "legal/regulatory"}
            yield _row(f"cap_{c.get('id')}", "caselaw_access_project", date, court.get("name") or "",
                       juris.get("name_long") or "", (ops[0].get("author") or "") if ops else "", text, meta)


ADAPTERS = {"congressional_record": rows_congressional_record, "hmd_newspapers": rows_hmd_newspapers,
            "loc_pd_books": rows_loc_pd_books, "pre_1929_books": rows_pre_1929_books,
            "chronicling_america": rows_chronicling_america, "federal_register": rows_federal_register,
            "caselaw_access_project": rows_caselaw_access_project,
            "ddb_newspapers_de": rows_ddb_newspapers_de,
            "europeana_newspapers_de": rows_europeana_newspapers_de,
            "voelkischer_beobachter_de": rows_voelkischer_beobachter_de}


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
    batch_bytes = 0
    for row in ADAPTERS[source](Path(path), key, dt.date.fromisoformat(cutoff_iso), stats):
        stats["rows_out"] += 1
        stats["words"] += row["n_words"]
        stats[f"lang_{row['lang']}"] += 1
        if writer is None:
            continue
        batch.append(row)
        batch_bytes += row["n_bytes"]
        # flush by size as well as by rows: page-level sources (Europeana, DDB) carry ~20 kB per row, and
        # 100k such rows as Python objects took ~9 GB and got a worker OOM-killed (2026-10-05)
        if len(batch) >= BATCH_ROWS or batch_bytes >= BATCH_BYTES:
            writer.write_table(pa.Table.from_pylist(batch, schema=EXTRA_SCHEMA))
            batch, batch_bytes = [], 0
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
        "stage": "ingested", "source": args.source, "hf_repo": src.get("hf_repo", args.source), "revision": raw_manifest["revision"],
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
    # ProcessPoolExecutor raises BrokenProcessPool if a worker dies (e.g. OOM); multiprocessing.Pool hung.
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        for fut in as_completed([ex.submit(ingest_file, j) for j in jobs]):
            res = fut.result()
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
