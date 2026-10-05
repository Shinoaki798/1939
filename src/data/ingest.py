"""Ingest American Stories archives into per-year article parquet files.

Input : data/raw/american_stories/faro_YYYY.tar.gz (sha256-verified by download.py)
Output: data/ingested/american_stories/<sha256[:12]>.parquet, one file per year,
        listed in data/ingested/american_stories/MANIFEST.json

One row per article. Nothing is filtered here except rows that cannot be used
at all (empty text, unparseable date); every later decision (language, OCR
quality, length, dedup, splits) is a separate stage that reads these files.

    python -m src.data.ingest --dry-run --years 1938 --max-scans 2000
    python -m src.data.ingest --years 1900-1955 --workers 12
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import tarfile
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.download import load_config, parse_years, repo_path

SOURCE = "american_stories"
BATCH_ROWS = 100_000

SCHEMA = pa.schema([
    ("article_id", pa.string()),
    ("source", pa.string()),
    ("year", pa.int16()),
    ("date", pa.string()),          # ISO yyyy-mm-dd, from the scan filename
    ("newspaper", pa.string()),
    ("lccn", pa.string()),
    ("state", pa.string()),
    ("page", pa.string()),
    ("edition", pa.string()),
    ("headline", pa.string()),
    ("byline", pa.string()),
    ("text", pa.string()),
    ("n_bytes", pa.int32()),        # UTF-8 bytes of text (bits-per-byte denominator)
    ("n_words", pa.int32()),
    ("lang", pa.string()),          # stopword-vote language guess, "und" if undecidable
    ("lang_en_share", pa.float32()),  # English stopword hits / all stopword hits
    ("lccn_language", pa.string()),   # LoC title-level language list, ";"-joined (often empty)
])

# Short, high-frequency function-word lists. Enough to separate English from the
# immigrant-press languages in Chronicling America; not a general language-ID.
STOPWORDS = {
    "en": "the of and to in a is that for it was on with as by at he his be from this have had not are but which or were they".split(),
    "es": "el la de que y en los del se las por un una con para es su al lo como mas pero sus le ya o".split(),
    "de": "der die und in den von zu das mit sich des auf fur ist im dem nicht ein eine als auch es an werden aus er hat".split(),
    "fr": "le la les de des et en un une du est que pour dans qui au par sur pas plus ne se ce sont avec".split(),
    "it": "il di che e la per un in del della non una sono le si al da con gli nel alla".split(),
    "cs": "a v se na je to ze s z do jako pro by jsou ale od za po tak jeho jak".split(),
    "pl": "i w nie na sie z do to jest ze jak co ale za od po tak przez jego".split(),
    "nl": "de het een van en in is dat op te zijn met voor niet aan er".split(),
    "sv": "och att det som en pa ar av for med till den har de inte om ett".split(),
    "no_da": "og i at det som en pa er af for med til den har de ikke om et".split(),
}
_SW_INDEX: dict[str, list[str]] = {}
for _lang, _words in STOPWORDS.items():
    for _w in _words:
        _SW_INDEX.setdefault(_w, []).append(_lang)
_WORD = re.compile(r"[a-zA-ZÀ-ÿ]+")
_FNAME = re.compile(r"^(\d{4}-\d{2}-\d{2})_(p\d+)_.*\.json$")


def guess_lang(text: str, max_words: int = 400) -> tuple[str, float]:
    votes: Counter = Counter()
    for i, m in enumerate(_WORD.finditer(text)):
        if i >= max_words:
            break
        for lang in _SW_INDEX.get(m.group(0).lower(), ()):
            votes[lang] += 1
    total = sum(votes.values())
    if total < 5:
        return "und", float("nan")
    en_share = votes["en"] / total
    lang = votes.most_common(1)[0][0]
    return lang, en_share


def parse_scan_name(name: str) -> tuple[str, str, str] | None:
    """'faro_1938/1938-05-20_p4_sn83025247_00393340095_1938052001_0455.json' -> (date, page, edition)."""
    base = name.rsplit("/", 1)[-1]
    m = _FNAME.match(base)
    if not m:
        return None
    date, page = m.group(1), m.group(2)
    parts = base[:-5].split("_")
    edition = parts[-2][8:] if len(parts) >= 2 else ""
    return date, page, edition


def valid_date(s: str, year: int) -> bool:
    try:
        d = dt.date.fromisoformat(s)
    except ValueError:
        return False
    return d.year == year


def iter_articles(tar_path: Path, year: int, stats: Counter, max_scans: int | None):
    with tarfile.open(tar_path, mode="r|gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            if max_scans is not None and stats["scans"] >= max_scans:
                break
            meta = parse_scan_name(member.name)
            try:
                data = json.loads(tf.extractfile(member).read())
            except Exception:
                stats["scans_bad_json"] += 1
                continue
            if "lccn" not in data:
                stats["scans_no_lccn"] += 1
                continue
            stats["scans"] += 1
            if meta is None or not valid_date(meta[0], year):
                stats["scans_bad_date"] += 1
                continue
            date, page, edition = meta
            lccn = data["lccn"] or {}
            scan_id = member.name.rsplit("/", 1)[-1][:-5]
            lccn_lang = ";".join(lccn.get("language") or [])
            for art in data.get("full articles") or []:
                text = (art.get("article") or "").strip()
                if not text:
                    stats["articles_empty"] += 1
                    continue
                lang, en_share = guess_lang(text)
                n_bytes = len(text.encode("utf-8"))
                n_words = len(text.split())
                stats["articles"] += 1
                stats["bytes"] += n_bytes
                stats["words"] += n_words
                stats[f"lang_{lang}"] += 1
                yield {
                    "article_id": f"{art.get('full_article_id')}_{scan_id}",
                    "source": SOURCE,
                    "year": year,
                    "date": date,
                    "newspaper": lccn.get("title") or "",
                    "lccn": lccn.get("lccn") or "",
                    "state": lccn.get("state") or "",
                    "page": page,
                    "edition": edition,
                    "headline": art.get("headline") or "",
                    "byline": art.get("byline") or "",
                    "text": text,
                    "n_bytes": n_bytes,
                    "n_words": n_words,
                    "lang": lang,
                    "lang_en_share": en_share,
                    "lccn_language": lccn_lang,
                }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def ingest_year(job: tuple) -> dict:
    year, tar_path, out_dir, dry_run, max_scans = job
    t0 = time.time()
    stats: Counter = Counter()
    tmp = Path(out_dir) / f".tmp_{year}.parquet"
    writer = None if dry_run else pq.ParquetWriter(tmp, SCHEMA, compression="zstd")
    batch: list[dict] = []
    for row in iter_articles(Path(tar_path), year, stats, max_scans):
        if writer is None:
            continue
        batch.append(row)
        if len(batch) >= BATCH_ROWS:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
            batch = []
    result = {"year": year, "stats": dict(stats), "seconds": round(time.time() - t0, 1)}
    if writer is not None:
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
        writer.close()
        digest = sha256_file(tmp)
        final = Path(out_dir) / f"{digest[:12]}.parquet"
        os.replace(tmp, final)
        result.update({"file": final.name, "sha256": digest, "file_bytes": final.stat().st_size})
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--years", default="1900-1955")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--max-scans", type=int, default=None, help="per-year cap (for quick dry runs)")
    ap.add_argument("--dry-run", action="store_true", help="count only; write nothing")
    ap.add_argument("--force", action="store_true", help="re-ingest years already in the MANIFEST")
    args = ap.parse_args()

    cfg = load_config(Path(args.config) if args.config else repo_path("config/paths.yaml"))
    raw_dir = repo_path(cfg["raw_american_stories"])
    out_dir = repo_path(cfg["ingested_american_stories"])
    raw_manifest = json.loads((raw_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    mpath = out_dir / "MANIFEST.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {
        "stage": "ingested", "source": SOURCE, "revision": raw_manifest["revision"], "years": {}}

    jobs, skipped = [], []
    for y in parse_years(args.years):
        entry = raw_manifest["files"].get(str(y))
        if entry is None:
            skipped.append(y)
            continue
        if str(y) in manifest["years"] and not args.force and not args.dry_run:
            continue
        jobs.append((y, str(raw_dir / entry["file"]), str(out_dir), args.dry_run, args.max_scans))
    if skipped:
        print(f"not downloaded/verified yet (skipped): {skipped}")
    if not jobs:
        print("nothing to do")
        return
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'year':>4} {'scans':>8} {'articles':>10} {'words':>14} {'MB text':>9} {'en%':>6} {'secs':>6}", flush=True)
    with Pool(min(args.workers, len(jobs))) as pool:
        for res in pool.imap_unordered(ingest_year, jobs):
            s = res["stats"]
            en = s.get("lang_en", 0) / max(1, s.get("articles", 0))
            print(f"{res['year']:>4} {s.get('scans', 0):>8} {s.get('articles', 0):>10} {s.get('words', 0):>14,} "
                  f"{s.get('bytes', 0)/1e6:>9.0f} {en:>6.1%} {res['seconds']:>6}", flush=True)
            if not args.dry_run:
                old = manifest["years"].get(str(res["year"]))
                if old and old["file"] != res["file"]:
                    (out_dir / old["file"]).unlink(missing_ok=True)
                res["raw_sha256"] = raw_manifest["files"][str(res["year"])]["sha256"]
                res["ingested_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
                manifest["years"][str(res["year"])] = res
                tmp = mpath.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
                os.replace(tmp, mpath)


if __name__ == "__main__":
    sys.exit(main())
