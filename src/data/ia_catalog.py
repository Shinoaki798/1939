"""Build download tables for archive.org text sources (science bucket, HANDOFF §12 2026-10-06).

For every source in config/sources.yaml with an `ia_query`, run the query through archive.org's
advancedsearch API (one request per source, through the Windows-side proxy like download.py), keep
items dated inside the source's `window`, and write
  config/<source>_files.tsv   key, url (<id>_djvu.txt), bytes '?', checksum '-'   -> src.data.download
  config/<source>_items.tsv   key, date, precision, year, volume, title, language  -> src.data.ingest_extra
Dates: issue date from the identifier where it carries one (sim_/per_ microfilm items), otherwise
the item's `date` field; a date of YYYY-01-01 or a bare year counts as year-only. Year-only items must
lie wholly inside the window (so a year-only 1939 item is dropped). Items in `ia_exclude` are skipped.

    python -m src.data.ia_catalog --source pnas_ia,bams_ia
    python -m src.data.ia_catalog --all --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import time
import urllib.parse

from src.data.download import CURL_EXE, load_config, repo_path

SEARCH = "https://archive.org/advancedsearch.php"
FIELDS = ("identifier", "date", "year", "volume", "title", "language")
_ID_DATE = re.compile(r"_(\d{4})-(\d{2})(?:-(\d{2}))?_")


def search(query: str, proxy: str, rows: int = 20000) -> list[dict]:
    params = [("q", query), ("rows", str(rows)), ("page", "1"), ("output", "json"), ("sort[]", "identifier asc")]
    params += [("fl[]", f) for f in FIELDS]
    url = SEARCH + "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        r = subprocess.run([CURL_EXE, "-sS", "--fail", "--max-time", "300", "-x", proxy, url],
                           stdin=subprocess.DEVNULL, capture_output=True)
        if r.returncode == 0:
            resp = json.loads(r.stdout.decode("utf-8"))["response"]
            if resp["numFound"] > rows:
                sys.exit(f"{query}: {resp['numFound']} hits > {rows}; narrow the query")
            return resp["docs"]
        print(f"search failed (curl {r.returncode}): {r.stderr.decode(errors='replace')[:200]}", flush=True)
        time.sleep(30 * (attempt + 1))
    sys.exit(f"search failed: {query}")


def _first(v):
    return (v[0] if isinstance(v, list) and v else v) or ""


def item_date(doc: dict) -> tuple[dt.date | None, str]:
    m = _ID_DATE.search(doc["identifier"])
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), m.group(3)
        try:
            return dt.date(y, mo, int(d) if d else 1), "day" if d else "month"
        except ValueError:
            pass
    s = str(_first(doc.get("date")) or _first(doc.get("year")))
    m = re.match(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?", s)
    if not m:
        return None, "none"
    y, mo, d = int(m.group(1)), m.group(2), m.group(3)
    if mo is None or (mo == "01" and (d in (None, "01"))):
        return dt.date(y, 1, 1), "year"
    return dt.date(y, int(mo), int(d) if d and d != "00" else 1), "day" if d and d != "00" else "month"


def in_window(date: dt.date, precision: str, lo: dt.date, hi: dt.date) -> bool:
    """The item's whole date span must lie inside [lo, hi]."""
    if precision == "year":
        start, end = dt.date(date.year, 1, 1), dt.date(date.year, 12, 31)
    elif precision == "month":
        start = date.replace(day=1)
        end = (start.replace(year=start.year + (start.month == 12), month=start.month % 12 + 1) - dt.timedelta(days=1))
    else:
        start = end = date
    return start >= lo and end <= hi


def build(name: str, src: dict, proxy: str, dry_run: bool) -> dict:
    lo, hi = (dt.date.fromisoformat(str(x)) for x in src["window"])
    exclude = set(src.get("ia_exclude") or [])
    docs = search(src["ia_query"], proxy)
    kept, stats = [], {"hits": len(docs), "excluded": 0, "undated": 0, "outside_window": 0}
    for doc in docs:
        ident = doc["identifier"]
        if ident in exclude:
            stats["excluded"] += 1
            continue
        date, precision = item_date(doc)
        if src.get("ia_date_from") == "volume":        # serial scans whose `date` is the run's start year
            years = [int(y) for y in re.findall(r"\b(1[6-9]\d\d)\b", str(_first(doc.get("volume"))))]
            date, precision = (dt.date(max(years), 1, 1), "year") if years else (None, "none")
        if date is None:
            stats["undated"] += 1
            continue
        if not in_window(date, precision, lo, hi):
            stats["outside_window"] += 1
            continue
        kept.append((ident, date, precision, doc))
    stats["kept"] = len(kept)
    if not dry_run:
        clean = lambda v: re.sub(r"\s+", " ", str(_first(v))).strip()
        with open(repo_path(src["files"]), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"# {name}: archive.org djvu.txt, query {src['ia_query']!r}, window {lo}..{hi}; "
                     f"built {dt.date.today()} by src.data.ia_catalog\nkey\turl\tbytes\tchecksum\n")
            for ident, *_ in kept:
                fh.write(f"{ident}\thttps://archive.org/download/{ident}/{ident}_djvu.txt\t?\t-\n")
        with open(repo_path(src["items"]), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("key\tdate\tprecision\tyear\tvolume\ttitle\tlanguage\n")
            for ident, date, precision, doc in kept:
                fh.write("\t".join([ident, date.isoformat(), precision, str(date.year), clean(doc.get("volume")),
                                    clean(doc.get("title")), clean(doc.get("language"))]) + "\n")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="", help="comma-separated source names")
    ap.add_argument("--all", action="store_true", help="every source with an ia_query")
    ap.add_argument("--dry-run", action="store_true", help="count only; write nothing")
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    sources = load_config(repo_path(cfg["sources"]))
    names = [n for n, s in sources.items() if s.get("ia_query")] if args.all else [n for n in args.source.split(",") if n]
    for name in names:
        stats = build(name, sources[name], cfg["download"]["proxy"], args.dry_run)
        print(f"{name:28s} {json.dumps(stats)}", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
