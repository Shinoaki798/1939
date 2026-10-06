"""Build the download table for USGS series via the USGS Publications Warehouse API (science bucket).

    https://pubs.usgs.gov/pubs-services/publication/?seriesName=<series>&startYear=..&endYear=..

The API gives only publicationYear, so items are year-only and must be dated <= 1938 (year-only rule).
Writes config/<source>_files.tsv (the report PDF, which carries an OCR text layer) and _items.tsv.
pubs.usgs.gov is reachable directly from the 5080 box: download with --via mirror (no VPN traffic).

    python -m src.data.usgs_catalog --source usgs_pp
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import urllib.parse
import urllib.request

from src.data.download import load_config, repo_path

API = "https://pubs.usgs.gov/pubs-services/publication/"


def records(series: str, first: int, last: int) -> list[dict]:
    out, page = [], 1
    while True:
        q = urllib.parse.urlencode({"seriesName": series, "startYear": first, "endYear": last,
                                    "page_size": 100, "page_number": page})
        with urllib.request.urlopen(API + "?" + q, timeout=120) as r:
            d = json.load(r)
        out += d["records"]
        if len(out) >= d["recordCount"] or not d["records"]:
            return out
        page += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="usgs_pp")
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    src = load_config(repo_path(cfg["sources"]))[args.source]
    lo, hi = (dt.date.fromisoformat(str(x)) for x in src["window"])
    recs = records(src["usgs_series"], lo.year, hi.year)
    kept = []
    for r in recs:
        year = int(r.get("publicationYear") or 0)
        pdfs = [l["url"] for l in r.get("links", []) if (l.get("type") or {}).get("text") == "Document"
                and l.get("url", "").lower().endswith(".pdf")]
        if lo.year <= year <= hi.year and dt.date(year, 12, 31) <= hi and pdfs:
            kept.append((r["indexId"], year, pdfs[0], re.sub(r"\s+", " ", r.get("title", "")).strip()))
    with open(repo_path(src["files"]), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# {args.source}: USGS {src['usgs_series']} {lo.year}-{hi.year} report PDFs; built {dt.date.today()} "
                 f"by src.data.usgs_catalog\nkey\turl\tbytes\tchecksum\tfile\n")
        for key, _, url, _ in kept:
            fh.write(f"{key}\t{url}\t?\t-\t{key}.pdf\n")
    with open(repo_path(src["items"]), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("key\tdate\tprecision\tyear\tvolume\ttitle\tlanguage\n")
        for key, year, _, title in kept:
            fh.write(f"{key}\t{year}-01-01\tyear\t{year}\t\t{title}\tEnglish\n")
    print(f"{args.source}: {len(recs)} records, {len(kept)} kept (year <= {hi.year}, with a PDF)", flush=True)


if __name__ == "__main__":
    main()
