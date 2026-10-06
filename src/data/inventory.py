"""Inventory of every ingested source: documents / words / est. tokens by period x form x language.

Reads data/ingested/<source>/ and data/foreign/<lang>/ingested/<source>/ (columns date, n_words,
lang, source). Counts are RAW: before dedup, OCR filtering and language filtering. The
`en-voted` column (English sources) is the share of words in articles whose stopword vote is
English, a preview of the language filter.

    python -m src.data.inventory            # writes reports/inventory.md and logs/inventory.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.download import load_config, repo_path
from src.data.ingest_extra import ingested_dir

PERIODS = [("pre-1900", "0000-01-01", "1899-12-31"), ("1900-1919", "1900-01-01", "1919-12-31"),
           ("1920-1929", "1920-01-01", "1929-12-31"), ("1930-1933", "1930-01-01", "1933-12-31"),
           ("1934-1936", "1934-01-01", "1936-12-31"), ("1937-1939.06", "1937-01-01", "1939-06-30"),
           ("embargo 1939.07-08", "1939-07-01", "1939-08-31"), ("1939.09-1955", "1939-09-01", "1955-12-31")]

FORM = {  # what kind of text each source is
    "american_stories": "newspaper (US, article)", "chronicling_america": "newspaper (US, page OCR)",
    "hmd_newspapers": "newspaper (UK, 19th c.)", "congressional_record": "parliamentary speech",
    "federal_register": "legal/regulatory", "caselaw_access_project": "legal/regulatory",
    "loc_pd_books": "books", "pre_1929_books": "books",
    "ddb_newspapers_de": "newspaper (DE, page)", "europeana_newspapers_de": "newspaper (DE, page)",
    "voelkischer_beobachter_de": "newspaper (DE, NSDAP organ)",
}
IN_TRAINING = {"american_stories", "chronicling_america", "congressional_record", "federal_register",
               "caselaw_access_project", "loc_pd_books", "pre_1929_books", "ddb_newspapers_de",
               "europeana_newspapers_de", "voelkischer_beobachter_de"}
TOKENS_PER_WORD = {"en": 1.3, "de": 1.5}   # rough, until the 48k BPE exists


def scan(path: str) -> dict:
    t = pq.read_table(path, columns=["date", "n_words", "lang"])
    out = {}
    for name, a, b in PERIODS:
        m = pc.and_(pc.greater_equal(t["date"], a), pc.less_equal(t["date"], b))
        if not pc.any(m).as_py():
            continue
        w = pc.if_else(m, t["n_words"], 0)
        en = pc.and_(m, pc.equal(t["lang"], "en"))
        out[name] = (pc.sum(pc.cast(m, "int64")).as_py(), pc.sum(w).as_py() or 0,
                     pc.sum(pc.if_else(en, t["n_words"], 0)).as_py() or 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    sources = load_config(repo_path(cfg["sources"]))
    jobs = []
    for name, src in sources.items():
        d = repo_path(cfg["ingested_american_stories"]) if name == "american_stories" else ingested_dir(cfg, name, src)
        for f in sorted(d.glob("*.parquet")) if d.exists() else []:
            jobs.append((name, src.get("lang", "en"), str(f)))
    agg = defaultdict(lambda: [0, 0, 0])
    with ProcessPoolExecutor(args.workers) as ex:
        for (name, lang, _), res in zip(jobs, ex.map(scan, [j[2] for j in jobs])):
            for per, (docs, words, en_words) in res.items():
                a = agg[(name, lang, per)]
                a[0] += docs; a[1] += words; a[2] += en_words
    rows = [{"source": n, "lang": l, "form": FORM.get(n, "?"), "in_training": n in IN_TRAINING, "period": p,
             "docs": v[0], "words": v[1], "en_voted_words": v[2]} for (n, l, p), v in sorted(agg.items())]
    repo_path("logs").mkdir(exist_ok=True)
    (repo_path("logs") / "inventory.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    order = [p[0] for p in PERIODS]
    lines = ["# Corpus inventory (raw: before dedup / OCR filter / language filter)", "",
             "Words = whitespace tokens; est. tokens = words x 1.3 (EN) / 1.5 (DE) until the BPE exists.", ""]
    for in_tr in (True, False):
        sel = [r for r in rows if r["in_training"] == in_tr]
        if not sel:
            continue
        lines += [f"## {'Training sources' if in_tr else 'Downloaded but excluded from training'}", "",
                  "| lang | form | source | " + " | ".join(order) + " |", "|---|---|---|" + "---|" * len(order)]
        for (n, l) in sorted({(r["source"], r["lang"]) for r in sel}, key=lambda x: (x[1], FORM.get(x[0], ""), x[0])):
            cells = {r["period"]: r for r in sel if r["source"] == n}
            lines.append(f"| {l} | {FORM.get(n, '?')} | {n} | " + " | ".join(
                f"{cells[p]['words'] / 1e6:,.0f}M" if p in cells else "" for p in order) + " |")
        lines.append("")
    lines += ["## Totals by period x language (training sources, est. tokens)", "",
              "| period | EN words | EN est. tokens | DE words | DE est. tokens |", "|---|---|---|---|---|"]
    for p in order:
        w = {lang: sum(r["words"] for r in rows if r["in_training"] and r["lang"] == lang and r["period"] == p)
             for lang in ("en", "de")}
        lines.append(f"| {p} | {w['en'] / 1e9:.2f}B | {w['en'] * TOKENS_PER_WORD['en'] / 1e9:.2f}B | "
                     f"{w['de'] / 1e9:.2f}B | {w['de'] * TOKENS_PER_WORD['de'] / 1e9:.2f}B |")
    out = repo_path(cfg["reports"]) / "inventory.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
