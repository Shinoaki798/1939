"""Select Project Gutenberg science/mathematics/engineering books (science bucket, 2026-10-06).

Input: PG's weekly catalog (data/raw/gutenberg/pg_catalog.csv.gz; columns Text#, Type, Issued, Title,
Language, Authors, Subjects, LoCC, Bookshelves). Selection:
  * Type Text, Language en or de;
  * LoC class Q* (science) or one of the engineering/technology classes in T_KEEP, or an
    Encyclopaedia Britannica 11th edition slice (keyed EB11, vols 2-17);
  * date screen (PG has no original publication date): drop if any listed person was born >= 1915;
    "safe" if every listed person died <= 1930; otherwise "check": the ingest adapter keeps the book
    only if its front matter names a year 1800-1938 and none >= 1939.
Writes config/gutenberg_sci_{en,de}_files.tsv (download table, PG mirror URLs) and _items.tsv.

    python -m src.data.gutenberg_select
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import io
import re

from src.data.download import load_config, repo_path

T_KEEP = ("TA", "TC", "TD", "TF", "TG", "TJ", "TK", "TL", "TN", "TP")
MIRROR = "https://gutenberg.pglaf.org/cache/epub/{n}/pg{n}.txt"
_PERSON = re.compile(r"(\d{3,4})\??\s*-\s*(\d{3,4})?")


def is_science(locc: str, title: str) -> str | None:
    classes = [c.strip() for c in locc.split(";") if c.strip()]
    if title.startswith("Encyclopaedia Britannica, 11th Edition"):
        return "eb11"
    if any(c.startswith("Q") for c in classes) or any(c[:2] in T_KEEP for c in classes):
        return "science"
    return None


def date_verdict(authors: str) -> str:
    people = [(_PERSON.search(p).groups() if _PERSON.search(p) else (None, None)) for p in authors.split(";")]
    births = [int(b) for b, _ in people if b]
    deaths = [int(d) if d else None for _, d in people]
    if any(b >= 1915 for b in births):
        return "drop"
    if people and all(d is not None and d <= 1930 for d in deaths):
        return "safe"
    return "check"


def main() -> None:
    cfg = load_config(repo_path("config/paths.yaml"))
    sources = load_config(repo_path(cfg["sources"]))
    cat = next(repo_path(sources["gutenberg_sci_en"]["catalog_dest"]).glob("pg_catalog*.csv.gz"))
    rows = list(csv.DictReader(io.TextIOWrapper(gzip.open(cat, "rb"), encoding="utf-8")))
    for lang in ("en", "de"):
        name = f"gutenberg_sci_{lang}"
        kept = []
        for r in rows:
            kind = is_science(r["LoCC"], r["Title"]) if r["Type"] == "Text" and r["Language"] == lang else None
            if not kind:
                continue
            verdict = date_verdict(r["Authors"]) if kind == "science" else "safe"   # EB11 is 1910-11
            if verdict != "drop":
                kept.append((r, kind, verdict))
        src = sources[name]
        with open(repo_path(src["files"]), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"# {name}: Project Gutenberg plain text from the PG mirror; selected from {cat.name} "
                     f"on {dt.date.today()} by src.data.gutenberg_select\nkey\turl\tbytes\tchecksum\n")
            for r, *_ in kept:
                fh.write(f"pg{r['Text#']}\t{MIRROR.format(n=r['Text#'])}\t?\t-\n")
        with open(repo_path(src["items"]), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("key\tkind\tverdict\tlocc\ttitle\tauthors\n")
            for r, kind, verdict in kept:
                clean = lambda s: re.sub(r"\s+", " ", s).strip()
                fh.write("\t".join([f"pg{r['Text#']}", kind, verdict, clean(r["LoCC"]), clean(r["Title"]),
                                    clean(r["Authors"])]) + "\n")
        by = {}
        for _, kind, verdict in kept:
            by[(kind, verdict)] = by.get((kind, verdict), 0) + 1
        print(name, len(kept), by, flush=True)


if __name__ == "__main__":
    main()
