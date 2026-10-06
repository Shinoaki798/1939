"""Select science/mathematics/engineering books from the existing book sets by title keywords.

Science bucket task item 15 (2026-10-06): filter pre_1929_books and loc_pd_books "by subject/title
keywords ... into the science bucket BEFORE fetching new books". Neither set carries subject headings,
so the rule is a title regex (TITLE_RE below; ambiguous words such as "science" alone, "light", "heat"
or "medical" are deliberately left out). Selected books move to the science bucket and no longer count
against the general books cap. Writes config/science_books_selection.tsv (article_id, source, year,
words, matched keyword, title) and prints counts.

    python -m src.data.science_books
"""

from __future__ import annotations

import re
from collections import Counter

import pyarrow.parquet as pq

from src.data.download import load_config, repo_path
from src.data.ingest_extra import ingested_dir

BOOK_SOURCES = ["loc_pd_books", "pre_1929_books"]
KEYWORDS = [
    # mathematics
    r"mathemat\w*", r"algebra\w*", r"geometr\w*", r"trigonometr\w*", r"calculus", r"arithmetic\w*",
    r"differential equations?", r"theory of (?:numbers|functions|groups|probability)", r"vector analysis",
    r"quaternions?", r"logarithm\w*", r"statistics", r"probabilit\w*",
    # physical sciences
    r"physics", r"physical chemistry", r"mechanics", r"dynamics", r"statics", r"thermodynamic\w*",
    r"electricity", r"electrical", r"electric", r"electro\w+", r"magnetism", r"magnetic", r"wireless", r"radio\w*",
    r"telegraph\w*", r"telephon\w*", r"optics", r"spectr\w+", r"relativity", r"quantum", r"atoms?", r"atomic",
    r"electrons?", r"radium", r"radioactiv\w*", r"x-rays?", r"rontgen", r"crystal\w*",
    r"chemistry", r"chemical", r"chemist\w*", r"metallurg\w*", r"mineralog\w*", r"astronom\w*", r"astrophysic\w*",
    r"celestial", r"planets?", r"telescope\w*", r"geolog\w*", r"seismolog\w*", r"meteorolog\w*", r"physiograph\w*",
    # life sciences
    r"biolog\w*", r"botany", r"botanical", r"zoolog\w*", r"physiolog\w*", r"anatomy", r"bacteriolog\w*",
    r"embryolog\w*", r"heredity", r"genetics", r"evolution", r"natural history", r"microscop\w*",
    # engineering and technology
    r"engineering", r"engineers?", r"machine design", r"machinery", r"steam", r"engines?", r"turbines?",
    r"hydraulic\w*", r"aeronaut\w*", r"aviation", r"aeroplanes?", r"airplanes?", r"airships?", r"dynamos?",
    r"motors?", r"internal combustion", r"locomotives?", r"bridges?", r"surveying", r"strength of materials",
]
TITLE_RE = re.compile(r"\b(" + "|".join(KEYWORDS) + r")\b", re.I)


def main() -> None:
    cfg = load_config(repo_path("config/paths.yaml"))
    sources = load_config(repo_path(cfg["sources"]))
    out, counts, words, kw = [], Counter(), Counter(), Counter()
    for src in BOOK_SOURCES:
        for f in sorted(ingested_dir(cfg, src, sources[src]).glob("*.parquet")):
            for r in pq.read_table(f, columns=["article_id", "newspaper", "year", "n_words"]).to_pylist():
                counts[(src, "all")] += 1
                m = TITLE_RE.search(r["newspaper"] or "")
                if not m:
                    continue
                k = m.group(1).lower()
                counts[(src, "science")] += 1
                words[src] += r["n_words"]
                kw[k] += 1
                out.append((r["article_id"], src, r["year"], r["n_words"], k, re.sub(r"\s+", " ", r["newspaper"]).strip()))
    with open(repo_path("config/science_books_selection.tsv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Books moved to the science bucket by title keyword (src.data.science_books; rule in its docstring)\n"
                 "article_id\tsource\tyear\twords\tkeyword\ttitle\n")
        for row in sorted(out):
            fh.write("\t".join(str(x) for x in row) + "\n")
    for src in BOOK_SOURCES:
        print(f"{src}: {counts[(src, 'science')]} of {counts[(src, 'all')]} books, {words[src] / 1e6:.0f}M words")
    print("top keywords:", kw.most_common(25))


if __name__ == "__main__":
    main()
