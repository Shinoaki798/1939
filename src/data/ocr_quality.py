"""Per-article OCR quality = period-lexicon hit rate (one lexicon per language).

German lexicon (built once, content-addressed under data/lexicon/de/):
  1. Deutsches Textarchiv (hand-keyed TEI, original orthography), texts published
     <= 1938: every word type seen at least DTA_MIN_FREQ times.
  2. Plus word types that are frequent AND widespread in the German newspaper pool
     itself (document frequency >= POOL_MIN_DF in a fixed sample, in >= POOL_MIN_TITLES
     distinct titles). DTA is mostly pre-1900, i.e. before the 1901 spelling reform
     (Thür -> Tür); the pool adds those spellings. OCR errors are idiosyncratic, so they
     rarely reach that spread.
The lexicon is a measuring instrument only; DTA is not training data here.

score(text) = share of word tokens (letters only, length >= 2, lower-cased, long s -> s)
found in the lexicon; NaN when there are fewer than MIN_TOKENS tokens.

Nothing is dropped by this module: the threshold is chosen from the histogram and
confirmed by the user (TASKS Phase 1), then applied in the filter stage.

    python -m src.data.ocr_quality build-lexicon --lang de
    python -m src.data.ocr_quality histogram --lang de --per-cell 400
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor

from src.data.download import load_config, repo_path
from src.data.ingest_extra import ingested_dir

TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)
MIN_TOKENS = 20
DTA_MIN_FREQ = 2
POOL_MIN_DF = 50
POOL_MIN_TITLES = 5
POOL_SAMPLE_PER_FILE = 1500
GERMAN_SOURCES = ["ddb_newspapers_de", "europeana_newspapers_de", "voelkischer_beobachter_de"]


def tokens(text: str) -> list[str]:
    return [t for t in (m.lower().replace("ſ", "s") for m in TOKEN.findall(text)) if len(t) >= 2]


def score(text: str, lexicon: set[str]) -> float:
    toks = tokens(text)
    if len(toks) < MIN_TOKENS:
        return float("nan")
    return sum(t in lexicon for t in toks) / len(toks)


# ---- lexicon ---------------------------------------------------------------------------

# The first <date type="publication"> in a DTA header is the digital edition (e.g. 2025); the
# original work's date sits inside <sourceDesc> (publication, or creation for manuscripts).
_DTA_SOURCE = re.compile(r"<sourceDesc>(.*?)</sourceDesc>", re.S)
_DTA_YEAR = re.compile(r'<date type="(?:publication|creation)">\s*(\d{4})')
_FW = re.compile(r"<fw\b[^>]*>.*?</fw>", re.S)            # running heads, page numbers, catchwords
_HYPH = re.compile(r"[-¬]\s*<lb\s*/>\s*")                  # line-end hyphenation
_TAG = re.compile(r"<[^>]+>")


def dta_text(tei: str) -> tuple[int | None, str]:
    header = tei.split("</teiHeader>", 1)[0]
    src = _DTA_SOURCE.search(header)
    m = _DTA_YEAR.search(src.group(1)) if src else None
    m = m or re.search(r'<date type="creation">\s*(\d{4})', header)
    year = int(m.group(1)) if m else None
    body = tei.split("</teiHeader>", 1)[-1].split("<text", 1)[-1]   # not the header's <textClass>
    body = _FW.sub(" ", body)
    body = _HYPH.sub("", body)
    return year, html.unescape(_TAG.sub(" ", body))


def _dta_job(args: tuple) -> tuple[Counter, int, int]:
    zpath, names = args
    c, used, late = Counter(), 0, 0
    with zipfile.ZipFile(zpath) as z:
        for name in names:
            year, text = dta_text(z.read(name).decode("utf-8", "replace"))
            if year is None or year > 1938:
                late += 1
                continue
            used += 1
            c.update(tokens(text))
    return c, used, late


def _pool_job(path: str) -> tuple[Counter, dict, int]:
    df, titles, n = Counter(), defaultdict(set), 0
    for b in pq.ParquetFile(path).iter_batches(batch_size=500, columns=["newspaper", "text"]):
        for paper, text in zip(b.column("newspaper").to_pylist(), b.column("text").to_pylist()):
            for w in set(tokens(text or "")):
                df[w] += 1
                if len(titles[w]) < POOL_MIN_TITLES:
                    titles[w].add(paper)
            n += 1
            if n >= POOL_SAMPLE_PER_FILE:
                return df, {w: sorted(t) for w, t in titles.items()}, n
    return df, {w: sorted(t) for w, t in titles.items()}, n


def build_lexicon_de(cfg: dict, workers: int) -> tuple[set[str], dict]:
    sources = load_config(repo_path(cfg["sources"]))
    zpath = next(repo_path(sources["dta"]["dest"]).glob("*.zip"))
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.endswith(".xml")]
    chunks = [(str(zpath), names[i::workers]) for i in range(workers)]
    dta, n_docs, n_late = Counter(), 0, 0
    with ProcessPoolExecutor(workers) as ex:
        for c, used, late in ex.map(_dta_job, chunks):
            dta.update(c); n_docs += used; n_late += late
    lex_dta = {w for w, c in dta.items() if c >= DTA_MIN_FREQ}

    files = [str(f) for src in GERMAN_SOURCES
             for f in sorted(ingested_dir(cfg, src, sources[src]).glob("*.parquet"))]
    df, titles, n_pool = Counter(), defaultdict(set), 0
    with ProcessPoolExecutor(workers) as ex:
        for c, t, n in ex.map(_pool_job, files):
            df.update(c); n_pool += n
            for w, ts in t.items():
                if len(titles[w]) < POOL_MIN_TITLES:
                    titles[w].update(ts)
    lex_pool = {w for w, c in df.items() if c >= POOL_MIN_DF and len(titles[w]) >= POOL_MIN_TITLES}
    lex = lex_dta | lex_pool
    info = {"dta_zip": zpath.name, "dta_docs_used": n_docs, "dta_docs_skipped_after_1938_or_undated": n_late,
            "dta_types": len(lex_dta), "pool_files": len(files), "pool_docs_sampled": n_pool,
            "pool_types": len(lex_pool), "pool_only_types": len(lex_pool - lex_dta), "total_types": len(lex),
            "params": {"DTA_MIN_FREQ": DTA_MIN_FREQ, "POOL_MIN_DF": POOL_MIN_DF,
                       "POOL_MIN_TITLES": POOL_MIN_TITLES, "POOL_SAMPLE_PER_FILE": POOL_SAMPLE_PER_FILE}}
    return lex, info


def lexicon_dir(cfg: dict, lang: str) -> Path:
    return repo_path(cfg["data_root"]) / "lexicon" / lang


def save_lexicon(cfg: dict, lang: str, lex: set[str], info: dict) -> Path:
    d = lexicon_dir(cfg, lang)
    d.mkdir(parents=True, exist_ok=True)
    data = ("\n".join(sorted(lex)) + "\n").encode("utf-8")
    sha = hashlib.sha256(data).hexdigest()
    path = d / f"{sha[:12]}.txt"
    path.write_bytes(data)
    info.update({"file": path.name, "sha256": sha, "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    (d / "MANIFEST.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_lexicon(cfg: dict, lang: str) -> set[str]:
    m = json.loads((lexicon_dir(cfg, lang) / "MANIFEST.json").read_text(encoding="utf-8"))
    return set((lexicon_dir(cfg, lang) / m["file"]).read_text(encoding="utf-8").split())


# ---- histogram -------------------------------------------------------------------------

def _keep(article_id: str, per_cell_rate: float) -> bool:
    """Deterministic sample: hash of the article id, so reruns pick the same rows."""
    h = int(hashlib.sha1(article_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < per_cell_rate


SMALL_SOURCE_ROWS = 50_000
_LEX: set[str] = set()


def _init_lex(lex: set[str]) -> None:
    _LEX.update(lex)


def _hist_job(args: tuple) -> dict:
    src, path, per_cell, rate = args
    cells = defaultdict(list)
    for b in pq.ParquetFile(path).iter_batches(batch_size=1000, columns=["article_id", "year", "text"]):
        for aid, y, text in zip(*(b.column(c).to_pylist() for c in ("article_id", "year", "text"))):
            if len(cells[y]) >= per_cell or not _keep(aid, rate):
                continue
            s = score(text or "", _LEX)
            if not math.isnan(s):
                cells[y].append((aid, s))
    return {(src, y): v for y, v in cells.items()}


def histogram(cfg: dict, lang: str, per_cell: int, rate: float, workers: int) -> dict:
    lex = load_lexicon(cfg, lang)
    sources = load_config(repo_path(cfg["sources"]))
    jobs = []
    for src in GERMAN_SOURCES:
        files = sorted(ingested_dir(cfg, src, sources[src]).glob("*.parquet"))
        n_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        src_rate = 1.0 if n_rows <= SMALL_SOURCE_ROWS else rate   # small sources: score every document
        jobs += [(src, str(f), per_cell, src_rate) for f in files]
    merged: dict = defaultdict(list)
    with ProcessPoolExecutor(workers, initializer=_init_lex, initargs=(lex,)) as ex:
        for part in ex.map(_hist_job, jobs):
            for k, v in part.items():
                merged[k].extend(v)
    # cap each source x year cell deterministically (lowest article-id hash first)
    out = {}
    for (src, y), v in merged.items():
        v.sort(key=lambda t: hashlib.sha1(t[0].encode()).hexdigest())
        out[f"{src}|{y}"] = [s for _, s in v[:per_cell]]
    return out


def quantiles(xs: list[float], qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> list[float]:
    xs = sorted(xs)
    return [xs[min(len(xs) - 1, int(q * len(xs)))] for q in qs] if xs else []


def write_report(cfg: dict, lang: str, cells: dict, out_md: Path, out_png: Path) -> None:
    periods = [(1900, 1919), (1920, 1929), (1930, 1933), (1934, 1936), (1937, 1939), (1940, 1955)]
    thresholds = (0.5, 0.6, 0.7, 0.8)
    by = defaultdict(list)
    for k, v in cells.items():
        src, y = k.split("|"); y = int(y)
        for a, b in periods:
            if a <= y <= b:
                by[(src, f"{a}-{b}")].extend(v)
    lines = [f"# OCR quality (period-lexicon hit rate), {lang}", "",
             "Sampled documents per source x year (deterministic hash sample). Nothing has been dropped.", "",
             "| source | period | n | p10 | p25 | p50 | p75 | p90 | " + " | ".join(f"< {t}" for t in thresholds) + " |",
             "|---|---|---|---|---|---|---|---|" + "---|" * len(thresholds)]
    for (src, per), v in sorted(by.items()):
        q = quantiles(v)
        below = [sum(x < t for x in v) / len(v) for t in thresholds]
        lines.append(f"| {src} | {per} | {len(v)} | " + " | ".join(f"{x:.2f}" for x in q) + " | "
                     + " | ".join(f"{b:.0%}" for b in below) + " |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    srcs = sorted({k[0] for k in by})
    fig, axes = plt.subplots(1, len(srcs), figsize=(5.2 * len(srcs), 3.6), squeeze=False)
    for ax, src in zip(axes[0], srcs):
        for a, b in periods:
            v = by.get((src, f"{a}-{b}"))
            if v:
                ax.hist(v, bins=40, range=(0, 1), histtype="step", lw=1.4, label=f"{a}-{b} (n={len(v)})")
        for t in thresholds:
            ax.axvline(t, color="grey", lw=0.6, ls=":")
        ax.set_title(src)
        ax.set_xlabel("lexicon hit rate")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build-lexicon", "histogram"])
    ap.add_argument("--lang", default="de", choices=["de"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--per-cell", type=int, default=400, help="max sampled docs per source x year")
    ap.add_argument("--rate", type=float, default=0.02, help="hash-sample rate before the per-cell cap")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(Path(args.config) if args.config else repo_path("config/paths.yaml"))
    if args.command == "build-lexicon":
        lex, info = build_lexicon_de(cfg, args.workers)
        path = save_lexicon(cfg, args.lang, lex, info)
        print(json.dumps(info, indent=1, ensure_ascii=False), "\n->", path)
    else:
        cells = histogram(cfg, args.lang, args.per_cell, args.rate, args.workers)
        rep = repo_path(cfg["reports"])
        rep.mkdir(parents=True, exist_ok=True)
        (repo_path("logs") / f"ocr_quality_{args.lang}_samples.json").write_text(json.dumps(cells), encoding="utf-8")
        write_report(cfg, args.lang, cells, rep / f"ocr_quality_{args.lang}.md", rep / f"ocr_quality_{args.lang}.png")
        print((rep / f"ocr_quality_{args.lang}.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    sys.exit(main())
