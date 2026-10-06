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

word_share(text) = share of whitespace-separated tokens that look like words (letters only
once edge punctuation and inner hyphens are removed, length >= 2). The hit rate ignores digits,
symbols and one-letter fragments, so number tables and shredded OCR can still score >= 0.8;
word_share is the second gate for those. Its threshold is chosen from its own histogram, among
documents that pass the hit-rate threshold.

Nothing is dropped by this module: thresholds are chosen from the histograms and
confirmed by the user (TASKS Phase 1), then applied in the filter stage.

    python -m src.data.ocr_quality build-lexicon --lang de
    python -m src.data.ocr_quality histogram --lang de --per-cell 400   # + reports/ocr_gates_de.md
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
# Gates confirmed by the user (reports/ocr_quality_<lang>.md; HANDOFF §12, 2026-10-05/06). A document is
# kept only if it has >= MIN_DOC_TOKENS tokens, hit rate >= HIT_THRESHOLD and word share >= WORD_SHARE_THRESHOLD.
# English uses the same values unless its own histogram shows they do not fit (then both are reported).
SCORER_VERSION = "2026-10-06.1"
HIT_THRESHOLD = {"de": 0.75}
WORD_SHARE_THRESHOLD = 0.70
MIN_DOC_TOKENS = 50
# Issue-level sources (one row = a whole issue, columns run together) are cleaned per segment instead of
# being gated whole: blank-line blocks are merged into segments of >= SEGMENT_MIN_TOKENS tokens and a
# segment is dropped if its word share < WORD_SHARE_THRESHOLD (user, 2026-10-06).
SEGMENT_SOURCES = {"voelkischer_beobachter_de"}
SEGMENT_MIN_TOKENS = 30
WORD_SHARE_CANDIDATES = (0.5, 0.6, 0.7, 0.8)
_EDGE = "\"'.,;:!?()[]{}<>*„“”‚‘’»«›‹/|"
_INNER_HYPHEN = re.compile(r"[-⸗¬]")


def tokens(text: str) -> list[str]:
    return [t for t in (m.lower().replace("ſ", "s") for m in TOKEN.findall(text)) if len(t) >= 2]


def score(text: str, lexicon: set[str]) -> float:
    toks = tokens(text)
    if len(toks) < MIN_TOKENS:
        return float("nan")
    return sum(t in lexicon for t in toks) / len(toks)


def word_share(text: str) -> float:
    # Punctuation-only tokens are not counted: some OCR (DDB) writes "Müller , a . d ." with spaced
    # punctuation, which says nothing about quality. Digits, symbols and fragments do count.
    ws = [w for w in text.split() if any(ch.isalnum() for ch in w)]
    if len(ws) < MIN_TOKENS:
        return float("nan")
    words = 0
    for w in ws:
        core = _INNER_HYPHEN.sub("", w.strip(_EDGE))
        words += len(core) >= 2 and core.isalpha()
    return words / len(ws)


# ---- gates -----------------------------------------------------------------------------

def gate(text: str, lexicon: set[str], lang: str) -> tuple[str | None, float, float, int]:
    """(reason the document is dropped or None, hit rate, word share, tokens). Gates are checked in
    order short -> hit_rate -> word_share; the first that fails is the reason. NaN scores fail."""
    n = len(text.split())
    hit, ws = score(text, lexicon), word_share(text)
    if n < MIN_DOC_TOKENS:
        return "short", hit, ws, n
    if not hit >= HIT_THRESHOLD[lang]:
        return "hit_rate", hit, ws, n
    if not ws >= WORD_SHARE_THRESHOLD:
        return "word_share", hit, ws, n
    return None, hit, ws, n


_BLANK = re.compile(r"\n\s*\n")


def clean_segments(text: str) -> tuple[str, int, int]:
    """Segment cleanup for issue-level text: merge consecutive blank-line blocks until a segment has
    >= SEGMENT_MIN_TOKENS tokens (a short tail joins the last segment), drop segments whose word share
    is below WORD_SHARE_THRESHOLD (or undefined), rejoin. Returns (text, tokens in, tokens kept)."""
    segs = segments(text)
    kept = [s for s in segs if word_share(s) >= WORD_SHARE_THRESHOLD]
    n_in = sum(len(s.split()) for s in segs)
    return "\n\n".join(kept), n_in, sum(len(s.split()) for s in kept)


def segments(text: str) -> list[str]:
    segs, cur, cur_n = [], [], 0
    for block in (b.strip() for b in _BLANK.split(text)):
        if not block:
            continue
        cur.append(block)
        cur_n += len(block.split())
        if cur_n >= SEGMENT_MIN_TOKENS:
            segs.append("\n".join(cur))
            cur, cur_n = [], 0
    if cur:
        if segs:
            segs[-1] += "\n" + "\n".join(cur)
        else:
            segs.append("\n".join(cur))
    return segs


def gate_params(cfg: dict, lang: str) -> dict:
    """Everything needed to reproduce a filter decision; goes into every filtered shard's MANIFEST."""
    m = json.loads((lexicon_dir(cfg, lang) / "MANIFEST.json").read_text(encoding="utf-8"))
    return {"scorer_version": SCORER_VERSION, "lang": lang, "lexicon_file": m["file"],
            "lexicon_sha256": m["sha256"], "hit_threshold": HIT_THRESHOLD[lang],
            "word_share_threshold": WORD_SHARE_THRESHOLD, "min_doc_tokens": MIN_DOC_TOKENS,
            "segment_sources": sorted(SEGMENT_SOURCES), "segment_min_tokens": SEGMENT_MIN_TOKENS,
            "gate_order": ["short", "hit_rate", "word_share"]}


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
FIRST_YEAR, LAST_YEAR = 1900, 1955   # study range (per-year holdout spans 1900-1955)
_LEX: set[str] = set()


def _init_lex(lex: set[str]) -> None:
    _LEX.update(lex)


def _snippet(text: str, n: int = 320) -> str:
    mid = max(0, len(text) // 2 - n // 2)
    return " ".join(text[mid:mid + n].split())


def _hist_job(args: tuple) -> dict:
    src, path, per_cell, rate, lang = args
    cells = defaultdict(list)
    cols = ("article_id", "year", "text", "n_words", "lang")
    for b in pq.ParquetFile(path).iter_batches(batch_size=1000, columns=list(cols)):
        for aid, y, text, nw, row_lang in zip(*(b.column(c).to_pylist() for c in cols)):
            # only pages in the study language and range (DDB also holds French/Italian and 18th-c. pages)
            if row_lang != lang or not FIRST_YEAR <= y <= LAST_YEAR:
                continue
            if len(cells[y]) >= per_cell or not _keep(aid, rate):
                continue
            text = text or ""
            # NaN scores (too few tokens) are kept: they count as drops in the gate report
            cells[y].append((aid, score(text, _LEX), word_share(text), nw, _snippet(text)))
    return {(src, y): v for y, v in cells.items()}


def histogram(cfg: dict, lang: str, per_cell: int, rate: float, workers: int) -> dict:
    lex = load_lexicon(cfg, lang)
    sources = load_config(repo_path(cfg["sources"]))
    jobs = []
    for src in GERMAN_SOURCES:
        files = sorted(ingested_dir(cfg, src, sources[src]).glob("*.parquet"))
        n_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        src_rate = 1.0 if n_rows <= SMALL_SOURCE_ROWS else rate   # small sources: score every document
        jobs += [(src, str(f), per_cell, src_rate, lang) for f in files]
    merged: dict = defaultdict(list)
    with ProcessPoolExecutor(workers, initializer=_init_lex, initargs=(lex,)) as ex:
        for part in ex.map(_hist_job, jobs):
            for k, v in part.items():
                merged[k].extend(v)
    # cap each source x year cell deterministically (lowest article-id hash first)
    out = {}
    for (src, y), v in merged.items():
        v.sort(key=lambda t: hashlib.sha1(t[0].encode()).hexdigest())
        out[f"{src}|{y}"] = [list(t[1:]) for t in v[:per_cell]]   # [hit rate, word share, n_words, snippet]
    return out


def quantiles(xs: list[float], qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> list[float]:
    xs = sorted(xs)
    return [xs[min(len(xs) - 1, int(q * len(xs)))] for q in qs] if xs else []


def _table(by: dict, idx: int, thresholds: tuple, weighted: bool) -> list[str]:
    head = "words lost" if weighted else "docs"
    lines = ["| source | period | n | p10 | p25 | p50 | p75 | p90 | "
             + " | ".join(f"{head} < {t}" for t in thresholds) + " |",
             "|---|---|---|---|---|---|---|---|" + "---|" * len(thresholds)]
    for (src, per), rows in sorted(by.items()):
        v = [r[idx] for r in rows]
        w = [(r[2] or 0) if weighted else 1 for r in rows]
        below = [sum(wi for x, wi in zip(v, w) if x < t) / max(1, sum(w)) for t in thresholds]
        lines.append(f"| {src} | {per} | {len(v)} | " + " | ".join(f"{x:.2f}" for x in quantiles(v)) + " | "
                     + " | ".join(f"{b:.1%}" for b in below) + " |")
    return lines


PERIODS = [(1900, 1919), (1920, 1929), (1930, 1933), (1934, 1936), (1937, 1939), (1940, 1955)]


def _period(y: int) -> str | None:
    return next((f"{a}-{b}" for a, b in PERIODS if a <= y <= b), None)


def segment_report(cfg: dict, lang: str) -> tuple[dict, list]:
    """Run the segment cleanup over every document of the segment sources: tokens in/kept per
    (source, period), plus (source, word share, kept, segment text) for every segment near the cut."""
    sources = load_config(repo_path(cfg["sources"]))
    stats, boundary = defaultdict(lambda: [0, 0, 0, 0]), []
    for src in sorted(SEGMENT_SOURCES):
        for f in sorted(ingested_dir(cfg, src, sources[src]).glob("*.parquet")):
            for r in pq.read_table(f, columns=["year", "text", "lang"]).to_pylist():
                per = _period(r["year"])
                if r["lang"] != lang or per is None:
                    continue
                st = stats[(src, per)]
                for s in segments(r["text"] or ""):
                    n, ws = len(s.split()), word_share(s)
                    ok = ws >= WORD_SHARE_THRESHOLD
                    st[0] += 1; st[1] += n; st[2] += (not ok); st[3] += 0 if ok else n
                    if WORD_SHARE_THRESHOLD - 0.05 <= ws < WORD_SHARE_THRESHOLD + 0.05:
                        boundary.append((src, ws, ok, _snippet(s)))
    return dict(stats), boundary


def write_gates_report(lang: str, cells: dict, seg_stats: dict, seg_boundary: list, params: dict,
                       out_md: Path, out_boundary: Path) -> None:
    import random
    thr = WORD_SHARE_THRESHOLD
    reasons = ("short", "hit_rate", "word_share")
    agg = defaultdict(lambda: {"docs": 0, "words": 0, **{f"d_{r}": 0 for r in reasons}, **{f"w_{r}": 0 for r in reasons}})
    near_keep, near_drop = defaultdict(list), defaultdict(list)
    for k, rows in cells.items():
        src, y = k.split("|")
        per = _period(int(y))
        if src in SEGMENT_SOURCES or per is None:
            continue
        for hit, ws, nw, snip in rows:
            nw = nw or 0
            reason = ("short" if nw < MIN_DOC_TOKENS else "hit_rate" if not hit >= HIT_THRESHOLD[lang]
                      else "word_share" if not ws >= thr else None)
            a = agg[(src, per)]
            a["docs"] += 1; a["words"] += nw
            if reason:
                a[f"d_{reason}"] += 1; a[f"w_{reason}"] += nw
            if int(y) <= 1939 and thr <= ws < thr + 0.05 and reason is None:
                near_keep[src].append((hit, ws, nw, snip))
            if int(y) <= 1939 and thr - 0.05 <= ws < thr and reason == "word_share":
                near_drop[src].append((hit, ws, nw, snip))
    pct = lambda x, n: f"{x / n:.1%}" if n else "-"
    lines = [f"# OCR gates, {lang}", "", "Gate parameters (also written to every filtered shard's MANIFEST):", "",
             "```json", json.dumps(params, indent=1), "```", "",
             "## Page / article gates", "",
             "Same deterministic sample as reports/ocr_quality_*.md, including documents too short to score. "
             "Gates are applied in order; each drop is attributed to the first gate it fails.", "",
             "| source | period | docs | docs: short | docs: hit rate | docs: word share | docs: total "
             "| words: short | words: hit rate | words: word share | words: total |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (src, per), a in sorted(agg.items()):
        d_tot = sum(a[f"d_{r}"] for r in reasons); w_tot = sum(a[f"w_{r}"] for r in reasons)
        lines.append(f"| {src} | {per} | {a['docs']} | " + " | ".join(pct(a[f'd_{r}'], a['docs']) for r in reasons)
                     + f" | {pct(d_tot, a['docs'])} | " + " | ".join(pct(a[f'w_{r}'], a['words']) for r in reasons)
                     + f" | {pct(w_tot, a['words'])} |")
    lines += ["", f"## Segment cleanup (issue-level sources: {', '.join(sorted(SEGMENT_SOURCES))})", "",
              f"Every document, not a sample. Blank-line blocks merged to >= {SEGMENT_MIN_TOKENS} tokens; "
              f"segments with word share < {thr} dropped.", "",
              "| source | period | segments | segments dropped | tokens | tokens dropped |", "|---|---|---|---|---|---|"]
    for (src, per), (n_seg, n_tok, d_seg, d_tok) in sorted(seg_stats.items()):
        lines.append(f"| {src} | {per} | {n_seg} | {pct(d_seg, n_seg)} | {n_tok:,} | {pct(d_tok, n_tok)} |")
    lines += ["", f"Boundary samples ({thr - 0.05:.2f}-{thr + 0.05:.2f} word share, dated <= 1939): "
              f"logs/{out_boundary.name}.", ""]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rnd = random.Random(0)
    b = [f"# OCR gate boundary samples, {lang}", "",
         f"10 random kept ({thr:.2f}-{thr + 0.05:.2f}) and 10 random dropped ({thr - 0.05:.2f}-{thr:.2f}) per source, "
         "by word share; text from the middle of the document or segment.", ""]
    for src in sorted(set(near_keep) | set(near_drop)):
        for label, pool in (("kept", near_keep[src]), ("dropped", near_drop[src])):
            b.append(f"## {src}: {label} ({len(pool)} in band)")
            b += [f"- hit {h:.2f}, share {w:.2f}, {n} words: {s}" for h, w, n, s in rnd.sample(pool, min(10, len(pool)))]
            b.append("")
    for src in sorted({x[0] for x in seg_boundary}):
        for label, ok in (("kept", True), ("dropped", False)):
            pool = [x for x in seg_boundary if x[0] == src and x[2] == ok]
            b.append(f"## {src} segments: {label} ({len(pool)} in band)")
            b += [f"- share {w:.2f}: {s}" for _, w, _, s in rnd.sample(pool, min(10, len(pool)))]
            b.append("")
    out_boundary.write_text("\n".join(b) + "\n", encoding="utf-8")


def write_report(cfg: dict, lang: str, cells: dict, out_md: Path, out_png: Path, out_examples: Path) -> None:
    periods = PERIODS
    hit_thr = HIT_THRESHOLD[lang]
    by, passed = defaultdict(list), defaultdict(list)
    for k, rows in cells.items():
        src, y = k.split("|"); y = int(y)
        for a, b in periods:
            if a <= y <= b:
                by[(src, f"{a}-{b}")].extend(r for r in rows if not math.isnan(r[0]))
                passed[(src, f"{a}-{b}")].extend(r for r in rows if r[0] >= hit_thr and not math.isnan(r[1]))
    lines = [f"# OCR quality, {lang}", "",
             f"Sampled documents per source x year (deterministic hash sample), lang = {lang} only, "
             f"{FIRST_YEAR}-{LAST_YEAR}. Nothing has been dropped.", "",
             "## 1. Period-lexicon hit rate (all sampled documents)", "",
             "Share of documents below each threshold:", ""]
    lines += _table(by, 0, (0.6, 0.7, 0.75, 0.8), weighted=False) + ["", "Share of words below each threshold:", ""]
    lines += _table(by, 0, (0.6, 0.7, 0.75, 0.8), weighted=True) + [""]
    lines += [f"Confirmed hit-rate threshold: {hit_thr} (user, 2026-10-05).", "",
              f"## 2. Word share, among documents with hit rate >= {hit_thr}", "",
              "Share of whitespace tokens that look like words (letters only, length >= 2). "
              "Low values are number tables and shredded OCR that the hit rate misses. "
              f"Examples per band: logs/{out_examples.name}.", "", "Share of documents below each threshold:", ""]
    lines += _table(passed, 1, WORD_SHARE_CANDIDATES, weighted=False) + ["", "Share of words below each threshold:", ""]
    lines += _table(passed, 1, WORD_SHARE_CANDIDATES, weighted=True) + [""]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ex = [f"# OCR examples, {lang}: documents with hit rate >= {hit_thr}, dated {FIRST_YEAR}-1939, "
          f"by word-share band", ""]
    bands = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
    for src in sorted({k.split("|")[0] for k in cells}):
        rows = [r for k, v in sorted(cells.items()) if k.split("|")[0] == src and int(k.split("|")[1]) <= 1939
                for r in v if r[0] >= hit_thr and not math.isnan(r[1])]
        for lo, hi in bands:
            band = [r for r in rows if lo <= r[1] < hi]
            ex.append(f"## {src}, word share {lo:.1f}-{min(hi, 1.0):.1f} ({len(band)} sampled)")
            ex += [f"- hit {r[0]:.2f}, share {r[1]:.2f}, {r[2]} words: {r[3]}" for r in band[:3]] + [""]
    out_examples.write_text("\n".join(ex) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    srcs = sorted({k[0] for k in by})
    panels = [(by, 0, "lexicon hit rate (all documents)", (hit_thr,)),
              (passed, 1, f"word share (hit rate >= {hit_thr})", WORD_SHARE_CANDIDATES)]
    fig, axes = plt.subplots(2, len(srcs), figsize=(5.2 * len(srcs), 7.0), squeeze=False)
    for j, src in enumerate(srcs):
        for i, (data, idx, xlabel, marks) in enumerate(panels):
            ax = axes[i][j]
            for a, b in periods:
                v = [r[idx] for r in data.get((src, f"{a}-{b}"), [])]
                if v:
                    ax.hist(v, bins=40, range=(0, 1), histtype="step", lw=1.4, label=f"{a}-{b} (n={len(v)})")
            for t in marks:
                ax.axvline(t, color="grey", lw=0.8, ls=":")
            ax.set_title(src)
            ax.set_xlabel(xlabel)
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
        logs = repo_path("logs")
        (logs / f"ocr_quality_{args.lang}_samples.json").write_text(
            json.dumps({k: [r[:3] for r in v] for k, v in cells.items()}), encoding="utf-8")
        write_report(cfg, args.lang, cells, rep / f"ocr_quality_{args.lang}.md", rep / f"ocr_quality_{args.lang}.png",
                     logs / f"ocr_quality_{args.lang}_examples.md")
        seg_stats, seg_boundary = segment_report(cfg, args.lang)
        write_gates_report(args.lang, cells, seg_stats, seg_boundary, gate_params(cfg, args.lang),
                           rep / f"ocr_gates_{args.lang}.md", logs / f"ocr_gates_{args.lang}_boundary.md")
        print((rep / f"ocr_quality_{args.lang}.md").read_text(encoding="utf-8"))
        print((rep / f"ocr_gates_{args.lang}.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    sys.exit(main())
