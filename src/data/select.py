"""Selection: which ingested documents enter the cleaning pool, and in which language pool.

Pass 1 (`census`) reads only metadata columns of every pool source and records, per language, the
words of general text by year (for the pre-1920 sampling rate) and the American Stories title-level
language statistics. Pass 2 (`run`) applies the rules below to every ingested parquet file and writes
one content-addressed parquet per input file and language:

    data/selected/<source>/<lang>/<sha256[:12]>.parquet   + data/selected/<source>/MANIFEST.json

Rules (HANDOFF §5, §12 2026-10-05 ... 2026-10-07):

  pool      every training source except those out by decision (HMD, NCSE); JSTOR EJC keeps only the
            titles in config/science_jstor_titles.txt. Lexicon anchors are never ingested.
  bucket    science = sources with `bucket: science`, the EJC science titles and the books listed in
            config/science_books_selection.tsv; everything else is general.
  language  a document goes to the pool of its detected language if that is en or de. English also
            needs filters.is_native_english; American Stories also the title rule (filters.english_titles
            per lccn and year, from the census). "und" (too few stopwords to tell) is dropped, except in
            keyed sources, where it takes the source's language (user, 2026-10-07: restores 9,668 short
            JFM reviews). Other languages are dropped and counted.
  dates     pre (<= 1939-06-30), embargo (1939-07-01 .. 08-31: RQ3 conditioning only), post (1939-09-01 ..
            1955-12-31: test sets only). General text before 1900 is not in the pool (its recency weight
            is ~0); the science bucket keeps any year up to the cutoff.
  pre-1920  general documents dated 1900-1919 are kept with probability p(lang, year) = min(1, target /
            census words): the per-language target PRE1920_WORDS is spread over the years by the recency
            weight exp(-(1939 - year) / 5), with a floor of PRE1920_FLOOR words per year. The draw is a
            fixed hash of the article id, so a rerun keeps the same documents.
  text      Unicode NFC; German also long s -> s and a/o/u + combining small e (U+0364) -> ä/ö/ü.

PRE1920_WORDS: pre-1920 may be at most 8 % of seen tokens (CLAUDE.md rule 7); at 10B seen tokens with
English ~75 % that is ~0.6B English and ~0.2B German tokens, i.e. ~0.45B and ~0.13B words; the pool
keeps about four times that (user, 2026-10-07).

    python -m src.data.select census
    python -m src.data.select run [--sources a,b] [--dry-run] [--workers 8]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.download import load_config, repo_path
from src.data.filters import MIN_TITLE_EN_FRACTION, is_native_english
from src.data.ingest_extra import ingested_dir, sha256_file

EXCLUDED = {"hmd_newspapers", "ncse"}                    # out by user decision (HANDOFF §12, 2026-10-05)
LANGS = ("en", "de")
CATEGORY = {"american_stories": "newspaper", "chronicling_america": "newspaper",
            "ddb_newspapers_de": "newspaper", "europeana_newspapers_de": "newspaper",
            "voelkischer_beobachter_de": "newspaper", "congressional_record": "legislative",
            "caselaw_access_project": "legal", "federal_register": "legal",
            "loc_pd_books": "books", "pre_1929_books": "books"}
PAGE_LEVEL = {"chronicling_america", "ddb_newspapers_de", "europeana_newspapers_de", "federal_register"}
CUTOFF, EMBARGO_END, LAST_DAY = "1939-06-30", "1939-08-31", "1955-12-31"
PRE1920_WORDS = {"en": 1.8e9, "de": 0.55e9}
PRE1920_FLOOR = 20e6
RECENCY_YEARS = 5.0

OUT_SCHEMA = pa.schema([
    ("article_id", pa.string()), ("source", pa.string()), ("lang", pa.string()),
    ("bucket", pa.string()), ("category", pa.string()), ("keyed", pa.bool_()), ("page_level", pa.bool_()),
    ("date", pa.string()), ("year", pa.int16()), ("date_class", pa.string()),
    ("newspaper", pa.string()), ("lccn", pa.string()), ("headline", pa.string()),
    ("text", pa.string()), ("n_bytes", pa.int32()), ("n_words", pa.int32()),
])
META_COLS = ["article_id", "year", "date", "lang", "lang_en_share", "n_words", "lccn", "newspaper"]

_COMB_E = {"a": "ä", "o": "ö", "u": "ü", "A": "Ä", "O": "Ö", "U": "Ü"}
_COMB_E_RE = re.compile("([aouAOU])ͤ")


def normalize(text: str, lang: str) -> str:
    text = unicodedata.normalize("NFC", text)
    if lang == "de":
        text = _COMB_E_RE.sub(lambda m: _COMB_E[m.group(1)], text).replace("ſ", "s")
    return text


def unit_hash(article_id: str, salt: str) -> float:
    """Deterministic uniform [0, 1) draw per document."""
    h = hashlib.blake2b(f"{salt}:{article_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2 ** 64


def date_class(date: str, science: bool) -> str | None:
    if date > LAST_DAY:
        return None
    if date > EMBARGO_END:
        return "post"
    if date > CUTOFF:
        return "embargo"
    if date < "1900-01-01" and not science:
        return None
    return "pre"


def pool_sources(cfg: dict) -> dict:
    """Pool sources with the facts the rules need."""
    sources = load_config(repo_path(cfg["sources"]))
    out = {}
    for name, src in sources.items():
        if not isinstance(src, dict) or name in EXCLUDED or "anchor" in str(src.get("role", "")):
            continue
        ing = ingested_dir(cfg, name, src)
        if not (ing / "MANIFEST.json").exists():
            continue
        out[name] = {"dir": str(ing), "science": src.get("bucket") == "science",
                     "keyed": str(src.get("ocr_or_keyed", "")).startswith("keyed"), "lang": src.get("lang", "en"),
                     "category": "science" if src.get("bucket") == "science" else CATEGORY.get(name, "other")}
    return out


def ingested_files(ing_dir: Path) -> list[tuple[str, str]]:
    """(file name, sha256) of every parquet listed in an ingested MANIFEST (American Stories lists years)."""
    m = json.loads((ing_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    entries = m.get("files") or m.get("years") or {}
    seen, out = set(), []
    for e in entries.values():
        if e.get("file") and e["file"] not in seen and (ing_dir / e["file"]).exists():
            seen.add(e["file"])
            out.append((e["file"], e.get("sha256", "")))
    return sorted(out)


def science_lists() -> tuple[set[str], set[str]]:
    titles = {l.strip() for l in open(repo_path("config/science_jstor_titles.txt"), encoding="utf-8")
              if l.strip() and not l.startswith("#")}
    with open(repo_path("config/science_books_selection.tsv"), encoding="utf-8") as f:
        rows = csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t")
        books = {r["article_id"] for r in rows}
    return titles, books


def doc_lang(source: str, lang: str, en_share, as_titles: dict | None, lccn: str, year: int,
             und_default: str | None = None) -> str | None:
    """Pool language of a document; und_default is the source language of a keyed source."""
    if lang == "und" and und_default in LANGS:
        return und_default
    if lang == "en":
        if not is_native_english(lang, en_share):
            return None
        if source == "american_stories" and as_titles is not None and lccn not in as_titles.get(str(year), ()):
            return None
        return "en"
    return "de" if lang == "de" else None


# ---------------------------------------------------------------- census

def _census_job(job: tuple) -> dict:
    source, path, science, sci_books = job
    words: Counter = Counter()
    titles: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    pf = pq.ParquetFile(path)
    cols = [c for c in META_COLS if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=200_000, columns=cols):
        b = batch.to_pydict()
        for i in range(batch.num_rows):
            year, lang = b["year"][i], b["lang"][i]
            if source == "american_stories":
                t = titles[(b["lccn"][i] or "", year)]
                t[0] += 1
                t[1] += lang == "en"
            if science or source == "jstor_ejc" or not 1900 <= year <= 1919 or b["article_id"][i] in sci_books:
                continue
            pool = "en" if is_native_english(lang, b["lang_en_share"][i]) else ("de" if lang == "de" else None)
            if pool:
                words[f"{pool}:{year}"] += b["n_words"][i]
    return {"words": dict(words), "titles": {f"{k[0]}|{k[1]}": v for k, v in titles.items()}}


def census(cfg: dict, workers: int) -> dict:
    pool = pool_sources(cfg)
    _, sci_books = science_lists()
    jobs = [(name, str(Path(p["dir"]) / f), p["science"], sci_books if name in CATEGORY else set())
            for name, p in pool.items() for f, _ in ingested_files(Path(p["dir"]))]
    words: Counter = Counter()
    titles: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs) or 1)) as ex:
        for fut in as_completed([ex.submit(_census_job, j) for j in jobs]):
            r = fut.result()
            words.update(r["words"])
            for k, (n, e) in r["titles"].items():
                titles[k][0] += n
                titles[k][1] += e
    as_titles: dict[str, list[str]] = defaultdict(list)
    for k, (n, e) in titles.items():
        lccn, year = k.rsplit("|", 1)
        if e / n >= MIN_TITLE_EN_FRACTION:
            as_titles[year].append(lccn)
    rates = {}
    for lang in LANGS:
        w = {y: math.exp(-(1939 - y) / RECENCY_YEARS) for y in range(1900, 1920)}
        total_w = sum(w.values())
        for y in range(1900, 1920):
            have = words.get(f"{lang}:{y}", 0)
            target = max(PRE1920_FLOOR, PRE1920_WORDS[lang] * w[y] / total_w)
            rates[f"{lang}:{y}"] = min(1.0, target / have) if have else 1.0
    out = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "sources": sorted(pool), "pre1920_words": dict(sorted(words.items())), "pre1920_rate": rates,
           "params": {"PRE1920_WORDS": PRE1920_WORDS, "PRE1920_FLOOR": PRE1920_FLOOR,
                      "RECENCY_YEARS": RECENCY_YEARS, "MIN_TITLE_EN_FRACTION": MIN_TITLE_EN_FRACTION},
           "as_titles": {y: sorted(v) for y, v in sorted(as_titles.items())}}
    return out


def census_path(cfg: dict) -> Path:
    return repo_path(cfg["selected"]) / "census.json"


# ---------------------------------------------------------------- run

def _select_job(job: tuple) -> dict:
    source, path, out_root, info, rates, as_titles, ejc_titles, sci_books, dry_run = job
    stats: Counter = Counter()
    rows: dict[str, list[dict]] = {lang: [] for lang in LANGS}
    pf = pq.ParquetFile(path)
    cols = [c for c in META_COLS + ["text", "headline"] if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=20_000, columns=cols):
        b = batch.to_pydict()
        for i in range(batch.num_rows):
            stats["rows_in"] += 1
            aid, date, year = b["article_id"][i], b["date"][i] or "", b["year"][i]
            science = info["science"] or aid in sci_books
            if source == "jstor_ejc":
                if (b["newspaper"][i] or "") not in ejc_titles:
                    stats["drop_ejc_not_science"] += 1
                    continue
                science = True
            dc = date_class(date, science)
            if dc is None:
                stats["drop_date"] += 1
                continue
            lang = doc_lang(source, b["lang"][i], b["lang_en_share"][i], as_titles,
                            (b.get("lccn") or [""] * batch.num_rows)[i] or "", year,
                            info["lang"] if info["keyed"] else None)
            if lang is None:
                stats[f"drop_lang_{b['lang'][i]}"] += 1
                continue
            if not science and dc == "pre" and 1900 <= year <= 1919:
                if unit_hash(aid, "pre1920") >= rates.get(f"{lang}:{year}", 1.0):
                    stats["drop_pre1920_sample"] += 1
                    continue
            text = normalize(b["text"][i] or "", lang)
            stats[f"keep_{lang}_{dc}"] += 1
            stats[f"words_{lang}_{dc}"] += len(text.split())
            if dry_run:
                continue
            rows[lang].append({
                "article_id": aid, "source": source, "lang": lang, "bucket": "science" if science else "general",
                "category": "science" if science else info["category"], "keyed": info["keyed"],
                "page_level": source in PAGE_LEVEL, "date": date, "year": year, "date_class": dc,
                "newspaper": b["newspaper"][i] if "newspaper" in b else "",
                "lccn": (b.get("lccn") or [""] * batch.num_rows)[i] or "",
                "headline": (b.get("headline") or [""] * batch.num_rows)[i] or "",
                "text": text, "n_bytes": len(text.encode("utf-8")), "n_words": len(text.split())})
    outputs = {}
    for lang, rs in rows.items():
        if not rs:
            continue
        d = Path(out_root) / source / lang
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".tmp_{Path(path).stem}.parquet"
        pq.write_table(pa.Table.from_pylist(rs, schema=OUT_SCHEMA), tmp, compression="zstd")
        digest = sha256_file(tmp)
        final = d / f"{digest[:12]}.parquet"
        if final.exists():
            os.remove(tmp)
        else:
            os.replace(tmp, final)
        outputs[lang] = {"file": f"{lang}/{final.name}", "sha256": digest, "rows": len(rs),
                         "words": sum(r["n_words"] for r in rs)}
    return {"input": Path(path).name, "stats": dict(stats), "outputs": outputs}


def run(cfg: dict, only: list[str], workers: int, dry_run: bool, force: bool = False) -> None:
    cpath = census_path(cfg)
    if not cpath.exists():
        sys.exit(f"{cpath} missing: run `python -m src.data.select census` first")
    cen = json.loads(cpath.read_text(encoding="utf-8"))
    as_titles = {y: set(v) for y, v in cen["as_titles"].items()} or None
    ejc_titles, sci_books = science_lists()
    pool = pool_sources(cfg)
    out_root = repo_path(cfg["selected"])
    for source in (only or sorted(pool)):
        info = pool[source]
        ing = Path(info["dir"])
        mpath = out_root / source / "MANIFEST.json"
        manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {
            "stage": "selected", "source": source, "files": {}}
        manifest["params"] = {"census": cen["created_at"], "pre1920_rate": {k: v for k, v in cen["pre1920_rate"].items()},
                              "science": info["science"], "category": info["category"], "keyed": info["keyed"]}
        files = [(f, sha) for f, sha in ingested_files(ing)
                 if dry_run or force or manifest["files"].get(f, {}).get("input_sha256") != sha]
        if not files:
            print(f"{source}: nothing to do", flush=True)
            continue
        t0, totals = time.time(), Counter()
        jobs = [(source, str(ing / f), str(out_root), info, cen["pre1920_rate"],
                 as_titles if source == "american_stories" else None, ejc_titles, sci_books, dry_run)
                for f, _ in files]
        shas = dict(files)
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            for fut in as_completed([ex.submit(_select_job, j) for j in jobs]):
                r = fut.result()
                totals.update(r["stats"])
                if not dry_run:
                    manifest["files"][r["input"]] = {"input_sha256": shas[r["input"]], "outputs": r["outputs"],
                                                     "stats": r["stats"],
                                                     "selected_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        if not dry_run:
            mpath.parent.mkdir(parents=True, exist_ok=True)
            tmp = mpath.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp, mpath)
        keep = {k: v for k, v in totals.items() if k.startswith(("keep_", "words_"))}
        drop = {k: v for k, v in totals.items() if k.startswith("drop_")}
        print(f"{source}: {len(files)} files, rows_in={totals['rows_in']:,} keep={keep} drop={drop} "
              f"{time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["census", "run"])
    ap.add_argument("--sources", default="", help="comma-separated subset (run only)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="count only; write nothing")
    ap.add_argument("--force", action="store_true", help="redo files already selected (after a rule change)")
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    if args.command == "census":
        out = census(cfg, args.workers)
        p = census_path(cfg)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"census: {len(out['sources'])} sources; pre-1920 general words "
              f"{ {k: v for k, v in out['pre1920_words'].items()} } -> {p}", flush=True)
    else:
        run(cfg, [s for s in args.sources.split(",") if s], args.workers, args.dry_run, args.force)


if __name__ == "__main__":
    main()
