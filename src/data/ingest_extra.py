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
import functools
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
SOURCE_META_FIELDS = ("title", "bucket", "lang", "licence", "access_method", "ocr_or_keyed", "subject_filter_rule",
                      "role", "approved")
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


def rows_jstor_ejc(path: Path, key: str, cutoff: dt.date, stats: Counter):
    """JSTOR Early Journal Content bundle (one XML per article, members bundle/10.2307_<id>, no
    suffix): one row per article, OCR text of all pages. Every journal is ingested; the science
    bucket keeps the STEM titles listed in config/science_jstor_titles.txt."""
    import tarfile
    import xml.etree.ElementTree as ET
    if not path.name.endswith(".tar.bz2"):          # readme.txt
        return
    with tarfile.open(path, "r|bz2") as tf:
        for m in tf:
            if not m.isfile() or "10.2307_" not in m.name.rsplit("/", 1)[-1]:
                continue
            stats["rows_in"] += 1
            try:
                a = ET.fromstring(tf.extractfile(m).read())
            except ET.ParseError:
                stats["dropped_bad_xml"] += 1
                continue
            g = lambda tag: (a.findtext(tag) or "").strip()
            precision = "day"
            try:
                date = dt.date.fromisoformat(g("pubdate")[:10])
            except ValueError:
                try:
                    date, precision = dt.date(int(g("year")), 1, 1), "year"
                except ValueError:
                    stats["dropped_bad_date"] += 1
                    continue
            if date > cutoff or (precision == "year" and date.year > cutoff.year - 1):
                stats["dropped_after_cutoff"] += 1
                continue
            text = "\n\n".join((p.text or "").strip() for p in a.findall("pages/list-item")).strip()
            if not text:
                stats["dropped_empty"] += 1
                continue
            langs = [(x.text or "").strip() for x in a.findall("languages/list-item")] or [g("languages")]
            meta = {"jstor_id": g("id"), "journal_id": g("journalid"), "journal_abbrv": g("journalabbrv"),
                    "type": g("type"), "volume": g("volume"), "pagerange": g("pagerange"), "issn": g("issn"),
                    "languages": [x for x in langs if x], "date_precision": precision}
            authors = "; ".join((x.text or "").strip() for x in a.findall("authors/list-item"))
            row = _row(f"ejc_{g('id').replace('/', '_')}", "jstor_ejc", date, g("journaltitle"), "", authors, text, meta)
            row["headline"] = g("title")
            yield row


def rows_royal_society_corpus(path: Path, key: str, cutoff: dt.date, stats: Counter):
    """Royal Society Corpus 6.0.4 open (Philosophical Transactions / Proceedings 1665-1920): one row
    per paper from the plain-text zip, joined to the metadata TSV in the same directory. Year-only."""
    import zipfile
    if "texts_txt" not in path.name:                 # the meta zip is read alongside the texts
        return
    meta_zip = next(path.parent.glob("*_meta.tsv.zip"))
    with zipfile.ZipFile(meta_zip) as z:
        lines = z.read(z.namelist()[0]).decode("utf-8", "replace").splitlines()
    header = lines[0].split("\t")
    meta = {f[0]: dict(zip(header, f)) for f in (l.split("\t") for l in lines[1:]) if len(f) == len(header)}
    with zipfile.ZipFile(path) as z:
        for n in sorted(z.namelist()):
            if not n.endswith(".txt"):
                continue
            stats["rows_in"] += 1
            # files are .../Royal_Society_Corpus_open_v6.0_text_<id>.txt; <id> is the meta TSV's id column
            rid = n.rsplit("/", 1)[-1][:-len(".txt")].rsplit("_text_", 1)[-1]
            md = meta.get(rid)
            if md is None:
                stats["dropped_no_meta"] += 1
                continue
            try:
                year = int(md["year"])
            except ValueError:
                stats["dropped_bad_date"] += 1
                continue
            if year > cutoff.year - 1:                  # year-only items: year <= 1938
                stats["dropped_after_cutoff"] += 1
                continue
            text = z.read(n).decode("utf-8", "replace").strip()
            if not text:
                stats["dropped_empty"] += 1
                continue
            row = _row(f"rsc_{rid}", "royal_society_corpus", dt.date(year, 1, 1), md.get("journal", ""), "",
                       md.get("author", ""), text,
                       {"rsc_id": rid, "doi": md.get("doi"), "volume": md.get("volume"), "type": md.get("type"),
                        "language": md.get("language"), "primary_topic": md.get("primaryTopic"),
                        "date_precision": "year"})
            row["headline"] = md.get("title", "")
            yield row


JFM_MAX_VOLUME = 61   # JFM volumes > 61 appeared after 1939-06-30 or on unknown dates (user, 2026-10-06)
JFM_PLACEHOLDER = "contents unavailable due to conflicting licenses"
_JFM_RECORD = re.compile(r"<record>(.*?)</record>", re.S)
_JFM_XREF = re.compile(r"\(\s*(?:JFM|Zbl)\s*[\d.*]+\s*\)|\b(?:JFM|Zbl)\s+[\d*]+\.[\d*]+\.[\d*]+")


def _zb(rec: str, tag: str) -> str:
    import html
    m = re.search(rf"<zbmath:{tag}>(.*?)</zbmath:{tag}>", rec, re.S)
    return html.unescape(m.group(1).strip()) if m else ""


def rows_jfm(path: Path, key: str, cutoff: dt.date, stats: Counter):
    """One zbMATH Open OAI page (gzip) of the Jahrbuch ueber die Fortschritte der Mathematik: one row per
    review in JFM volumes <= JFM_MAX_VOLUME. Keyed text (ERAM); date = reviewed paper's year (year-only)."""
    import gzip
    text = gzip.decompress(path.read_bytes()).decode("utf-8", "replace")
    for rec in _JFM_RECORD.findall(text):
        stats["rows_in"] += 1
        zid = _zb(rec, "zbl_id")
        try:
            vol = int(zid.split(".")[0])
        except ValueError:
            stats["dropped_bad_id"] += 1
            continue
        if vol > JFM_MAX_VOLUME:
            stats["dropped_volume_after_61"] += 1
            continue
        review, rtype = _zb(rec, "review_text"), _zb(rec, "review_type")
        if not review or JFM_PLACEHOLDER in review:
            stats["dropped_no_review"] += 1
            continue
        if rtype not in ("", "review"):
            stats[f"dropped_review_type_{rtype}"] += 1
            continue
        try:
            year = int(_zb(rec, "publication_year"))
        except ValueError:
            stats["dropped_bad_date"] += 1
            continue
        if year > cutoff.year - 1:
            stats["dropped_after_cutoff"] += 1
            continue
        review = re.sub(r"\s+", " ", _JFM_XREF.sub("", review)).strip()
        meta = {"zbl_id": zid, "jfm_volume": vol, "paper_year": year, "paper_language": _zb(rec, "language"),
                "review_language": _zb(rec, "review_language"), "source": _zb(rec, "source"),
                "date_precision": "year", "bucket": "science"}
        row = _row(f"jfm_{zid}", "jfm", dt.date(year, 1, 1), "Jahrbuch über die Fortschritte der Mathematik", "",
                   _zb(rec, "reviewer"), review, meta)
        row["headline"] = _zb(rec, "document_title")
        yield row


IA_CHUNK_WORDS = 2000          # whole issues/volumes are cut into ~2k-word documents (C1 drop, dedup)
_IA_HYPHEN = re.compile(r"(\w)[-¬]\s*\n\s*(\w)")
_IA_BLANK = re.compile(r"\n\s*\n")
_GOOGLE_END = re.compile(r"google\s*\.\s*com\s*/?", re.I)
_TRANSLATION = re.compile(r"\b(?:NACA[- ]?TM|Technical Memorand|translat)", re.I)
_IA_ITEMS: dict[str, dict] = {}


def ia_items(source: str) -> dict:
    """config/<source>_items.tsv written by src.data.ia_catalog (date, precision, title, ...)."""
    if source not in _IA_ITEMS:
        cfg = load_config(repo_path("config/paths.yaml"))
        path = repo_path(load_config(repo_path(cfg["sources"]))[source]["items"])
        lines = path.read_text(encoding="utf-8").splitlines()
        header = lines[0].split("\t")
        _IA_ITEMS[source] = {f[0]: dict(zip(header, f)) for f in (l.split("\t") for l in lines[1:]) if f and f[0]}
    return _IA_ITEMS[source]


def ia_clean(text: str) -> str:
    """Strip Google's scan boilerplate, join line-end hyphenation, unwrap lines inside paragraphs."""
    head = text[:6000]
    if "digital copy of a book" in head or "Google Book Search" in head:
        ends = list(_GOOGLE_END.finditer(head))
        if ends:
            text = text[ends[-1].end():]
    text = _IA_HYPHEN.sub(r"\1\2", text)
    paras = (" ".join(l.strip() for l in p.splitlines() if l.strip()) for p in _IA_BLANK.split(text))
    return "\n\n".join(p for p in paras if p)


def ia_chunks(text: str, words: int = IA_CHUNK_WORDS) -> list[str]:
    out, cur, n = [], [], 0
    for p in text.split("\n\n"):
        cur.append(p)
        n += len(p.split())
        if n >= words:
            out.append("\n\n".join(cur))
            cur, n = [], 0
    if cur:
        if out and n < words // 4:
            out[-1] += "\n\n" + "\n\n".join(cur)
        else:
            out.append("\n\n".join(cur))
    return out


def rows_ia_text(path: Path, key: str, cutoff: dt.date, stats: Counter, source: str = ""):
    """archive.org <id>_djvu.txt of an issue or volume (science bucket): cleaned, cut into ~2k-word
    documents, dated from config/<source>_items.tsv. Period human translations are flagged in meta."""
    item = ia_items(source).get(key)
    stats["rows_in"] += 1
    if item is None:
        stats["dropped_not_in_catalog"] += 1
        return
    date, precision = dt.date.fromisoformat(item["date"]), item["precision"]
    if precision == "year":
        span_end = dt.date(date.year, 12, 31)                  # year-only 1939 -> after the cutoff
    elif precision == "month":
        span_end = (date.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)
    else:
        span_end = date
    if span_end > cutoff:
        stats["dropped_after_cutoff"] += 1
        return
    if path.suffix.lower() == ".pdf":                         # e.g. USGS reports: OCR text layer
        import fitz   # pymupdf
        with fitz.open(path) as doc:
            raw = "\n\n".join(page.get_text() for page in doc)
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")
    text = ia_clean(raw)
    if not text:
        stats["dropped_empty"] += 1
        return
    translated = bool(_TRANSLATION.search(item.get("title", "")))
    chunks = ia_chunks(text)
    for i, chunk in enumerate(chunks):
        row = _row(f"{source}_{key}_c{i:03d}", source, date, item.get("title", ""), "", "", chunk,
                   {"ia_id": key, "volume": item.get("volume"), "chunk": i, "n_chunks": len(chunks),
                    "date_precision": precision, "period_translation": translated, "bucket": "science"})
        row["headline"] = item.get("title", "")
        yield row


PSM_MAX_VOLUME = 87      # vols 1-87 (May 1872 - Sep 1915) are proofread on Wikisource
_WS_TITLE = re.compile(r"^Page:Popular Science Monthly Volume (\d+)\.djvu/(\d+)$")
_WS_QUALITY = re.compile(r'<pagequality level="(\d)"')
_WS_NOINCLUDE = re.compile(r"<noinclude>.*?</noinclude>", re.S)
_WS_TEMPLATE = re.compile(r"\{\{([^{}]*)\}\}")
_WS_DROP = {"nop", "dhr", "rule", "clear", "dotted tc line", "rh", "running header", "pline", "ppoem", "gap"}


def _ws_template(m: re.Match) -> str:
    parts = [p.strip() for p in m.group(1).split("|")]
    name = parts[0].lower()
    if name in _WS_DROP or name.startswith("hwe"):
        return ""
    if name.startswith("hws"):
        return parts[2] if len(parts) > 2 else (parts[1] if len(parts) > 1 else "")
    positional = [p for p in parts[1:] if "=" not in p]
    return positional[-1] if positional else ""


def wikitext_to_text(wt: str) -> str:
    """Plain text from a proofread Wikisource page: header/footer, templates (keeping their text),
    tables, images, math and markup removed; hyphenated words across pages joined via hws/hwe."""
    wt = _WS_NOINCLUDE.sub("", wt)
    wt = re.sub(r"<math>.*?</math>|\{\|.*?\|\}", " ", wt, flags=re.S)
    while _WS_TEMPLATE.search(wt):
        wt = _WS_TEMPLATE.sub(_ws_template, wt)
    wt = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", "", wt, flags=re.I)
    wt = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", wt)
    wt = re.sub(r"<br\s*/?>", "\n", wt)
    wt = re.sub(r"<[^>]+>", "", wt).replace("'''", "").replace("''", "")
    paras = (re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", wt))
    return "\n\n".join(p for p in paras if p)


def rows_psm_wikisource(path: Path, key: str, cutoff: dt.date, stats: Counter):
    """English Wikisource pages-articles dump: Popular Science Monthly vols <= PSM_MAX_VOLUME, Page namespace,
    proofread or validated pages only (pagequality 3/4), assembled per volume in page order and cut into
    ~2k-word documents. Keyed text: skips the OCR gates, gets the C1 screen. Dated by the volume's first year."""
    import bz2
    import xml.etree.ElementTree as ET
    if not path.name.endswith(".xml.bz2"):
        return
    vols: dict[int, dict[int, str]] = {}
    title = None
    for _, el in ET.iterparse(bz2.open(path), events=("end",)):
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "title":
            title = el.text or ""
        elif tag == "text":
            m = _WS_TITLE.match(title or "")
            if m and int(m.group(1)) <= PSM_MAX_VOLUME:
                stats["rows_in"] += 1
                wt = el.text or ""
                q = _WS_QUALITY.search(wt)
                if not q or q.group(1) not in ("3", "4"):
                    stats["dropped_not_proofread"] += 1
                else:
                    vols.setdefault(int(m.group(1)), {})[int(m.group(2))] = wikitext_to_text(wt)
        elif tag == "page":
            el.clear()
    for vol in sorted(vols):
        start = 1872 * 12 + 4 + 6 * (vol - 1)            # vol 1 = May 1872, six months per volume
        year = start // 12
        text = "\n\n".join(t for _, t in sorted(vols[vol].items()) if t)
        for i, chunk in enumerate(ia_chunks(text)):
            row = _row(f"psm_ws_v{vol:03d}_c{i:03d}", "psm_wikisource", dt.date(year, 1, 1),
                       "Popular Science Monthly", "", "", chunk,
                       {"volume": vol, "pages": len(vols[vol]), "chunk": i, "date_precision": "year",
                        "bucket": "science"})
            row["headline"] = f"Popular Science Monthly, Volume {vol}"
            yield row


_PG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\n", re.I)
_PG_END = re.compile(r"\n[^\n]*\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", re.I)
_YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")
_DEATH = re.compile(r"\d{3,4}\??\s*-\s*(\d{3,4})")


def html_to_text(doc: str) -> str:
    """Plain text from a PG HTML edition: block elements become paragraph breaks; script/style and the
    page-number spans PG uses are dropped; whitespace inside a paragraph is collapsed."""
    from html.parser import HTMLParser

    class P(HTMLParser):
        BLOCK = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "pre", "blockquote", "hr", "table"}

        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.out, self.skip = [], 0

        def handle_starttag(self, tag, attrs):
            cls = dict(attrs).get("class") or ""
            if tag in ("script", "style") or (tag == "span" and "pagenum" in cls):
                self.skip += 1
            elif tag in self.BLOCK:
                self.out.append("\n\n")

        def handle_endtag(self, tag):
            if tag in ("script", "style", "span") and self.skip:
                self.skip -= 1
            elif tag in self.BLOCK:
                self.out.append("\n\n")

        def handle_data(self, data):
            if not self.skip:
                self.out.append(data)

    p = P()
    p.feed(doc)
    paras = (re.sub(r"\s+", " ", x).strip() for x in re.split(r"\n\s*\n", "".join(p.out)))
    return "\n\n".join(x for x in paras if x)


def rows_gutenberg(path: Path, key: str, cutoff: dt.date, stats: Counter, source: str = ""):
    """Project Gutenberg plain text (science bucket): PG header/footer removed; one row per book. Books
    whose authors are not all dead by 1930 ("check") are kept only if the front matter names a year
    1800-1938 and none >= 1939. Keyed text: skips the OCR gates, gets the C1 screen."""
    item = ia_items(source).get(key)
    stats["rows_in"] += 1
    if item is None:
        stats["dropped_not_in_catalog"] += 1
        return
    raw = path.read_text(encoding="utf-8", errors="replace")
    if raw.lstrip()[:500].lower().startswith(("<!doctype", "<?xml", "<html")):
        raw = html_to_text(raw)
    s, e = _PG_START.search(raw), _PG_END.search(raw)
    text = raw[s.end() if s else 0: e.start() if e else len(raw)].strip()
    years = [int(y) for y in _YEAR.findall(text[:5000])]
    early = [y for y in years if 1800 <= y <= cutoff.year - 1]
    if item["verdict"] == "check" and (not early or any(y > cutoff.year - 1 for y in years)):
        stats["dropped_date_unverified"] += 1
        return
    deaths = [int(d) for d in _DEATH.findall(item.get("authors", ""))]
    year = max(early) if early else (1910 if item["kind"] == "eb11" else min(max(deaths, default=1930), 1930))
    if not text:
        stats["dropped_empty"] += 1
        return
    row = _row(f"{source}_{key}", source, dt.date(year, 1, 1), item.get("title", ""), "", item.get("authors", ""),
               text, {"pg_id": key, "kind": item["kind"], "date_verdict": item["verdict"], "locc": item.get("locc"),
                      "date_precision": "year", "bucket": "science"})
    row["headline"] = item.get("title", "")
    yield row


_TEI = "{http://www.tei-c.org/ns/1.0}"
_DINGLER_VOL = re.compile(r"/sources/volumes/pj(\d+)\.xml$")
_DINGLER_YEAR = re.compile(r"Jahrgang\s+(\d{4})")


def _tei_text(el) -> str:
    """Text of a TEI element without editorial additions: typed <note>s (the project's remarks, e.g.
    "Anmerkungszeichen ... fehlt im Text") and <figDesc> placeholders are dropped, author footnotes
    (untyped notes) kept. Block elements start a new line; cells and line breaks become spaces."""
    parts: list[str] = []
    flat = lambda s: (s or "").replace("\n", " ")   # source line wrapping is not a paragraph break

    def walk(e) -> None:
        tag = e.tag.rsplit("}", 1)[-1]
        if tag == "figDesc" or (tag == "note" and e.get("type")):
            parts.append(flat(e.tail))
            return
        if tag in ("p", "head", "item", "row", "titlePart", "bibl", "byline"):
            parts.append("\n")
        elif tag in ("cell", "lb"):
            parts.append(" ")
        parts.append(flat(e.text))
        for c in e:
            walk(c)
        parts.append(flat(e.tail))

    parts.append(flat(el.text))
    for c in el:
        walk(c)
    lines = (" ".join(line.split()) for line in "".join(parts).split("\n"))
    return "\n".join(line for line in lines if line)


def rows_dingler(path: Path, key: str, cutoff: dt.date, stats: Counter):
    """Dinglers Polytechnisches Journal, TEI volumes from the pinned GitHub tarball: one row per article
    (<text type="art_*">); teiHeader, volume/issue front matter, editorial notes and figure placeholders
    dropped. Keyed text: skips the OCR gates, gets the C1 screen. Dated by the volume's Jahrgang
    (year-only). Historical typography (long s, combining e) is kept as in the source."""
    import tarfile
    import xml.etree.ElementTree as ET
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            v = _DINGLER_VOL.search(member.name)
            if not v or not member.isfile():
                continue
            vol = int(v.group(1))
            root = ET.parse(tar.extractfile(member)).getroot()
            header = root.find(f"{_TEI}teiHeader")
            titles = " ".join(t.text or "" for t in header.iter(f"{_TEI}title")) if header is not None else ""
            y = _DINGLER_YEAR.search(titles)
            for t in root.iter(f"{_TEI}text"):
                if not t.get("type", "").startswith("art_"):
                    continue
                stats["rows_in"] += 1
                if not y:
                    stats["dropped_bad_date"] += 1
                    continue
                year = int(y.group(1))
                if year > cutoff.year - 1:
                    stats["dropped_after_cutoff"] += 1
                    continue
                text = _tei_text(t)
                if not text:
                    stats["dropped_empty"] += 1
                    continue
                aid = t.get("{http://www.w3.org/XML/1998/namespace}id") or f"pj{vol:03d}_{stats['rows_in']}"
                front = t.find(f"{_TEI}front")
                headline = " ".join(_tei_text(front).split())[:300] if front is not None else ""
                row = _row(f"dingler_{aid}", "dingler", dt.date(year, 1, 1), "Polytechnisches Journal", "", "", text,
                           {"volume": vol, "tei_id": aid, "article_type": t.get("type"), "date_precision": "year",
                            "bucket": "science"})
                row["headline"] = headline
                yield row


ADAPTERS = {"congressional_record": rows_congressional_record, "hmd_newspapers": rows_hmd_newspapers,
            "dingler": rows_dingler,
            "gutenberg_sci_en": functools.partial(rows_gutenberg, source="gutenberg_sci_en"),
            "gutenberg_sci_de": functools.partial(rows_gutenberg, source="gutenberg_sci_de"),
            "jstor_ejc": rows_jstor_ejc, "royal_society_corpus": rows_royal_society_corpus,
            "jfm": rows_jfm,
            "psm_wikisource": rows_psm_wikisource,
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
    adapter = ADAPTERS.get(source) or functools.partial(rows_ia_text, source=source)   # ia_query, usgs sources
    for row in adapter(Path(path), key, dt.date.fromisoformat(cutoff_iso), stats):
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
        # content-addressed: an existing file is the same bytes (e.g. the empty file of every dropped item).
        # On Windows two workers renaming onto the same name collide (access denied), so keep the first.
        try:
            if final.exists():
                os.remove(tmp)
            else:
                os.replace(tmp, final)
        except PermissionError:
            if not final.exists():
                raise
            os.remove(tmp)
        result.update({"file": final.name, "sha256": digest, "file_bytes": final.stat().st_size})
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", required=True, help=f"one of {sorted(ADAPTERS)} or a source with ia_query")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="count only; write nothing")
    args = ap.parse_args()

    cfg = load_config(Path(args.config) if args.config else repo_path("config/paths.yaml"))
    src = load_config(repo_path(cfg["sources"]))[args.source]
    if args.source not in ADAPTERS and not (src.get("ia_query") or src.get("usgs_series")):
        sys.exit(f"no adapter for {args.source}")
    raw_dir = repo_path(src["dest"])
    out_dir = ingested_dir(cfg, args.source, src)
    raw_manifest = json.loads((raw_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    mpath = out_dir / "MANIFEST.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {
        "stage": "ingested", "source": args.source, "hf_repo": src.get("hf_repo", args.source), "revision": raw_manifest["revision"],
        "cutoff": str(cfg["cutoff"]), "files": {}}
    # source-level provenance (science-bucket task, 2026-10-06); item-level title/year/language are row columns
    manifest["source_meta"] = {k: (str(v) if isinstance(v, dt.date) else v)     # YAML dates -> ISO strings
                               for k, v in ((k, src.get(k)) for k in SOURCE_META_FIELDS)}

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
