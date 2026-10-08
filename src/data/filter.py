"""Filter stage: paragraph removals, chunking, OCR gates and the C1 screen (CLAUDE.md rules 3-7).

Input per language: the selected parquet (src.data.select) minus document-dedup drops (src.data.dedup)
and, for training documents, minus the blocks listed by src.data.para_dedup (heldout at its CUTOFF, then
reprint). Output, one file per selected input: data/filtered/<lang>/<source>/<sha12>.parquet with every
surviving document of every split, and data/filtered/<lang>/MANIFEST.json (gate parameters, C1 list,
counts per source x period x outcome, C1 hits per term x year).

  split    src.data.splits: holdout / val / train / embargo / post, and test_set A / B / C.
  chunks   documents longer than CHUNK_MAX words (whole books) are cut at line boundaries into chunks of
           about CHUNK_WORDS words, `<id>_cNNN`; a chunk keeps its parent's split.
  OCR      hit rate, word share and token count are columns on every row (the OCR covariate, rule 4).
           The gates (src.data.ocr_quality.gate; Voelkischer Beobachter: segment cleanup) apply to train
           and val only; keyed sources skip them; held-out rows are never gated.
  C1       rows containing a C1 coinage (probes/c1_screen.csv, whole word, case-insensitive) are KEPT and
           counted per term, year and split; the term is a column (c1_term) and every hit is listed in the
           audit for a misdating check (user, 2026-10-07: pre-cutoff hits are genuine in-window text such as
           surnames, OCR noise, Popeye's Jeep; CLAUDE.md rule 6 revised).
  train_ok split in (train, val) and gate passed. Seen-token sampling happens later.

    python -m src.data.filter --lang en [--workers 8] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.data import ocr_quality as oq
from src.data.dedup import selected_files
from src.data.download import load_config, repo_path
from src.data.ingest_extra import sha256_file
from src.data.para_dedup import CUTOFF as PARA_CUTOFF
from src.data.splits import parent_id, split_of, test_set

CHUNK_MAX, CHUNK_WORDS, CHUNK_TAIL = 10_000, 2_000, 500
FLUSH_ROWS = 20_000          # output rows held per worker before they are written (one row group)
C1_FILE = "probes/c1_screen.csv"

OUT_SCHEMA = pa.schema([
    ("article_id", pa.string()), ("parent_id", pa.string()), ("source", pa.string()), ("lang", pa.string()),
    ("bucket", pa.string()), ("category", pa.string()), ("keyed", pa.bool_()), ("page_level", pa.bool_()),
    ("date", pa.string()), ("year", pa.int16()), ("period", pa.string()), ("split", pa.string()),
    ("test_set", pa.string()), ("text", pa.string()), ("n_words", pa.int32()), ("n_bytes", pa.int32()),
    ("ocr_hit", pa.float32()), ("ocr_word_share", pa.float32()), ("ocr_tokens", pa.int32()),
    ("gate", pa.string()), ("c1_term", pa.string()), ("train_ok", pa.bool_()),
])


def period(date: str, date_class: str) -> str:
    if date_class != "pre":
        return date_class
    y = date[:4]
    return "<1900" if y < "1900" else "1900-19" if y < "1920" else "1920-29" if y < "1930" else "1930-39.06"


def chunks(text: str) -> list[str]:
    out, cur, n = [], [], 0
    for line in text.splitlines(keepends=True):
        cur.append(line)
        n += len(line.split())
        if n >= CHUNK_WORDS:
            out.append("".join(cur))
            cur, n = [], 0
    if cur:
        if out and n < CHUNK_TAIL:
            out[-1] += "".join(cur)
        else:
            out.append("".join(cur))
    return out


def c1_patterns(lang: str) -> tuple[list[tuple[str, re.Pattern]], str]:
    p = repo_path(C1_FILE)
    with open(p, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["lang"] == lang]
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    return [(r["term"], re.compile(r["regex"], re.IGNORECASE)) for r in rows], digest


def c1_hit(text: str, pats: list) -> str:
    for term, rx in pats:
        if rx.search(text):
            return term
    return ""


def remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    for s, e in sorted(spans, reverse=True):
        text = text[:s] + text[e:]
    return text


_W: dict = {}


def _worker_state(lang: str, drops_file: str, para_files: tuple, source: str) -> dict:
    """Lexicon, C1 patterns, document drops and this source's paragraph removals, loaded once per worker."""
    if _W.get("lang") != lang:
        cfg = load_config(repo_path("config/paths.yaml"))
        _W.update(lang=lang, lexicon=oq.load_lexicon(cfg, lang), c1=c1_patterns(lang)[0], drops=set(), spans={})
        if drops_file:
            _W["drops"] = set(pq.read_table(drops_file, columns=["article_id"]).column("article_id").to_pylist())
    if source not in _W["spans"]:
        spans: dict[str, list] = defaultdict(list)
        for path, min_c in para_files:
            t = pq.read_table(path, columns=["article_id", "start", "end", "containment"],
                              filters=[("source", "=", source), ("containment", ">=", min_c)])
            for a, s, e in zip(*(t.column(c).to_pylist() for c in ("article_id", "start", "end"))):
                spans[a].append((s, e))
        _W["spans"][source] = dict(spans)
    return _W


def _flush(rows: list[dict], writer, tmp: Path | None, out_dir: str, path: str):
    """Append rows to this input's temporary output file (opened on first use) and empty the list."""
    if writer is None:
        d = Path(out_dir) / rows[0]["source"]
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".tmp_{Path(path).stem}.parquet"
        writer = pq.ParquetWriter(tmp, OUT_SCHEMA, compression="zstd")
    writer.write_table(pa.Table.from_pylist(rows, schema=OUT_SCHEMA))
    rows.clear()
    return writer, tmp


def _filter_job(job: tuple) -> dict:
    path, lang, drops_file, para_files, out_dir, dry_run = job
    stats: Counter = Counter()
    rows: list[dict] = []
    writer, tmp, n_rows = None, None, 0
    pf = pq.ParquetFile(path)
    state = None
    for batch in pf.iter_batches(batch_size=2000):
        if len(rows) >= FLUSH_ROWS:
            n_rows += len(rows)
            writer, tmp = _flush(rows, writer, tmp, out_dir, path)
        b = batch.to_pydict()
        for i in range(batch.num_rows):
            src = b["source"][i]
            if state is None:
                state = _worker_state(lang, drops_file, para_files, src)
            aid = b["article_id"][i]
            if aid in state["drops"]:
                continue
            date, dc = b["date"][i], b["date_class"][i]
            split = split_of(aid, dc)
            per = period(date, dc)
            text = b["text"][i] or ""
            removed = state["spans"].get(src, {}).get(aid)
            if removed and split == "train":
                before = len(text.split())
                text = remove_spans(text, removed)
                stats[f"{src}|{per}|para_removed_words"] += before - len(text.split())
            parts = chunks(text) if len(text.split()) > CHUNK_MAX else [text]
            for k, part in enumerate(parts):
                cid = f"{aid}_c{k:03d}" if len(parts) > 1 else aid
                gate = ""
                hit, ws, ntok = float("nan"), float("nan"), len(part.split())
                if src in oq.SEGMENT_SOURCES and split in ("train", "val"):
                    part, n_in, n_kept = oq.clean_segments(part)        # cleaned per segment, not gated whole
                    stats[f"{src}|{per}|segment_dropped_words"] += n_in - n_kept
                    hit, ws, ntok = oq.score(part, state["lexicon"]), oq.word_share(part), len(part.split())
                    gate = "short" if ntok < oq.min_tokens(src) else ""
                elif b["keyed"][i]:
                    hit, ws = oq.score(part, state["lexicon"]), oq.word_share(part)
                else:
                    reason, hit, ws, ntok = oq.gate(part, state["lexicon"], lang, src)
                    if split in ("train", "val") and reason:
                        gate = reason
                term = c1_hit(part, state["c1"])
                if term:
                    stats[f"c1|{term}|{date[:4]}|{split}"] += 1
                ok = split in ("train", "val") and not gate
                nw = len(part.split())
                stats[f"{src}|{per}|{split}|docs"] += 1
                stats[f"{src}|{per}|{split}|words"] += nw
                outcome = "ok" if ok else (gate or "not_train")
                if term:
                    stats[f"{src}|{per}|{split}|c1_words"] += nw
                stats[f"{src}|{per}|{split}|{outcome}_words"] += nw
                if dry_run:
                    continue
                rows.append({"article_id": cid, "parent_id": parent_id(aid), "source": src, "lang": lang,
                             "bucket": b["bucket"][i], "category": b["category"][i], "keyed": b["keyed"][i],
                             "page_level": b["page_level"][i], "date": date, "year": b["year"][i], "period": per,
                             "split": split, "test_set": test_set(date, split) or "", "text": part, "n_words": nw,
                             "n_bytes": len(part.encode("utf-8")), "ocr_hit": hit, "ocr_word_share": ws,
                             "ocr_tokens": ntok, "gate": gate, "c1_term": term, "train_ok": ok})
    if rows:
        n_rows += len(rows)
        writer, tmp = _flush(rows, writer, tmp, out_dir, path)
    out = {}
    if writer is not None:
        writer.close()
        digest = sha256_file(tmp)
        final = tmp.parent / f"{digest[:12]}.parquet"
        if final.exists():
            os.remove(tmp)
        else:
            os.replace(tmp, final)
        out = {"file": f"{tmp.parent.name}/{final.name}", "sha256": digest, "rows": n_rows}
    return {"input": path, "output": out, "stats": dict(stats)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["en", "de"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="count only; write nothing")
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    files, _ = selected_files(cfg, args.lang)
    drops = repo_path(cfg["dedup"]) / args.lang / "drops.parquet"
    if not drops.exists():
        sys.exit(f"{drops} missing: run src.data.dedup --lang {args.lang} first")
    pd = repo_path(cfg["data_root"]) / "para_dedup" / args.lang
    para = tuple((str(pd / f"{n}_drops.parquet"), c) for n, c in (("heldout", PARA_CUTOFF), ("reprint", 0.0))
                 if (pd / f"{n}_drops.parquet").exists())
    if args.lang == "en" and not any("heldout" in p for p, _ in para):
        print("WARNING: no heldout_drops.parquet: rule 2a (held-out protection) has not run", flush=True)
    out_dir = repo_path(cfg["filtered"]) / args.lang
    t0 = time.time()
    stats: Counter = Counter()
    outputs = {}
    jobs = [(p, args.lang, str(drops), para, str(out_dir), args.dry_run) for p, _ in files]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        for k, fut in enumerate(as_completed([ex.submit(_filter_job, j) for j in jobs]), 1):
            r = fut.result()
            stats.update(r["stats"])
            if r["output"]:
                outputs[r["input"]] = r["output"]
            if k % 500 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} files, {time.time() - t0:.0f}s", flush=True)
    per = defaultdict(Counter)
    c1 = defaultdict(Counter)
    for key, v in stats.items():
        if key.startswith("c1|"):
            _, term, year, split = key.split("|")
            c1[term][f"{year}|{split}"] += v
        else:
            src, rest = key.split("|", 1)
            per[src][rest] += v
    for src, c in sorted(per.items()):
        tr = sum(v for k, v in c.items() if k.endswith("|train|words"))
        ok = sum(v for k, v in c.items() if k.endswith("|train|ok_words"))
        print(f"  {src:24} train words {tr:>14,} kept {ok:>14,} ({100 * ok / max(tr, 1):5.1f} %)", flush=True)
    for term, c in sorted(c1.items()):
        print(f"  C1 {term}: {sum(c.values())} hits; {dict(sorted(c.items()))}", flush=True)
    if args.dry_run:
        return
    pats, c1_sha = c1_patterns(args.lang)
    manifest = {"stage": "filtered", "lang": args.lang,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "gate_params": oq.gate_params(cfg, args.lang),
                "chunks": {"CHUNK_MAX": CHUNK_MAX, "CHUNK_WORDS": CHUNK_WORDS, "CHUNK_TAIL": CHUNK_TAIL},
                "c1": {"file": C1_FILE, "sha256": c1_sha, "terms": [t for t, _ in pats]},
                "para_dedup": [{"file": p, "min_containment": c} for p, c in para],
                "outputs": outputs, "per_source": {s: dict(c) for s, c in sorted(per.items())},
                "c1_hits": {t: dict(c) for t, c in sorted(c1.items())}, "seconds": round(time.time() - t0)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
