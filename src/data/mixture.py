"""Training mixture: how many times (0, 1, 2) each filtered training document is seen (CLAUDE.md rule 7,
user decisions 2026-10-08 in HANDOFF §12; parameters in config/mixture.yaml).

Unit = a parent document (chunks of a book or a long opinion share their parent's count). Only rows with
split == train and train_ok are candidates; general (non-science) text before 1900 never is. Tokens are
word counts x tokens_per_word until the BPE exists (decision 5: re-run this stage then).

  general periods  1930-39.06 (target 7.37B): every document once, documents from repeat_from_year on a
                   second time; 1920-29 (2.6B) and 1900-19 (0.8B): sampled once. Within a period the order
                   is an exponential race with weight w = exp(-(1939 - year) / half_life): key = -ln(u) / w,
                   u a seeded hash of the parent id, smallest key first; all first epochs come before any
                   second epoch. Caps per period, as shares of the period's seen tokens: German <= 25 %,
                   English books <= 12 %, legal (case law, Federal Register) <= 10 % of the period's English
                   (1930s also <= 0.35B). A capped class takes its units in key order until its cap; the
                   period takes accepted units in key order until its target. Caps depend on the period
                   total, so the plan is iterated to a fixed point.
  science          per language (EN 0.85B, DE 0.30B), outside the period caps and recency weighting; tiers
                   from config/mixture.yaml in order, uniform random order inside a tier, a second epoch only
                   if every first epoch fits.

Output data/mixture/<lang>/<sha12>.parquet (article_id, parent_id, source, lang, period, category, year,
n_words, est_tokens, count) for every row with count >= 1, and data/mixture/MANIFEST.json (parameters,
inputs, unique vs seen tokens per period x language x category, per source, cap checks).

    python -m src.data.mixture [--profile 12B] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.data.download import load_config, repo_path

LANGS = ("en", "de")
PERIODS = ("1930-39.06", "1920-29", "1900-19")
FREE, BOOKS, LEGAL, GERMAN = 0, 1, 2, 3          # cap classes
SECOND_EPOCH = 1e12                              # key offset: every first epoch before any second
META = ["parent_id", "source", "bucket", "category", "period", "split", "train_ok", "year", "n_words"]
CODED = ("source", "bucket", "category", "period")
OUT_SCHEMA = pa.schema([("article_id", pa.string()), ("parent_id", pa.string()), ("source", pa.string()),
                        ("lang", pa.string()), ("period", pa.string()), ("category", pa.string()),
                        ("year", pa.int16()), ("n_words", pa.int32()), ("est_tokens", pa.float32()),
                        ("count", pa.int8())])


def uniform(parent_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded 64-bit hash of each parent id and a uniform (0, 1) number derived from it."""
    h = pd.util.hash_array(np.asarray(parent_ids, dtype=object), hash_key=f"{seed:016d}"[:16])
    return h, ((h >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)


def race_keys(u: np.ndarray, year: np.ndarray, half_life: float, ref_year: int) -> np.ndarray:
    """Exponential race: weight exp(-(ref - year) / half_life); a smaller key is drawn earlier."""
    return -np.log(u) / np.exp(-(ref_year - year.astype(np.float64)) / half_life)


def plan(tokens: np.ndarray, cls: np.ndarray, key: np.ndarray, second: np.ndarray, target: float,
         caps: dict | None = None, legal_absolute: float = math.inf) -> np.ndarray:
    """Seen count (0, 1, 2) per unit. second: units allowed a second epoch. caps: german, books,
    legal_of_english as shares of the seen total; None = no caps (science)."""
    n = len(tokens)
    idx = np.concatenate([np.arange(n), np.flatnonzero(second)])
    ukey = np.concatenate([key, key[second] + SECOND_EPOCH])
    order = idx[np.argsort(ukey, kind="stable")]
    tok, c = tokens[order], cls[order]
    total = min(target, float(tok.sum()))
    de_seen = float(tok[c == GERMAN].sum())
    accept = np.ones(len(order), dtype=bool)
    for _ in range(200):
        accept = np.ones(len(order), dtype=bool)
        if caps:
            limits = {GERMAN: caps["german"] * total, BOOKS: caps["books"] * total,
                      LEGAL: min(caps["legal_of_english"] * (total - min(de_seen, caps["german"] * total)),
                                 legal_absolute)}
            for k, lim in limits.items():
                m = c == k
                accept[m & (np.cumsum(np.where(m, tok, 0.0)) > lim)] = False
        accept &= np.cumsum(np.where(accept, tok, 0.0)) <= target
        new_total = float(tok[accept].sum())
        de_seen = float(tok[accept & (c == GERMAN)].sum())
        done = abs(new_total - total) < 1.0
        total = new_total
        if done:
            break
    return np.bincount(order[accept], minlength=n).astype(np.int8)


def _candidates(t: pa.Table) -> np.ndarray:
    return np.asarray(t.column("train_ok").to_numpy(zero_copy_only=False), dtype=bool) & \
        (np.asarray(t.column("split").to_numpy(zero_copy_only=False)) == "train")


def load_meta(cfg: dict, mcfg: dict) -> dict:
    """Candidate rows of both languages as numeric arrays (string columns as codes into meta["vocab"],
    parent ids as their seeded hash), plus the per-file candidate counts for the write pass."""
    vocab = {k: {} for k in CODED}
    cols = defaultdict(list)
    files, inputs = [], []
    for li, lang in enumerate(LANGS):
        root = repo_path(cfg["filtered"]) / lang
        m = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        for o in sorted(m["outputs"].values(), key=lambda o: o["file"]):
            t = pq.read_table(root / o["file"], columns=META)
            keep = _candidates(t)
            files.append((lang, root / o["file"], int(keep.sum())))
            if not keep.any():
                continue
            t = t.filter(pa.array(keep))
            h, u = uniform(t.column("parent_id").to_numpy(zero_copy_only=False), mcfg["seed"])
            cols["h"].append(h)
            cols["u"].append(u)
            for k in CODED:
                d = t.column(k).combine_chunks().dictionary_encode()
                lut = np.array([vocab[k].setdefault(v, len(vocab[k])) for v in d.dictionary.to_pylist()],
                               dtype=np.int16)
                cols[k].append(lut[d.indices.to_numpy(zero_copy_only=False)])
            cols["year"].append(t.column("year").to_numpy(zero_copy_only=False).astype(np.int16))
            cols["n_words"].append(t.column("n_words").to_numpy(zero_copy_only=False).astype(np.int64))
            cols["lang"].append(np.full(t.num_rows, li, dtype=np.int8))
        inputs.append((lang, str(root / "MANIFEST.json"),
                       hashlib.sha256((root / "MANIFEST.json").read_bytes()).hexdigest()))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out.update(inputs=inputs, files=files, vocab=vocab)
    tpw = np.array([mcfg["tokens_per_word"][lang] for lang in LANGS])
    out["tokens"] = out["n_words"] * tpw[out["lang"]]
    return out


def code(meta: dict, col: str, value: str) -> int:
    return meta["vocab"][col].get(value, -1)


def names(meta: dict, col: str) -> np.ndarray:
    v = meta["vocab"][col]
    return np.array(sorted(v, key=v.get), dtype=object)


def science_key(source_names: np.ndarray, u: np.ndarray, tiers: list[list[str]]) -> np.ndarray:
    named = {s: i for i, tier in enumerate(tiers) for s in tier if s != "*"}
    rest = next((i for i, tier in enumerate(tiers) if "*" in tier), len(tiers))
    tier = np.array([named.get(s, rest) for s in source_names], dtype=np.float64)
    return tier * 1e6 + u


def draw(meta: dict, mcfg: dict, profile: str) -> tuple[np.ndarray, dict]:
    """Seen count per row, and the plan summary."""
    prof = mcfg["profiles"][profile]
    parents, first, inv = np.unique(meta["h"], return_index=True, return_inverse=True)
    p_tok = np.bincount(inv, weights=meta["tokens"])
    p_lang, p_src, p_cat = meta["lang"][first], meta["source"][first], meta["category"][first]
    p_per, p_year, p_u = meta["period"][first], meta["year"][first], meta["u"][first]
    p_sci = meta["bucket"][first] == code(meta, "bucket", "science")
    en, de = LANGS.index("en"), LANGS.index("de")
    cls = np.full(len(parents), FREE, dtype=np.int8)
    cls[(p_lang == en) & (p_cat == code(meta, "category", "books"))] = BOOKS
    cls[(p_lang == en) & (p_cat == code(meta, "category", "legal"))] = LEGAL
    cls[p_lang == de] = GERMAN
    count = np.zeros(len(parents), dtype=np.int8)
    key = race_keys(p_u, p_year, mcfg["half_life_years"], mcfg["reference_year"])
    summary = {"candidate_parents": int(len(parents)),
               "excluded_general_pre1900_parents": int(((~p_sci) & (p_per == code(meta, "period", "<1900"))).sum())}
    for per in PERIODS:
        m = (~p_sci) & (p_per == code(meta, "period", per))
        second = (p_year[m] >= mcfg["repeat_from_year"]) if per == "1930-39.06" else np.zeros(int(m.sum()), dtype=bool)
        count[m] = plan(p_tok[m], cls[m], key[m], second, float(prof["periods"][per]), mcfg["caps"],
                        float(mcfg["caps"].get("legal_absolute", {}).get(per, math.inf)))
    src_names = names(meta, "source")
    for li, lang in enumerate(LANGS):
        m = p_sci & (p_lang == li)
        k = science_key(src_names[p_src[m]], p_u[m], mcfg["science_tiers"][lang])
        count[m] = plan(p_tok[m], np.full(int(m.sum()), FREE, dtype=np.int8), k,
                        np.full(int(m.sum()), mcfg["max_epochs"] >= 2), float(prof["science"][lang]))
    return count[inv], summary


def tables(meta: dict, count: np.ndarray, mcfg: dict, profile: str) -> dict:
    """Unique vs seen tokens per period x language x category and per source; cap checks."""
    sci = meta["bucket"] == code(meta, "bucket", "science")
    df = pd.DataFrame({"lang": meta["lang"], "period": np.where(sci, -1, meta["period"]),
                       "category": meta["category"], "source": meta["source"], "tokens": meta["tokens"],
                       "seen": meta["tokens"] * count, "second": meta["tokens"] * (count == 2)})
    lut = {k: names(meta, k) for k in ("period", "category", "source")}

    def label(col: str, v: int) -> str:
        if col == "lang":
            return LANGS[v]
        if col == "period" and v == -1:
            return "science"
        return str(lut[col][v])

    def rows(keys: list[str], vals: list[str]) -> list[dict]:
        g = df.groupby(keys)[vals].sum()
        return [{**{k: label(k, int(x)) for k, x in zip(keys, ix)}, **{v: float(r[v]) for v in vals}}
                for ix, r in g.iterrows()]

    en, de = LANGS.index("en"), LANGS.index("de")
    books, legal = code(meta, "category", "books"), code(meta, "category", "legal")
    checks = {}
    for p in PERIODS:
        d = df[df["period"] == code(meta, "period", p)]
        tot = float(d["seen"].sum()) or 1.0
        en_seen = float(d.loc[d["lang"] == en, "seen"].sum()) or 1.0
        lg = float(d.loc[(d["lang"] == en) & (d["category"] == legal), "seen"].sum())
        checks[p] = {"seen": float(d["seen"].sum()), "target": mcfg["profiles"][profile]["periods"][p],
                     "second_epoch": float(d["second"].sum()),
                     "german_share": float(d.loc[d["lang"] == de, "seen"].sum()) / tot,
                     "books_share": float(d.loc[(d["lang"] == en) & (d["category"] == books), "seen"].sum()) / tot,
                     "legal_share_of_english": lg / en_seen, "legal_seen": lg}
    for li, lang in enumerate(LANGS):
        d = df[df["lang"] == li]
        s = float(d.loc[d["period"] == -1, "seen"].sum())
        checks[f"science_{lang}"] = {"seen": s, "target": mcfg["profiles"][profile]["science"][lang],
                                     "share_of_language": s / (float(d["seen"].sum()) or 1.0)}
    general = ~sci
    checks["max_count"] = int(count.max()) if len(count) else 0
    checks["second_epoch_rows_before_repeat_year"] = int(((count == 2) & general
                                                          & (meta["year"] < mcfg["repeat_from_year"])).sum())
    checks["general_pre1900_rows_selected"] = int(((count > 0) & general
                                                   & (meta["period"] == code(meta, "period", "<1900"))).sum())
    cells = rows(["period", "lang", "category"], ["tokens", "seen", "second"])
    return {"cells": [{"period": c["period"], "lang": c["lang"], "category": c["category"],
                       "unique_tokens": c["tokens"], "seen_tokens": c["seen"], "second_epoch_tokens": c["second"]}
                      for c in cells],
            "sources": [{"period": c["period"], "lang": c["lang"], "source": c["source"],
                         "unique_tokens": c["tokens"], "seen_tokens": c["seen"]}
                        for c in rows(["period", "lang", "source"], ["tokens", "seen"])],
            "period_seen": {label("period", int(k)): float(v) for k, v in df.groupby("period")["seen"].sum().items()},
            "total_seen": float(df["seen"].sum()), "checks": checks}


def write(meta: dict, count: np.ndarray, out_root: Path, mcfg: dict) -> dict:
    """Second pass over the filtered files: every candidate row with count >= 1, one file per language."""
    writers, pos = {}, 0
    for lang, path, n in meta["files"]:
        if n == 0:
            continue
        c = count[pos:pos + n]
        pos += n
        sel = c > 0
        if not sel.any():
            continue
        t = pq.read_table(path, columns=["article_id", "parent_id", "source", "period", "category", "bucket",
                                         "year", "n_words", "split", "train_ok"])
        t = t.filter(pa.array(_candidates(t))).filter(pa.array(sel))
        per = np.where(t.column("bucket").to_numpy(zero_copy_only=False) == "science", "science",
                       t.column("period").to_numpy(zero_copy_only=False))
        words = t.column("n_words").to_numpy(zero_copy_only=False)
        tbl = pa.table({"article_id": t.column("article_id"), "parent_id": t.column("parent_id"),
                        "source": t.column("source"), "lang": pa.array([lang] * t.num_rows),
                        "period": pa.array(per.astype(str)), "category": t.column("category"),
                        "year": t.column("year").cast(pa.int16()), "n_words": t.column("n_words").cast(pa.int32()),
                        "est_tokens": pa.array((words * mcfg["tokens_per_word"][lang]).astype(np.float32)),
                        "count": pa.array(c[sel].astype(np.int8))}, schema=OUT_SCHEMA)
        if lang not in writers:
            (out_root / lang).mkdir(parents=True, exist_ok=True)
            tmp = out_root / lang / ".tmp_selection.parquet"
            writers[lang] = [pq.ParquetWriter(tmp, OUT_SCHEMA, compression="zstd"), tmp, 0]
        writers[lang][0].write_table(tbl)
        writers[lang][2] += tbl.num_rows
    assert pos == len(count), "row alignment between the two passes broke"
    outputs = {}
    for lang, (w, tmp, rows) in writers.items():
        w.close()
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        final = tmp.parent / f"{digest[:12]}.parquet"
        os.replace(tmp, final)
        outputs[lang] = {"file": f"{lang}/{final.name}", "sha256": digest, "rows": rows}
    return outputs


def print_plan(tab: dict) -> None:
    for c in tab["cells"]:
        print(f"  {c['period']:11} {c['lang']} {c['category']:12} unique {c['unique_tokens'] / 1e9:6.3f}B  "
              f"seen {c['seen_tokens'] / 1e9:6.3f}B  (2nd epoch {c['second_epoch_tokens'] / 1e9:5.3f}B)", flush=True)
    for p, v in tab["period_seen"].items():
        print(f"  {p:11} seen {v / 1e9:6.3f}B", flush=True)
    print(f"  total seen {tab['total_seen'] / 1e9:6.3f}B", flush=True)
    print("  checks: " + json.dumps(tab["checks"], indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=None, help="budget profile in config/mixture.yaml (default: its `profile`)")
    ap.add_argument("--dry-run", action="store_true", help="plan and print; write nothing")
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    mcfg = yaml.safe_load(repo_path("config/mixture.yaml").read_text(encoding="utf-8"))
    profile = args.profile or mcfg["profile"]
    t0 = dt.datetime.now()
    meta = load_meta(cfg, mcfg)
    print(f"{len(meta['tokens']):,} candidate rows loaded, {(dt.datetime.now() - t0).seconds}s", flush=True)
    count, summary = draw(meta, mcfg, profile)
    tab = tables(meta, count, mcfg, profile)
    print_plan(tab)
    if args.dry_run:
        return
    out_root = repo_path(cfg["mixture"])
    outputs = write(meta, count, out_root, mcfg)
    manifest = {"stage": "mixture", "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "profile": profile, "params": mcfg,
                "inputs": [{"lang": l, "manifest": p, "sha256": s} for l, p, s in meta["inputs"]],
                "token_estimate": "n_words x tokens_per_word (placeholder until the BPE exists)",
                "outputs": outputs, **summary, **tab, "seconds": (dt.datetime.now() - t0).seconds}
    (out_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"-> {out_root}", flush=True)


if __name__ == "__main__":
    main()
