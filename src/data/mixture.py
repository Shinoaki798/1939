"""Training mixture: how many times (0, 1, 2) each filtered training document is seen (CLAUDE.md rule 7,
user decisions 2026-10-08 in HANDOFF §12; parameters in config/mixture.yaml).

Unit = a parent document (chunks of a book or a long opinion share their parent's count). Only rows with
split == train and train_ok are candidates; general (non-science) text before 1900 never is. Tokens are
word counts x tokens_per_word until the BPE exists (decision 5: re-run this stage then).

  general periods  1930-39.06 (target 7.37B): every document once, documents from repeat_from_year on a
                   second time; 1920-29 (2.6B) and 1900-19 (0.8B): sampled once. A profile's fill_to_budget
                   period then takes whatever the caps leave short of the budget (user 2026-10-08: the
                   1920s), iterated with science, up to fill_max_share of the budget. Within a period the order
                   is an exponential race with weight w = exp(-(1939 - year) / half_life): key = -ln(u) / w,
                   u a seeded hash of the parent id, smallest key first; all first epochs come before any
                   second epoch. Caps per period, as shares of the period's seen tokens: German <= 25 %,
                   English books <= 12 %, legal (case law, Federal Register) <= 10 % of the period's English
                   (1930s also <= 0.35B). A capped class takes its units in key order until its cap; the
                   period takes accepted units in key order until its target. Caps depend on the period
                   total, so the plan is iterated to a fixed point.
  science          per language (EN 0.85B, DE 0.30B, but never above 10 % of the language's seen tokens,
                   i.e. 1/9 of its general seen tokens), outside the period caps and recency weighting; tiers
                   from config/mixture.yaml in order, uniform random order inside a tier, a second epoch only
                   if every first epoch fits.

Output data/mixture/<lang>/<sha12>.parquet (article_id, parent_id, source, lang, period, category, year,
n_words, tokens, count) for every row with count >= 1, and data/mixture/MANIFEST.json (parameters,
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
                        ("year", pa.int16()), ("n_words", pa.int32()), ("tokens", pa.float32()),
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


def take_in_order(tokens: np.ndarray, key: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    """Indices of the `mask` units, smallest key first, whose cumulative tokens stay within `amount`."""
    idx = np.flatnonzero(mask)
    order = idx[np.argsort(key[idx], kind="stable")]
    return order[np.cumsum(tokens[order]) <= amount]


def _candidates(t: pa.Table) -> np.ndarray:
    return np.asarray(t.column("train_ok").to_numpy(zero_copy_only=False), dtype=bool) & \
        (np.asarray(t.column("split").to_numpy(zero_copy_only=False)) == "train")


def load_meta(cfg: dict, mcfg: dict) -> dict:
    """Candidate rows of both languages as numeric arrays (string columns as codes into meta["vocab"],
    parent ids as their seeded hash), plus the per-file candidate counts for the write pass."""
    vocab = {k: {} for k in CODED}
    cols = defaultdict(list)
    files, inputs = [], []
    real = mcfg.get("token_source", "words") == "tokenized"
    if real:            # n_tokens per filtered row from src.data.tokenize_corpus, + 1 <|endoftext|> per document
        troot = repo_path(cfg["tokenized"])
        tm = json.loads((troot / "MANIFEST.json").read_text(encoding="utf-8"))
        inputs.append(("tokenized", str(troot / "MANIFEST.json"),
                       hashlib.sha256((troot / "MANIFEST.json").read_bytes()).hexdigest()))
    for li, lang in enumerate(LANGS):
        root = repo_path(cfg["filtered"]) / lang
        m = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        for o in sorted(m["outputs"].values(), key=lambda o: o["file"]):
            t = pq.read_table(root / o["file"], columns=META)
            if real:
                tok = tm["outputs"][f"{lang}/{o['file']}"]
                if tok.get("filtered_sha256") != o["sha256"]:
                    raise SystemExit(f"{lang}/{o['file']}: tokenized from another version; rerun tokenize_corpus")
                n_tok = pq.read_table(troot / tok["index"], columns=["n_tokens"]).column("n_tokens")
                t = t.append_column("n_tokens", n_tok)
            keep = _candidates(t)
            files.append((lang, root / o["file"], int(keep.sum())))
            if not keep.any():
                continue
            t = t.filter(pa.array(keep))
            h, u = uniform(t.column("parent_id").to_numpy(zero_copy_only=False), mcfg["seed"])
            # chunks of one item can land in both language pools: a parent is per language
            cols["h"].append(h ^ np.uint64(li * 0x9E3779B97F4A7C15))
            cols["u"].append(u)
            for k in CODED:
                d = t.column(k).combine_chunks().dictionary_encode()
                lut = np.array([vocab[k].setdefault(v, len(vocab[k])) for v in d.dictionary.to_pylist()],
                               dtype=np.int16)
                cols[k].append(lut[d.indices.to_numpy(zero_copy_only=False)])
            cols["year"].append(t.column("year").to_numpy(zero_copy_only=False).astype(np.int16))
            cols["n_words"].append(t.column("n_words").to_numpy(zero_copy_only=False).astype(np.int64))
            if real:
                cols["n_tok"].append(t.column("n_tokens").to_numpy(zero_copy_only=False).astype(np.int64) + 1)
            cols["lang"].append(np.full(t.num_rows, li, dtype=np.int8))
        inputs.append((lang, str(root / "MANIFEST.json"),
                       hashlib.sha256((root / "MANIFEST.json").read_bytes()).hexdigest()))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out.update(inputs=inputs, files=files, vocab=vocab)
    if real:
        out["tokens"] = out.pop("n_tok").astype(np.float64)
    else:
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
    src_names = names(meta, "source")
    share = float(mcfg["caps"]["science_of_language"])

    def plan_period(per: str, target: float) -> None:
        m = (~p_sci) & (p_per == code(meta, "period", per))
        second = (p_year[m] >= mcfg["repeat_from_year"]) if per == "1930-39.06" else np.zeros(int(m.sum()), dtype=bool)
        count[m] = plan(p_tok[m], cls[m], key[m], second, target, mcfg["caps"],
                        float(mcfg["caps"].get("legal_absolute", {}).get(per, math.inf)))

    def plan_science() -> None:
        summary["science_target"] = {}
        for li, lang in enumerate(LANGS):
            general = float((p_tok * count)[(~p_sci) & (p_lang == li)].sum())
            limit = share / (1 - share) * general                  # rule 7: <= 10 % of the language
            target = min(float(prof["science"][lang]), limit)
            summary["science_target"][lang] = {"configured": float(prof["science"][lang]), "used": target,
                                               "limit_from_share": limit}
            m = p_sci & (p_lang == li)
            k = science_key(src_names[p_src[m]], p_u[m], mcfg["science_tiers"][lang])
            count[m] = plan(p_tok[m], np.full(int(m.sum()), FREE, dtype=np.int8), k,
                            np.full(int(m.sum()), mcfg["max_epochs"] >= 2), target)

    targets = {per: float(prof["periods"][per]) for per in PERIODS}
    for per in PERIODS:
        plan_period(per, targets[per])
    plan_science()
    fill = prof.get("fill_to_budget")          # user 2026-10-08: a shortfall after the caps goes to this period
    if fill:
        ceiling = float(prof.get("fill_max_share", 1.0)) * float(prof["budget"])
        seen = float((p_tok * count).sum())
        for _ in range(30):
            gap = float(prof["budget"]) - seen
            if abs(gap) < 1e6 or targets[fill] >= ceiling:
                break
            targets[fill] = min(ceiling, targets[fill] + gap)
            plan_period(fill, targets[fill])
            plan_science()
            new_seen = float((p_tok * count).sum())
            if abs(new_seen - seen) < 1e6:                 # the period is exhausted
                break
            seen = new_seen
    summary["period_targets_used"] = targets

    # The English-only twin (user, 2026-10-09): same seen tokens per period as the main run, the German
    # slice replaced by English of the same period under the same repetition rule; no third epoch.
    twin = np.where(p_lang == de, 0, count).astype(np.int8)
    tw = {"replaced_german": {}, "shortfall": {}}
    for per in PERIODS:
        m = (~p_sci) & (p_per == code(meta, "period", per))
        german = float((p_tok * count)[m & (p_lang == de)].sum())
        total = float((p_tok * count)[m].sum())
        tw["replaced_german"][per] = german
        if per == "1930-39.06":           # a second epoch of 1930-33 English (legal stays at its absolute cap)
            cand = m & (p_lang == en) & (count == 1) & (p_year < mcfg["repeat_from_year"]) & (cls != LEGAL)
            twin[take_in_order(p_tok, key, cand, german)] = 2
        else:                              # the period redrawn without German: extra English, same race order
            me = m & (p_lang == en)
            twin[me] = plan(p_tok[me], cls[me], key[me], np.zeros(int(me.sum()), dtype=bool), total, mcfg["caps"],
                            float(mcfg["caps"].get("legal_absolute", {}).get(per, math.inf)))
        tw["shortfall"][per] = total - float((p_tok * twin)[m].sum())
    ms = p_sci & (p_lang == en)
    sci_main = float((p_tok * count)[p_sci].sum())
    sci_target = min(sci_main, share / (1 - share) * float((p_tok * twin)[~p_sci].sum()))
    k = science_key(src_names[p_src[ms]], p_u[ms], mcfg["science_tiers"]["en"])
    twin[ms] = plan(p_tok[ms], np.full(int(ms.sum()), FREE, dtype=np.int8), k,
                    np.full(int(ms.sum()), mcfg["max_epochs"] >= 2), sci_target)
    tw["science_target"] = sci_target
    tw["shortfall"]["science"] = sci_main - float((p_tok * twin)[ms].sum())
    summary["twin"] = tw
    return count[inv], twin[inv], summary


def tables(meta: dict, count: np.ndarray, mcfg: dict, profile: str, targets: dict) -> dict:
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
        checks[p] = {"seen": float(d["seen"].sum()), "target": targets[p],
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
        tk = meta["tokens"][pos:pos + n]
        pos += n
        sel = c > 0
        if not sel.any():
            continue
        t = pq.read_table(path, columns=["article_id", "parent_id", "source", "period", "category", "bucket",
                                         "year", "n_words", "split", "train_ok"])
        t = t.filter(pa.array(_candidates(t))).filter(pa.array(sel))
        per = np.where(t.column("bucket").to_numpy(zero_copy_only=False) == "science", "science",
                       t.column("period").to_numpy(zero_copy_only=False))
        tbl = pa.table({"article_id": t.column("article_id"), "parent_id": t.column("parent_id"),
                        "source": t.column("source"), "lang": pa.array([lang] * t.num_rows),
                        "period": pa.array(per.astype(str)), "category": t.column("category"),
                        "year": t.column("year").cast(pa.int16()), "n_words": t.column("n_words").cast(pa.int32()),
                        "tokens": pa.array(tk[sel].astype(np.float32)),
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
        for old in tmp.parent.glob("*.parquet"):          # this stage's own earlier draws
            if old != final:
                old.unlink()
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
    count, twin, summary = draw(meta, mcfg, profile)
    tab = tables(meta, count, mcfg, profile, summary["period_targets_used"])
    print_plan(tab)
    tab_twin = tables(meta, twin, mcfg, profile, {p: tab["period_seen"].get(p, 0.0) for p in PERIODS})
    print("  --- twin (English only, compute-matched)", flush=True)
    print_plan(tab_twin)
    print("  twin: " + json.dumps(summary["twin"]), flush=True)
    if args.dry_run:
        return
    out_root = repo_path(cfg["mixture"])
    common = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "profile": profile,
              "params": mcfg, "inputs": [{"lang": l, "manifest": p, "sha256": s} for l, p, s in meta["inputs"]],
              "token_source": ("n_tokens from data/tokenized (+1 <|endoftext|> per document)"
                               if mcfg.get("token_source") == "tokenized" else
                               "n_words x tokens_per_word (placeholder until the BPE exists)")}
    outputs = write(meta, count, out_root, mcfg)
    manifest = {"stage": "mixture", **common, "outputs": outputs, **{k: v for k, v in summary.items() if k != "twin"},
                **tab, "seconds": (dt.datetime.now() - t0).seconds}
    (out_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    twin_out = write(meta, twin, out_root / "twin", mcfg)
    manifest_twin = {"stage": "mixture_twin", **common, "rule": "compute-matched English-only twin (user, 2026-10-09)",
                     "main_manifest_sha256": hashlib.sha256((out_root / "MANIFEST.json").read_bytes()).hexdigest(),
                     "outputs": twin_out, **summary["twin"], **tab_twin}
    (out_root / "MANIFEST_twin.json").write_text(json.dumps(manifest_twin, indent=1), encoding="utf-8")
    print(f"-> {out_root}", flush=True)


if __name__ == "__main__":
    main()
