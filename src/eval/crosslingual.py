"""Cross-lingual check for the twin decision (user 2026-10-22 and 10-28; HANDOFF §12; CLAUDE.md rules 1, 5, 8).

Question. Does reading a German article make a native English passage on the same topic more probable than
reading a German article on another topic? If the main model links German to English, a German context primes
English text; if it does not, an English-only twin could not show a German-press effect, and is not trained.

Items. Topics with an English and a German pattern (`probes/crosslingual_topics.csv`, matched case-blind). From
the held-out split (never trained on) dated 1930-01-01 .. 1939-06-30, per topic and in seeded hash order: German
documents (DDB, Europeana) and English documents (American Stories) with >= min_matches hits. Every passage
starts at the first hit's line, or ~200 characters before the hit on a long line: the English target is
target_tokens long, a context is context_tokens long and needs min_matches hits of its own; a passage with
fewer than min_letter_share letters and spaces (OCR debris) is skipped. An item is one English target with four contexts:
a German and an English on-topic context (other documents), and the two contexts of an item of another topic
whose text has no hit of the target's topic (mismatched). No document is used twice. Built once to
data/eval/crosslingual/items.parquet; its sha256 and the topic file's are written to probes/CHECKSUMS before
any model is scored on them (rule 9).

Scoring. <|endoftext|> context <|endoftext|> target; per-token NLL of the target. The effect of a language is
the summed NLL(target | mismatched) - NLL(target | matched) over the scored target tokens, in bits per byte of
those tokens. Primary: only target tokens whose id occurs in none of the item's four contexts, so copying a
string from the context cannot count (names such as Danzig are spelt the same in both languages); also
reported: every target token. 95 % CIs by resampling topics (rule 8). Decision: train the twin if the German
effect's CI lies above 0 and the German effect is >= ratio_threshold x the English effect.

    python -m src.eval.crosslingual --build              # build and freeze the items (CPU)
    python -m src.eval.crosslingual --run main_s335m     # score (GPU)
Output: reports/crosslingual/<run>.json and .md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import yaml

from src.data.download import load_config, repo_path
from src.data.mixture import uniform
from src.model.build import build
from src.model.loss import token_nll

CONDITIONS = ("de_match", "de_mis", "en_match", "en_mis")


def load_topics(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("en_pattern", "de_pattern"):
        for p in t[col]:
            re.compile(p)
    return t


def holdout_texts(cfg: dict, lang: str, sources: list[str], d0: str, d1: str) -> pd.DataFrame:
    """Held-out documents of one language dated d0..d1: article_id, date, text."""
    troot, froot = repo_path(cfg["tokenized"]), repo_path(cfg["filtered"])
    tman = json.loads((troot / "MANIFEST.json").read_text(encoding="utf-8"))
    parts = []
    for key, o in sorted(tman["outputs"].items()):
        klang, source = key.split("/")[:2]
        if klang != lang or source not in sources or not o["tokens_by_split"].get("holdout"):
            continue
        t = pq.read_table(froot / key, columns=["article_id", "split", "date", "text"])
        keep = pc.and_(pc.equal(t.column("split"), "holdout"),
                       pc.and_(pc.greater_equal(t.column("date"), d0), pc.less_equal(t.column("date"), d1 + "~")))
        t = t.filter(keep)
        if t.num_rows:
            parts.append(t.select(["article_id", "date", "text"]).to_pandas())
    return pd.concat(parts, ignore_index=True)


def hits(texts, pattern: str) -> np.ndarray:
    arr = texts if isinstance(texts, pa.Array) else pa.array(list(texts), type=pa.large_string())
    return pc.count_substring_regex(arr, pattern, ignore_case=True).to_numpy(zero_copy_only=False)


def passage(text: str, pattern: str, tok, n_tokens: int, back: int = 200) -> list[int]:
    """Up to n_tokens tokens from the first hit's line start, or from `back` characters before the hit (at a word
    start) when the line is longer: whole newspaper pages run for thousands of characters without a break."""
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if m is None:
        return []
    start = text.rfind("\n", 0, m.start()) + 1
    if m.start() - start > back:
        start = m.start() - back
        while start < m.start() and not text[start - 1].isspace():
            start += 1
    return tok.encode(text[start:start + 10 * n_tokens], add_special_tokens=False).ids[:n_tokens]


LETTERS = re.compile(r"[^\W\d_]|\s")


def letter_share(text: str) -> float:
    """Share of letters and white space; OCR debris (punctuation runs, broken glyphs) scores low."""
    return len(LETTERS.findall(text)) / max(1, len(text))


def assign_mismatch(topics: list[str], ok, seed: int) -> list[int]:
    """For each item an item of another topic whose contexts pass ok(i, j), each item lending its contexts once
    (a maximum bipartite matching, augmenting paths in seeded order), so every context is used as often matched as
    mismatched; -1 for an item left without a partner."""
    rng = np.random.default_rng(seed)
    n = len(topics)
    order = [int(j) for j in rng.permutation(n)]
    adj = [[j for j in order if topics[j] != topics[i] and ok(i, j)] for i in range(n)]
    borrower = [-1] * n                     # lender j -> the item i that uses j's contexts as its mismatch

    def augment(i: int, seen: set) -> bool:
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                if borrower[j] == -1 or augment(borrower[j], seen):
                    borrower[j] = i
                    return True
        return False

    for i in order:
        augment(i, set())
    out = [-1] * n
    for j, i in enumerate(borrower):
        if i >= 0:
            out[i] = j
    return out


def build_items(cfg: dict, ecfg: dict, tok) -> pd.DataFrame:
    c = ecfg["crosslingual"]
    topics = load_topics(repo_path(c["topics"]))
    d0, d1 = c["dates"]
    docs = {lang: holdout_texts(cfg, lang, c[lang]["sources"], d0, d1) for lang in ("en", "de")}
    for lang, d in docs.items():
        d["u"] = uniform(d["article_id"].to_numpy(dtype=object), int(ecfg["seed"]))[1]
        print(f"{lang}: {len(d):,} held-out documents {d0} .. {d1}", flush=True)
    arrays = {lang: pa.array(d["text"].tolist(), type=pa.large_string()) for lang, d in docs.items()}
    used: set[str] = set()
    rows = []
    for t in topics.itertuples():
        cand = {}
        for lang, pat in (("en", t.en_pattern), ("de", t.de_pattern)):
            d = docs[lang]
            cand[lang] = d[hits(arrays[lang], pat) >= int(c["min_matches"])].sort_values("u")
        got = {"de": [], "en_ctx": [], "en_tgt": []}
        for r in cand["de"].itertuples():
            ids = passage(r.text, t.de_pattern, tok, int(c["context_tokens"]))
            if r.article_id not in used and len(ids) == int(c["context_tokens"]) and \
                    hits([tok.decode(ids)], t.de_pattern)[0] >= int(c["min_matches"]) and \
                    letter_share(tok.decode(ids)) >= float(c["min_letter_share"]):
                got["de"].append((r.article_id, ids))
                used.add(r.article_id)
            if len(got["de"]) == int(c["items_per_topic"]):
                break
        for r in cand["en"].itertuples():
            if len(got["en_tgt"]) == len(got["de"]) and len(got["en_ctx"]) == len(got["de"]):
                break
            if r.article_id in used:
                continue
            if len(got["en_tgt"]) <= len(got["en_ctx"]) and len(got["en_tgt"]) < len(got["de"]):
                ids = passage(r.text, t.en_pattern, tok, int(c["target_tokens"]))
                if len(ids) >= int(c["min_target_tokens"]) and \
                        letter_share(tok.decode(ids)) >= float(c["min_letter_share"]):
                    got["en_tgt"].append((r.article_id, ids))
                    used.add(r.article_id)
            elif len(got["en_ctx"]) < len(got["de"]):
                ids = passage(r.text, t.en_pattern, tok, int(c["context_tokens"]))
                if len(ids) == int(c["context_tokens"]) and \
                        hits([tok.decode(ids)], t.en_pattern)[0] >= int(c["min_matches"]) and \
                        letter_share(tok.decode(ids)) >= float(c["min_letter_share"]):
                    got["en_ctx"].append((r.article_id, ids))
                    used.add(r.article_id)
        n = min(len(got["de"]), len(got["en_ctx"]), len(got["en_tgt"]))
        for k in range(n):
            rows.append({"topic": t.topic, "target_doc": got["en_tgt"][k][0], "target": got["en_tgt"][k][1],
                         "de_doc": got["de"][k][0], "de_ctx": got["de"][k][1],
                         "en_doc": got["en_ctx"][k][0], "en_ctx": got["en_ctx"][k][1]})
        print(f"  {t.topic:20} de {len(cand['de']):5,}  en {len(cand['en']):6,}  items {n}", flush=True)
    items = pd.DataFrame(rows)
    pats = topics.set_index("topic")
    text = {col: [tok.decode(ids) for ids in items[col]] for col in ("de_ctx", "en_ctx")}

    def clean(i: int, j: int) -> bool:          # j's contexts carry no hit of i's topic, in either language
        p = pats.loc[items.topic[i]]
        return all(hits([text[col][j]], pat)[0] == 0 for col in ("de_ctx", "en_ctx")
                   for pat in (p.en_pattern, p.de_pattern))

    items["mis"] = assign_mismatch(items["topic"].tolist(), clean, int(ecfg["seed"]))
    return items[items["mis"] >= 0].reset_index(drop=True)


def load_items(cfg: dict, ecfg: dict, tok, rebuild: bool = False) -> tuple[pd.DataFrame, str]:
    path = repo_path(cfg["data_root"]) / "eval" / "crosslingual" / "items.parquet"
    if rebuild or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(build_items(cfg, ecfg, tok), preserve_index=False), path,
                       compression="zstd")
    return pq.read_table(path).to_pandas(), hashlib.sha256(path.read_bytes()).hexdigest()


def contexts(items: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    mis = items["mis"].to_numpy()
    return {"de_match": list(items["de_ctx"]), "de_mis": [items["de_ctx"][j] for j in mis],
            "en_match": list(items["en_ctx"]), "en_mis": [items["en_ctx"][j] for j in mis]}


@torch.no_grad()
def target_nll(model, ctxs: list, targets: list, device: str, batch_rows: int) -> list[np.ndarray]:
    """Per-token NLL (nats) of each target after <|endoftext|> context <|endoftext|>."""
    out = []
    for a in range(0, len(targets), batch_rows):
        seqs = [np.concatenate([[0], c, [0], t]) for c, t in zip(ctxs[a:a + batch_rows], targets[a:a + batch_rows])]
        L = max(len(s) for s in seqs) - 1
        x = torch.zeros(len(seqs), L, dtype=torch.long)
        y = torch.full((len(seqs), L), -1, dtype=torch.long)
        for r, (s, c) in enumerate(zip(seqs, ctxs[a:a + batch_rows])):
            x[r, :len(s) - 1] = torch.from_numpy(s[:-1].astype(np.int64))
            y[r, len(c) + 1:len(s) - 1] = torch.from_numpy(s[len(c) + 2:].astype(np.int64))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            h = model(x.to(device))
        per = token_nll(h, model.output_weight, y.to(device)).double().cpu().numpy()
        for r, (s, c) in enumerate(zip(seqs, ctxs[a:a + batch_rows])):
            out.append(per[r, len(c) + 1:len(s) - 1])
    return out


def scored(items: pd.DataFrame, ctx: dict[str, list], mode: str) -> list[np.ndarray]:
    """Boolean mask over each target: every token ('all'), or tokens in none of the four contexts ('novel')."""
    out = []
    for i, t in enumerate(items["target"]):
        t = np.asarray(t)
        if mode == "all":
            out.append(np.ones(len(t), dtype=bool))
        else:
            seen = np.unique(np.concatenate([np.asarray(ctx[k][i]) for k in CONDITIONS]))
            out.append(~np.isin(t, seen))
    return out


def per_item(nll: dict[str, list], masks: list[np.ndarray], nbytes: list[np.ndarray]) -> pd.DataFrame:
    """Per item: NLL saved by the matched context (nats) per language, and bytes of the scored tokens."""
    rows = []
    for i, m in enumerate(masks):
        rows.append({"de": float((nll["de_mis"][i] - nll["de_match"][i])[m].sum()),
                     "en": float((nll["en_mis"][i] - nll["en_match"][i])[m].sum()),
                     "bytes": float(nbytes[i][m].sum()), "tokens": int(m.sum())})
    return pd.DataFrame(rows)


def effects(d: pd.DataFrame, topics: np.ndarray, n_boot: int, seed: int) -> dict:
    """Effects in bits per byte with 95 % CIs from resampling topics; ratio German / English."""
    def point(g):
        b = g["bytes"].sum()
        return g["de"].sum() / math.log(2) / b, g["en"].sum() / math.log(2) / b

    de, en = point(d)
    groups = [d[topics == t] for t in np.unique(topics)]
    sums = np.array([[g["de"].sum(), g["en"].sum(), g["bytes"].sum()] for g in groups])
    rng = np.random.default_rng(seed)
    s = sums[rng.integers(0, len(groups), (n_boot, len(groups)))].sum(1)
    bde, ben = s[:, 0] / math.log(2) / s[:, 2], s[:, 1] / math.log(2) / s[:, 2]
    ratio = np.divide(bde, ben, out=np.full_like(bde, np.nan), where=ben != 0)
    ci = lambda a: [round(float(v), 5) for v in np.nanpercentile(a, [2.5, 97.5])]   # noqa: E731
    return {"de": round(de, 5), "de_ci": ci(bde), "en": round(en, 5), "en_ci": ci(ben),
            "ratio": round(de / en, 4) if en else None, "ratio_ci": ci(ratio),
            "items": int(len(d)), "topics": int(len(groups)), "scored_tokens": int(d["tokens"].sum())}


def token_bytes(tok) -> np.ndarray:
    """UTF-8 bytes of every token (byte-level BPE: one vocabulary character per byte); 0 for specials."""
    v = tok.get_vocab()
    out = np.zeros(max(v.values()) + 1)
    for s, i in v.items():
        out[i] = 0 if s.startswith("<|") else len(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="runs/<name>/ of a neural model")
    ap.add_argument("--checkpoint", help="file in runs/<name>/checkpoints/ (default: latest)")
    ap.add_argument("--build", action="store_true", help="(re)build the items, then stop")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    cfg = load_config(repo_path("config/paths.yaml"))
    ecfg = yaml.safe_load(repo_path("config/eval.yaml").read_text(encoding="utf-8"))
    c = ecfg["crosslingual"]
    tok = Tokenizer.from_file(str(repo_path(yaml.safe_load(repo_path("config/tokenizer.yaml").read_text())["file"])))
    items, sha = load_items(cfg, ecfg, tok, rebuild=args.build)
    topic_sha = hashlib.sha256(repo_path(c["topics"]).read_bytes()).hexdigest()
    print(f"{len(items)} items over {items['topic'].nunique()} topics; items sha256 {sha[:12]}, topics {topic_sha[:12]}")
    if args.build or not args.run:
        return
    run_dir = repo_path(cfg["runs"]) / args.run
    run = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    ck_name = args.checkpoint or (run_dir / "checkpoints" / "latest.txt").read_text(encoding="utf-8").strip()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build(run["model"]).to(device).eval()
    ck = torch.load(run_dir / "checkpoints" / ck_name, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    ctx = contexts(items)
    targets = [np.asarray(t) for t in items["target"]]
    nll = {k: target_nll(model, ctx[k], targets, device, int(c["batch_rows"])) for k in CONDITIONS}
    tb = token_bytes(tok)
    nbytes = [tb[t] for t in targets]
    topics = items["topic"].to_numpy()
    sha_code = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_path(".")).stdout.strip()
    res = {"run": args.run, "checkpoint": ck_name, "step": int(ck["step"]), "git_sha": sha_code,
           "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "items_sha256": sha, "topics_sha256": topic_sha, "ratio_threshold": float(c["ratio_threshold"])}
    tables = {}
    for mode in ("novel", "all"):
        d = per_item(nll, scored(items, ctx, mode), nbytes)
        res[mode] = effects(d, topics, int(ecfg["bootstrap"]), int(ecfg["seed"]))
        tables[mode] = d.assign(topic=topics)
    p = res["novel"]
    res["train_twin"] = bool(p["de_ci"][0] > 0 and p["de"] >= float(c["ratio_threshold"]) * p["en"])
    out = repo_path("reports") / "crosslingual"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.run}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    by_topic = tables["novel"].groupby("topic")[["de", "en", "bytes"]].sum()
    by_topic = (by_topic[["de", "en"]].div(by_topic["bytes"], axis=0) / math.log(2)).round(4)
    lines = [f"# Cross-lingual check, `{args.run}` ({ck_name})", "",
             f"Items `{sha[:12]}`, topics `{topic_sha[:12]}`: {p['items']} items over {p['topics']} topics.", "",
             "| tokens scored | German -> English (bpb) | 95 % CI | English -> English (bpb) | 95 % CI | ratio | 95 % CI |",
             "|---|---|---|---|---|---|---|"]
    for mode, label in (("novel", "not in any context (primary)"), ("all", "all target tokens")):
        e = res[mode]
        lines.append(f"| {label} | {e['de']} | {e['de_ci']} | {e['en']} | {e['en_ci']} | {e['ratio']} | {e['ratio_ci']} |")
    lines += ["", f"Decision (primary): German CI above 0 and ratio >= {c['ratio_threshold']}: "
              f"**{'train the twin' if res['train_twin'] else 'no twin'}**.", "",
              "Per topic, primary tokens (bits per byte saved by the matched context):", "",
              "| topic | German -> English | English -> English |", "|---|---|---|"]
    lines += [f"| {t} | {r.de} | {r.en} |" for t, r in by_topic.iterrows()]
    (out / f"{args.run}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
