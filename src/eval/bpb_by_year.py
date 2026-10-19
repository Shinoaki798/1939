"""Bits per byte by publication year on the held-out documents (TASKS Phase 3; success criterion 2;
CLAUDE.md rules 4, 5, 8; settings in config/eval.yaml).

Set. The `holdout` split (2 % of every year 1900-1955 per language, never trained on), embargo dates
(1939-07-01 .. 1939-08-31) excluded. English: American Stories only, the scored curve (rule 5). German:
DDB and Europeana newspapers, reported but not a success criterion. Per year and language, documents in
seeded hash order until tokens_per_year; built once (data/eval/bpb_by_year/<lang>.parquet) and reused, so
every model is scored on the same documents.

Scoring. Each document on its own: the context starts with <|endoftext|>, then the document's tokens are
predicted one by one. A document longer than the context is scored in windows of `context` tokens that
advance by `stride`; every token is scored once, with at least context - stride tokens of history after the
first window. bpb(year) = total NLL / ln 2 / total UTF-8 bytes of the documents; the 95 % CI comes from
resampling documents (config `bootstrap`). The Kneser-Ney 5-gram is scored the same way through KenLM
(<s> in place of <|endoftext|>).

    python -m src.eval.bpb_by_year --run lstm_50m [--checkpoint step_045776.pt] [--langs en,de]
    python -m src.eval.bpb_by_year --kenlm ngram_kn5
    python -m src.eval.bpb_by_year --build-set                     # only (re)build the document sets
Output: reports/<run>/bpb_by_year.json and .csv (lang, year, documents, tokens, bytes, bpb, ci_lo, ci_hi).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
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

EMBARGO = ("1939-07-01", "1939-08-31")


def build_set(cfg: dict, ecfg: dict, lang: str) -> pd.DataFrame:
    """Held-out documents of one language: article_id, year, bin, offset, n_tokens, n_bytes."""
    troot, froot = repo_path(cfg["tokenized"]), repo_path(cfg["filtered"])
    tman = json.loads((troot / "MANIFEST.json").read_text(encoding="utf-8"))
    spec = ecfg["bpb_by_year"][lang]
    y0, y1 = ecfg["bpb_by_year"]["years"]
    parts = []
    for key, o in sorted(tman["outputs"].items()):
        klang, source = key.split("/")[:2]
        if klang != lang or source not in spec["sources"] or not o["tokens_by_split"].get("holdout"):
            continue
        meta = pq.read_table(froot / key, columns=["split", "date", "year"])
        idx = pq.read_table(troot / o["index"], columns=["article_id", "offset", "n_tokens", "n_bytes"])
        keep = pc.and_(pc.equal(meta.column("split"), "holdout"),
                       pc.invert(pc.and_(pc.greater_equal(meta.column("date"), EMBARGO[0]),
                                         pc.less_equal(meta.column("date"), EMBARGO[1] + "~"))))
        keep = pc.and_(keep, pc.and_(pc.greater_equal(meta.column("year"), y0), pc.less_equal(meta.column("year"), y1)))
        t = idx.filter(keep)
        if t.num_rows:
            d = t.to_pandas()
            d["year"] = meta.filter(keep).column("year").to_numpy()
            d["bin"] = o["bin"]
            parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    _, u = uniform(d["article_id"].to_numpy(dtype=object), int(ecfg["seed"]))
    d = d.assign(u=u).sort_values(["year", "u"]).drop(columns="u")
    cum = d.groupby("year")["n_tokens"].cumsum() - d["n_tokens"]
    return d[cum < float(spec["tokens_per_year"])].reset_index(drop=True)


def load_set(cfg: dict, ecfg: dict, lang: str, rebuild: bool = False) -> tuple[pd.DataFrame, str]:
    path = repo_path(cfg["data_root"]) / "eval" / "bpb_by_year" / f"{lang}.parquet"
    if rebuild or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(build_set(cfg, ecfg, lang), preserve_index=False), path, compression="zstd")
    return pq.read_table(path).to_pandas(), hashlib.sha256(path.read_bytes()).hexdigest()


def document_tokens(d: pd.DataFrame, troot: Path) -> list[np.ndarray]:
    maps: dict[str, np.ndarray] = {}
    out = []
    for b, o, n in zip(d["bin"], d["offset"], d["n_tokens"]):
        if b not in maps:
            maps[b] = np.memmap(troot / b, dtype="<u2", mode="r")
        out.append(np.asarray(maps[b][o:o + n], dtype=np.int64))
    return out


def pieces(docs: list[np.ndarray], context: int, stride: int, eos: int = 0):
    """(document index, inputs, targets with -1 where a token is not scored in this window)."""
    for i, doc in enumerate(docs):
        seq = np.concatenate([[eos], doc])
        n = len(doc)
        if n <= context:
            yield i, seq[:-1], seq[1:]
            continue
        start, scored = 0, 0
        while scored < n:
            y = seq[start + 1:start + context + 1].copy()       # targets: document tokens start .. start+len-1
            x = seq[start:start + len(y)]
            y[:max(0, scored - start)] = -1
            yield i, x, y
            scored = start + len(y)
            start += stride


@torch.no_grad()
def neural_nll(model, docs: list[np.ndarray], context: int, stride: int, batch_tokens: int, device: str) -> np.ndarray:
    nll = np.zeros(len(docs))
    items = sorted(pieces(docs, context, stride), key=lambda p: len(p[1]))
    k = 0
    while k < len(items):
        L = len(items[k][1])
        j = k
        while j < len(items) and (j - k + 1) * max(L, len(items[j][1])) <= batch_tokens:
            L = max(L, len(items[j][1]))
            j += 1
        j = max(j, k + 1)
        x = torch.zeros(j - k, L, dtype=torch.long)
        y = torch.full((j - k, L), -1, dtype=torch.long)
        for r, (_, xi, yi) in enumerate(items[k:j]):
            x[r, :len(xi)], y[r, :len(yi)] = torch.from_numpy(xi), torch.from_numpy(yi)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            h = model(x.to(device))
        per = token_nll(h, model.output_weight, y.to(device)).double().sum(1).cpu().numpy()
        for r, (i, _, _) in enumerate(items[k:j]):
            nll[i] += per[r]
        k = j
    return nll


def kenlm_nll(model_path: Path, docs: list[np.ndarray]) -> np.ndarray:
    import kenlm
    lm = kenlm.Model(str(model_path))
    ln10 = math.log(10)
    return np.array([-sum(s[0] for s in lm.full_scores(" ".join(map(str, d)), bos=True, eos=False)) * ln10
                     for d in docs])


def summarize(d: pd.DataFrame, nll: np.ndarray, lang: str, n_boot: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for year, g in d.assign(nll=nll).groupby("year"):
        a, b = g["nll"].to_numpy(), g["n_bytes"].to_numpy().astype(np.float64)
        bpb = a.sum() / math.log(2) / b.sum()
        s = rng.integers(0, len(g), (n_boot, len(g)))
        boot = a[s].sum(1) / math.log(2) / b[s].sum(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"lang": lang, "year": int(year), "documents": int(len(g)), "tokens": int(g["n_tokens"].sum()),
                     "bytes": int(b.sum()), "bpb": round(float(bpb), 5), "ci_lo": round(float(lo), 5),
                     "ci_hi": round(float(hi), 5)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="runs/<name>/ of a neural model")
    ap.add_argument("--checkpoint", help="file in runs/<name>/checkpoints/ (default: latest)")
    ap.add_argument("--kenlm", help="runs/<name>/ holding model.binary")
    ap.add_argument("--langs", default="en,de")
    ap.add_argument("--build-set", action="store_true")
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    ecfg = yaml.safe_load(repo_path("config/eval.yaml").read_text(encoding="utf-8"))
    langs = args.langs.split(",")
    sets = {lang: load_set(cfg, ecfg, lang, rebuild=args.build_set) for lang in langs}
    for lang, (d, sha) in sets.items():
        print(f"{lang}: {len(d):,} documents, {int(d['n_tokens'].sum()):,} tokens, years {d['year'].min()}-"
              f"{d['year'].max()}, set sha256 {sha[:12]}", flush=True)
    if args.build_set:
        return
    name = args.run or args.kenlm
    run_dir = repo_path(cfg["runs"]) / name
    troot = repo_path(cfg["tokenized"])
    rows, info = [], {"run": name, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                      "sets_sha256": {k: v[1] for k, v in sets.items()}}
    if args.run:
        run = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        ck_name = args.checkpoint or (run_dir / "checkpoints" / "latest.txt").read_text(encoding="utf-8").strip()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build(run["model"]).to(device).eval()
        ck = torch.load(run_dir / "checkpoints" / ck_name, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        info.update(checkpoint=ck_name, step=int(ck["step"]))
        kind = "transformer" if run["model"]["type"] == "transformer" else "recurrent"
        ctx = int(run["model"].get("context", 4096))
        for lang, (d, _) in sets.items():
            nll = neural_nll(model, document_tokens(d, troot), ctx, int(ecfg["bpb_by_year"]["stride"]),
                             int(ecfg["bpb_by_year"]["batch_tokens"][kind]), device)
            rows += summarize(d, nll, lang, int(ecfg["bootstrap"]), int(ecfg["seed"]))
    else:
        for lang, (d, _) in sets.items():
            nll = kenlm_nll(run_dir / "model.binary", document_tokens(d, troot))
            rows += summarize(d, nll, lang, int(ecfg["bootstrap"]), int(ecfg["seed"]))
    out = repo_path(cfg["reports"]) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "bpb_by_year.json").write_text(json.dumps(info | {"rows": rows}, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "bpb_by_year.csv", index=False)
    print(f"-> {out}/bpb_by_year.json", flush=True)


if __name__ == "__main__":
    main()
