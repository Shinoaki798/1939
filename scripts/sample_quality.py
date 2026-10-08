"""Random text windows of the filtered training text per language x source x period, for a quality read
before the mixture is drawn. Only cells with >= --min-words training words; --per-cell windows each.

    python scripts/sample_quality.py --out reports/mixture_samples_2026-10-08.md
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.download import load_config, repo_path

COLS = ["source", "bucket", "period", "split", "train_ok", "date", "n_words", "ocr_hit", "text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", type=int, default=2)
    ap.add_argument("--chars", type=int, default=300)
    ap.add_argument("--min-words", type=float, default=5e6)
    ap.add_argument("--files-per-source", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261008)
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    rng = random.Random(args.seed)
    out = [f"# Quality sample of the filtered training text ({args.per_cell} windows of {args.chars} chars per "
           f"language x source x period with >= {args.min_words / 1e6:.0f}M words; seed {args.seed})", ""]
    for lang in ("en", "de"):
        root = repo_path(cfg["filtered"]) / lang
        m = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        by_source = defaultdict(list)
        for o in m["outputs"].values():
            by_source[o["file"].split("/")[0]].append(root / o["file"])
        cells = {}
        for src, st in m["per_source"].items():
            for key, v in st.items():
                per, sp, kind = (key.split("|") + ["", "", ""])[:3]
                if sp == "train" and kind == "ok_words" and v >= args.min_words:
                    cells[(src, per)] = v
        for src in sorted({s for s, _ in cells}):
            files = by_source.get(src, [])
            picked = rng.sample(files, min(args.files_per_source, len(files)))
            pool = defaultdict(list)
            for f in picked:
                t = pq.read_table(f, columns=COLS)
                t = t.filter(pc.and_(t.column("train_ok"), pc.equal(t.column("split"), "train")))
                for i in rng.sample(range(t.num_rows), min(40, t.num_rows)):
                    per = "science" if t.column("bucket")[i].as_py() == "science" else t.column("period")[i].as_py()
                    pool[per].append({c: t.column(c)[i].as_py() for c in ("date", "n_words", "ocr_hit", "text")})
            for per in sorted(p for s, p in cells if s == src):
                out += [f"## {lang} | {src} | {per} ({cells[(src, per)] / 1e6:,.0f}M train words kept)", ""]
                docs = pool.get(per, []) or pool.get("science", [])
                for d in rng.sample(docs, min(args.per_cell, len(docs))):
                    text = " ".join((d["text"] or "").split())
                    start = rng.randrange(max(1, len(text) - args.chars))
                    hit = "" if d["ocr_hit"] is None else f"{d['ocr_hit']:.2f}"
                    out += [f"- {d['date']} | {d['n_words']:,} words | ocr {hit}: {text[start:start + args.chars]}", ""]
    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
