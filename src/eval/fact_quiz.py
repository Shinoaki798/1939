"""Multiple-choice quiz on the news of 1939 (user 2026-10-29; probes/fact_quiz_1939.csv, frozen in probes/CHECKSUMS
before any model is scored, rule 9).

Each item asks one fact, with the answer true as of the cutoff and three distractors of the same kind (earlier office
holders, rival names, neighbouring places). Tiers by the date the fact became true (`as_of`):
    standing   before 1938-07-01        long established by the cutoff
    recent     1938-07-01 ... 1939-06-30 the last year before the cutoff
    post       >= 1939-09-01            a control: knowable only after the cutoff (the embargo is never asked)
Two formats, both scored for every candidate after <|endoftext|>:
    qa    the American Stories question column: `Q. {question}\\n\\n A.` + ' ' + Candidate + '.'
    stem  a statement: `{stem}` + ' ' + candidate + '.'
A model's choice is the candidate with the lowest bits per byte of its continuation (length-normalised forced
choice, as RQ1-C4 / RQ3); accuracy per model x format x tier with bootstrap 95 % CIs over items, and paired
differences from the untuned main model. Chance is 25 %.

    python -m src.eval.fact_quiz
Output: reports/fact_quiz/fact_quiz_1939.{json,md}, fact_quiz_1939_items.csv (every choice).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.download import load_config, repo_path
from src.data.qa_pairs import prompt_text
from src.eval.crosslingual import target_nll, token_bytes
from src.eval.finetune_eval import load_model, paired

TIERS = {"standing": (None, "1938-07-01"), "recent": ("1938-07-01", "1939-07-01"), "post": ("1939-09-01", None)}
CANDIDATES = ("answer", "distractor_1", "distractor_2", "distractor_3")
FORMATS = ("qa", "stem")


def check(df: pd.DataFrame) -> None:
    """The construction rules, so the frozen file cannot drift from them."""
    if df["id"].duplicated().any():
        raise SystemExit(f"duplicate ids: {df['id'][df['id'].duplicated()].tolist()}")
    for r in df.itertuples():
        if r.tier not in TIERS:
            raise SystemExit(f"{r.id}: tier {r.tier!r}")
        lo, hi = TIERS[r.tier]
        if (lo and r.as_of < lo) or (hi and r.as_of >= hi):
            raise SystemExit(f"{r.id}: as_of {r.as_of} outside tier {r.tier}")
        c = [getattr(r, k) for k in CANDIDATES]
        if any(not isinstance(x, str) or not x.strip() for x in c) or len(set(c)) != 4:
            raise SystemExit(f"{r.id}: candidates must be four distinct strings")
        if not r.question.endswith("?") or r.stem.endswith((".", "?")):
            raise SystemExit(f"{r.id}: a question ends in '?', a stem has no closing punctuation")


def continuation(candidate: str, fmt: str) -> str:
    c = candidate[0].upper() + candidate[1:] if fmt == "qa" else candidate
    return " " + c + ("" if c.endswith(".") else ".")


def prompt(row, fmt: str) -> str:
    return prompt_text(row.question) if fmt == "qa" else row.stem


def choices(bits: np.ndarray, nbytes: np.ndarray) -> np.ndarray:
    """(items, 4) bits and bytes of each candidate -> index of the chosen candidate (lowest bits per byte)."""
    return np.argmin(bits / nbytes, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    from tokenizers import Tokenizer
    ecfg = yaml.safe_load(repo_path("config/eval.yaml").read_text(encoding="utf-8"))
    q = ecfg["fact_quiz"]
    path = repo_path(q["items"])
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    listed = repo_path("probes/CHECKSUMS").read_text(encoding="utf-8").splitlines()
    if not any(ln.split()[:2] == [sha, q["items"]] for ln in listed):
        raise SystemExit(f"{q['items']} (sha256 {sha[:12]}) is not in probes/CHECKSUMS: freeze it first (rule 9)")
    df = pd.read_csv(path, dtype=str)
    check(df)
    tok = Tokenizer.from_file(str(repo_path(yaml.safe_load(repo_path("config/tokenizer.yaml").read_text())["file"])))
    enc = lambda s: np.asarray(tok.encode(s, add_special_tokens=False).ids)   # noqa: E731
    tb = token_bytes(tok)
    jobs = {}                                        # fmt -> (prompts, continuations), item-major, 4 per item
    for fmt in FORMATS:
        ps, cs = [], []
        for r in df.itertuples():
            p = enc(prompt(r, fmt))
            for k in CANDIDATES:
                ps.append(p)
                cs.append(enc(continuation(getattr(r, k), fmt)))
        jobs[fmt] = (ps, cs)
    nbytes = {fmt: np.array([tb[c].sum() for c in cs]).reshape(-1, 4) for fmt, (_, cs) in jobs.items()}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = repo_path(load_config(repo_path("config/paths.yaml"))["runs"])
    got, rows = {}, []
    for m in q["models"]:
        rd = runs / m["run"]
        ck = m.get("checkpoint") or (rd / "checkpoints" / "latest.txt").read_text(encoding="utf-8").strip()
        model, step = load_model(rd, ck, device)
        got[m["name"]] = {"run": m["run"], "checkpoint": ck, "step": step, "correct": {}}
        for fmt, (ps, cs) in jobs.items():
            with torch.no_grad():
                bits = np.array([x.sum() for x in target_nll(model, ps, cs, device, int(q["batch_rows"]), sep=[])])
            bits = bits.reshape(-1, 4) / math.log(2)
            pick = choices(bits, nbytes[fmt])
            got[m["name"]]["correct"][fmt] = (pick == 0).astype(float)
            for i, r in enumerate(df.itertuples()):
                rows.append({"id": r.id, "tier": r.tier, "model": m["name"], "format": fmt,
                             "chosen": getattr(r, CANDIDATES[pick[i]]), "correct": int(pick[i] == 0),
                             "bpb": [round(float(v), 4) for v in bits[i] / nbytes[fmt][i]]})
        print(m["name"], step, {f: round(float(v.mean()), 3) for f, v in got[m["name"]]["correct"].items()}, flush=True)
        del model
        torch.cuda.empty_cache()
    nb, seed = int(ecfg["bootstrap"]), int(ecfg["seed"])
    base = q["models"][0]["name"]
    res = {"items": q["items"], "items_sha256": sha, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds"), "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                                       cwd=repo_path(".")).stdout.strip(),
           "chance": 0.25, "base": base, "models": {}}
    tiers = {t: (df["tier"] == t).to_numpy() for t in ("standing", "recent", "post")}
    tiers["standing+recent"] = tiers["standing"] | tiers["recent"]
    for name, g in got.items():
        out = {k: g[k] for k in ("run", "checkpoint", "step")}
        for fmt in FORMATS:
            for t, sel in tiers.items():
                a = g["correct"][fmt][sel]
                b = got[base]["correct"][fmt][sel]
                r = paired(a, a, None, nb, seed)
                d = paired(a, b, None, nb, seed)              # diff = this model - base
                out[f"{fmt}/{t}"] = {"accuracy": r["before"], "ci": r["before_ci"], "items": r["items"],
                                     "minus_base": d["diff"], "minus_base_ci": d["diff_ci"]}
        res["models"][name] = out
    od = repo_path("reports") / "fact_quiz"
    od.mkdir(parents=True, exist_ok=True)
    (od / "fact_quiz_1939.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    pd.DataFrame(rows).to_csv(od / "fact_quiz_1939_items.csv", index=False)
    lines = [f"# 1939 news quiz: `{q['items']}` (sha256 {sha[:12]}, frozen)", "",
             f"{len(df)} items, 4 candidates each (chance 25 %); choice = lowest bits per byte of the candidate. "
             f"Accuracy with bootstrap 95 % CI over items; minus base = paired difference from `{base}`.", ""]
    for fmt in FORMATS:
        lines += [f"## Format `{fmt}`", "", "| model | " + " | ".join(tiers) + " |", "|---|" + "---|" * len(tiers)]
        for name in got:
            cells = []
            for t in tiers:
                x = res["models"][name][f"{fmt}/{t}"]
                cell = f"{x['accuracy']:.2f} [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}]"
                if name != base:
                    cell += f"; {x['minus_base']:+.2f} [{x['minus_base_ci'][0]:+.2f}, {x['minus_base_ci'][1]:+.2f}]"
                cells.append(cell)
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += ["", "items per tier: " + ", ".join(f"{t} {int(s.sum())}" for t, s in tiers.items()), ""]
    (od / "fact_quiz_1939.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
