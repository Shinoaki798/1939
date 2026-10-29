"""Before / after evaluation of the fine-tuned copy of the main model (user 2026-10-22 and 10-29; CLAUDE.md
rules 1, 8; TASKS Phase 6; config/finetune.yaml).

The untuned main model (config/finetune.yaml base) and the fine-tuned run are scored on the same frozen items
(src.data.qa_pairs; the pairs file's sha256 must be listed in probes/CHECKSUMS):
    period_pre    held-out period pairs <= cutoff      answer bits per byte given the question
    period_post   period pairs >= 1939-09-01           the same: the knowledge boundary
    caqa_test     ChroniclingAmericaQA test sample     answer bits per byte given passage + question, and exact
                                                       match / token F1 of the greedy answer (SQuAD normalisation)
Each model reads <|endoftext|> prompt (`qa_pairs.prompt_text`, the column layout the untuned model was trained
on) and is scored on ' ' + answer, without the closing token. Greedy answers stop at <|endoftext|> or a line
break. Paired differences (before - after) with bootstrap 95 % CIs over items. Also: Val bpb (EN, DE) of both
models, and the greedy answers of both to samples_per_set items of each set and to the fixed questions in
probes/finetune_prompts.txt.

    python -m src.eval.finetune_eval --run ft_main_s335m_qa
Output: reports/finetune/<run>.json, <run>.md and <run>_samples.md.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import re
import string
import subprocess

import numpy as np
import torch
import yaml

from src.data.download import load_config, repo_path
from src.data.qa_pairs import encode, load_pairs, prompt_text
from src.eval.crosslingual import target_nll, token_bytes
from src.model.build import build
from src.train.train import ValSlice

SETS = ("period_pre", "period_post", "caqa_test")
PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = "".join(ch for ch in s.lower() if ch not in PUNCT)
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


def em_f1(pred: str, ref: str) -> tuple[float, float]:
    p, r = normalize_answer(pred).split(), normalize_answer(ref).split()
    common = sum((collections.Counter(p) & collections.Counter(r)).values())
    f1 = 0.0 if common == 0 else 2 * common / (len(p) + len(r))
    return float(p == r), f1


def paired(before: np.ndarray, after: np.ndarray, weight: np.ndarray | None, n_boot: int, seed: int) -> dict:
    """Ratio of sums (bits per byte when weight is bytes; a mean when weight is None) of both models and of their
    difference before - after, with 95 % CIs from resampling items."""
    w = np.ones(len(before)) if weight is None else weight
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(before), (n_boot, len(before)))
    f = lambda v, i: v[i].sum(-1) / w[i].sum(-1)                            # noqa: E731
    ci = lambda a: [round(float(x), 5) for x in np.percentile(a, [2.5, 97.5])]   # noqa: E731
    return {"before": round(float(before.sum() / w.sum()), 5), "before_ci": ci(f(before, idx)),
            "after": round(float(after.sum() / w.sum()), 5), "after_ci": ci(f(after, idx)),
            "diff": round(float((before - after).sum() / w.sum()), 5), "diff_ci": ci(f(before - after, idx)),
            "items": int(len(before))}


@torch.no_grad()
def generate(model, prompts: list, device: str, max_new: int, rows: int, stop: np.ndarray) -> list[list[int]]:
    """Greedy continuation of each prompt (after <|endoftext|>) until a stop token or max_new tokens."""
    out = []
    for a in range(0, len(prompts), rows):
        seqs = [[0] + list(p) for p in prompts[a:a + rows]]
        new, done = [[] for _ in seqs], [False] * len(seqs)
        for _ in range(max_new):
            x = torch.zeros(len(seqs), max(len(s) for s in seqs), dtype=torch.long)
            for r, s in enumerate(seqs):
                x[r, :len(s)] = torch.tensor(s)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                h = model(x.to(device))
            last = torch.stack([h[r, len(s) - 1] for r, s in enumerate(seqs)]).to(model.output_weight.dtype)
            nxt = (last @ model.output_weight.t()).argmax(-1).tolist()
            for r, t in enumerate(nxt):
                if done[r]:
                    continue
                if stop[t]:
                    done[r] = True
                else:
                    seqs[r].append(t)
                    new[r].append(t)
            if all(done):
                break
        out += new
    return out


def load_model(run_dir, ck_name: str, device: str):
    run = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    model = build(run["model"]).to(device).eval()
    ck = torch.load(run_dir / "checkpoints" / ck_name, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    return model, int(ck["step"])


def frozen(sha: str) -> bool:
    lines = repo_path("probes/CHECKSUMS").read_text(encoding="utf-8").splitlines()
    return any(ln.split()[:1] == [sha] and "data/finetune/" in ln for ln in lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="runs/<name>/ of the fine-tuned model")
    ap.add_argument("--checkpoint", help="file in runs/<name>/checkpoints/ (default: latest)")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    cfg = load_config(repo_path("config/paths.yaml"))
    fcfg = yaml.safe_load(repo_path("config/finetune.yaml").read_text(encoding="utf-8"))
    e = fcfg["eval"]
    tok = Tokenizer.from_file(str(repo_path(yaml.safe_load(repo_path("config/tokenizer.yaml").read_text())["file"])))
    df, sha = load_pairs(cfg)
    if not frozen(sha):
        raise SystemExit(f"pairs file sha256 {sha[:12]} is not in probes/CHECKSUMS: freeze the items first (rule 9)")
    ctx_tok = int(fcfg["data"]["caqa"]["context_tokens"])
    items = {}
    for s in SETS:
        d = df[df["set"] == s].reset_index(drop=True)
        enc = [encode(tok, r.question, r.answer, r.context if isinstance(r.context, str) else None, ctx_tok)
               for r in d.itertuples()]
        keep = [i for i, x in enumerate(enc) if x is not None]
        items[s] = {"df": d.iloc[keep].reset_index(drop=True), "prompts": [enc[i][0] for i in keep],
                    "answers": [np.asarray(enc[i][1]) for i in keep]}
    rng = np.random.default_rng(int(fcfg["seed"]))
    for s in SETS:
        n = len(items[s]["df"])
        items[s]["sample"] = sorted(rng.choice(n, size=min(int(e["samples_per_set"]), n), replace=False).tolist())
    questions = [q.strip() for q in repo_path("probes/finetune_prompts.txt").read_text(encoding="utf-8").splitlines()
                 if q.strip() and not q.startswith("#")]
    fixed = [tok.encode(prompt_text(q), add_special_tokens=False).ids for q in questions]
    tb = token_bytes(tok)
    stop = np.array([i == 0 or "\n" in tok.decode([i]) for i in range(tok.get_vocab_size())])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = repo_path(cfg["runs"])
    tuned_ck = args.checkpoint or (runs / args.run / "checkpoints" / "latest.txt").read_text(encoding="utf-8").strip()
    models = {"before": (runs / fcfg["base"]["run"], fcfg["base"]["checkpoint"]), "after": (runs / args.run, tuned_ck)}
    shards = repo_path(cfg["shards"])
    got = {}
    for which, (rd, ck) in models.items():
        model, step = load_model(rd, ck, device)
        g = {"step": step, "nll": {}, "gen": {}}
        for s in SETS:
            it = items[s]
            g["nll"][s] = np.array([x.sum() for x in target_nll(model, it["prompts"], it["answers"], device,
                                                                 int(e["batch_rows"]), sep=[])])
            pick = list(range(len(it["df"]))) if s == "caqa_test" else it["sample"]
            key = "caqa" if s.startswith("caqa") else "period"
            ids = generate(model, [it["prompts"][i] for i in pick], device, int(e["gen_tokens"][key]),
                           int(e["batch_rows"]), stop)
            g["gen"][s] = dict(zip(pick, [tok.decode(x).strip() for x in ids]))
        g["gen"]["fixed"] = [tok.decode(x).strip() for x in generate(model, fixed, device, int(e["gen_tokens"]["period"]),
                                                                     int(e["batch_rows"]), stop)]
        C = int(yaml.safe_load((rd / "config.yaml").read_text(encoding="utf-8"))["model"]["context"])
        g["val"] = {lang: round(ValSlice(shards / "val", lang, C).evaluate(model, model.output_weight, device, 1)["bpb"], 5)
                    for lang in ("en", "de") if (shards / "val" / f"{lang}.bin").exists()}
        got[which] = g
        print(which, step, g["val"], flush=True)
        del model
        torch.cuda.empty_cache()
    nb, seed = int(e["bootstrap"]), int(fcfg["seed"])
    res = {"run": args.run, "checkpoint": tuned_ck, "base": fcfg["base"], "pairs_sha256": sha,
           "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                     cwd=repo_path(".")).stdout.strip(),
           "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "val_bpb": {w: got[w]["val"] for w in got}, "sets": {}}
    for s in SETS:
        nbytes = np.array([tb[a].sum() for a in items[s]["answers"]])
        r = {"answer_bpb": paired(got["before"]["nll"][s] / math.log(2), got["after"]["nll"][s] / math.log(2),
                                  nbytes, nb, seed)}
        if s == "caqa_test":
            refs = items[s]["df"]["answer"].tolist()
            sc = {w: np.array([em_f1(got[w]["gen"][s][i], refs[i]) for i in range(len(refs))]) for w in got}
            r["exact_match"] = paired(sc["before"][:, 0], sc["after"][:, 0], None, nb, seed)
            r["f1"] = paired(sc["before"][:, 1], sc["after"][:, 1], None, nb, seed)
        res["sets"][s] = r
    out = repo_path("reports") / "finetune"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.run}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    lines = [f"# Fine-tuning before / after: `{args.run}` ({tuned_ck}) vs `{fcfg['base']['run']}` "
             f"({fcfg['base']['checkpoint']})", "", f"Items: pairs file `{sha[:12]}` (frozen in probes/CHECKSUMS).", "",
             "Answer bits per byte (lower is better); diff = before - after (> 0: the fine-tuned model is better).", "",
             "| set | items | before | 95 % CI | after | 95 % CI | diff | 95 % CI |", "|---|---|---|---|---|---|---|---|"]
    for s in SETS:
        a = res["sets"][s]["answer_bpb"]
        lines.append(f"| {s} | {a['items']} | {a['before']} | {a['before_ci']} | {a['after']} | {a['after_ci']} | "
                     f"{a['diff']} | {a['diff_ci']} |")
    c = res["sets"]["caqa_test"]
    lines += ["", "ChroniclingAmericaQA test, greedy answer (higher is better; gain = after - before):", "",
              "| metric | before | after | gain | 95 % CI |", "|---|---|---|---|---|"]
    for m in ("exact_match", "f1"):
        x = c[m]
        lines.append(f"| {m} | {x['before']} | {x['after']} | {round(-x['diff'], 5)} | "
                     f"{[-x['diff_ci'][1], -x['diff_ci'][0]]} |")
    lines += ["", f"Val bpb: before {res['val_bpb']['before']}, after {res['val_bpb']['after']}."]
    (out / f"{args.run}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sm = [f"# Greedy answers before / after fine-tuning: `{args.run}`", ""]
    sm += ["## Fixed questions", ""]
    for q, b, a in zip(questions, got["before"]["gen"]["fixed"], got["after"]["gen"]["fixed"]):
        sm += [f"**Q. {q}**", "", f"- before: {b or '(nothing)'}", f"- after: {a or '(nothing)'}", ""]
    for s in SETS:
        sm += [f"## {s}", ""]
        d = items[s]["df"]
        for i in items[s]["sample"]:
            r = d.iloc[i]
            if isinstance(r.context, str):
                sm += [f"> {re.sub(chr(10), ' ', r.context[:400])} ..."]
            sm += ["", f"**Q. {r.question}** ({r.date})", "", f"- reference: {r.answer}",
                   f"- before: {got['before']['gen'][s][i] or '(nothing)'}",
                   f"- after: {got['after']['gen'][s][i] or '(nothing)'}", ""]
    (out / f"{args.run}_samples.md").write_text("\n".join(sm) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
