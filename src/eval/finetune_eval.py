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
models with paired CIs over Val documents, and the greedy answers of both to samples_per_set items of each set
and to the fixed questions in probes/finetune_prompts.txt. Every per-item score of both models is saved, so
two fine-tuned runs can be compared on the same items (--compare).

    python -m src.eval.finetune_eval --run ft_main_s335m_qa
    python -m src.eval.finetune_eval --compare ft_main_s335m_qa ft_main_s335m_qa_replay
Output: reports/finetune/<run>.json, <run>.md, <run>_samples.md and <run>_items.npz; --compare writes
reports/finetune/compare_<a>_vs_<b>.{json,md} (diff = a - b of the after scores; in bits per byte > 0: b is
better).
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


def compare(a: str, b: str, n_boot: int, seed: int) -> dict:
    """Two fine-tuned runs on the same items: their before scores must agree (same base model, same items);
    paired differences a - b of the after scores, with bootstrap 95 % CIs over items."""
    out = repo_path("reports") / "finetune"
    za, zb = np.load(out / f"{a}_items.npz"), np.load(out / f"{b}_items.npz")
    if sorted(za.files) != sorted(zb.files):
        raise SystemExit(f"{a} and {b} were scored on different sets")
    res = {"a": a, "b": b, "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                                      cwd=repo_path(".")).stdout.strip(), "scores": {}}
    lines = [f"# Fine-tuned runs compared: `{a}` (a) vs `{b}` (b)", "",
             "Same frozen items and base model (the before scores of both evaluations agree). diff = a - b of the "
             "after scores, 95 % CI over items (Val: documents): in bits per byte > 0 means b is better, in "
             "exact match and F1 < 0 means b is better.", "",
             "| score | items | base | a | b | diff a - b | 95 % CI |", "|---|---|---|---|---|---|---|"]
    for k in sorted(f[:-len("_before")] for f in za.files if f.endswith("_before")):
        if not np.allclose(za[f"{k}_before"], zb[f"{k}_before"], rtol=1e-4, atol=1e-3):
            raise SystemExit(f"{k}: the before scores differ between {a} and {b}")
        w = za[f"{k}_weight"] if f"{k}_weight" in za.files else None
        r = paired(za[f"{k}_after"], zb[f"{k}_after"], w, n_boot, seed)
        base = za[f"{k}_before"].sum() / (w.sum() if w is not None else len(za[f"{k}_before"]))
        res["scores"][k] = {"base": round(float(base), 5), "a": r["before"], "a_ci": r["before_ci"], "b": r["after"],
                            "b_ci": r["after_ci"], "diff": r["diff"], "diff_ci": r["diff_ci"], "items": r["items"]}
        lines.append(f"| {k} | {r['items']} | {round(float(base), 5)} | {r['before']} | {r['after']} | {r['diff']} | "
                     f"{r['diff_ci']} |")
    (out / f"compare_{a}_vs_{b}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / f"compare_{a}_vs_{b}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="runs/<name>/ of the fine-tuned model")
    ap.add_argument("--checkpoint", help="file in runs/<name>/checkpoints/ (default: latest)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two evaluated runs on the same items")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    cfg = load_config(repo_path("config/paths.yaml"))
    fcfg = yaml.safe_load(repo_path("config/finetune.yaml").read_text(encoding="utf-8"))
    e = fcfg["eval"]
    if args.compare:
        compare(*args.compare, int(e["bootstrap"]), int(fcfg["seed"]))
        return
    if not args.run:
        ap.error("--run or --compare is required")
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
        g["val"] = {lang: ValSlice(shards / "val", lang, C).evaluate(model, model.output_weight, device, 1, per_doc=True)
                    for lang in ("en", "de") if (shards / "val" / f"{lang}.bin").exists()}
        got[which] = g
        print(which, step, {lang: round(v["bpb"], 5) for lang, v in g["val"].items()}, flush=True)
        del model
        torch.cuda.empty_cache()
    nb, seed = int(e["bootstrap"]), int(fcfg["seed"])
    res = {"run": args.run, "checkpoint": tuned_ck, "base": fcfg["base"], "pairs_sha256": sha,
           "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                     cwd=repo_path(".")).stdout.strip(),
           "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "val_bpb": {}, "sets": {}}
    arrays = {}                                       # every per-item score, for --compare
    for lang in got["before"]["val"]:
        b, a = got["before"]["val"][lang], got["after"]["val"][lang]
        res["val_bpb"][lang] = paired(b["doc_bits"], a["doc_bits"], b["doc_bytes"], nb, seed)
        arrays.update({f"val_{lang}_before": b["doc_bits"], f"val_{lang}_after": a["doc_bits"],
                       f"val_{lang}_weight": b["doc_bytes"]})
    for s in SETS:
        nbytes = np.array([tb[a].sum() for a in items[s]["answers"]])
        bits = {w: got[w]["nll"][s] / math.log(2) for w in got}
        r = {"answer_bpb": paired(bits["before"], bits["after"], nbytes, nb, seed)}
        arrays.update({f"{s}_before": bits["before"], f"{s}_after": bits["after"], f"{s}_weight": nbytes})
        if s == "caqa_test":
            refs = items[s]["df"]["answer"].tolist()
            sc = {w: np.array([em_f1(got[w]["gen"][s][i], refs[i]) for i in range(len(refs))]) for w in got}
            r["exact_match"] = paired(sc["before"][:, 0], sc["after"][:, 0], None, nb, seed)
            r["f1"] = paired(sc["before"][:, 1], sc["after"][:, 1], None, nb, seed)
            for m, j in (("exact_match", 0), ("f1", 1)):
                arrays.update({f"{s}_{m}_before": sc["before"][:, j], f"{s}_{m}_after": sc["after"][:, j]})
        res["sets"][s] = r
    out = repo_path("reports") / "finetune"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.run}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez_compressed(out / f"{args.run}_items.npz", **arrays)
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
    lines += ["", "Val bpb, whole Val slices (lower is better; diff = before - after, CI over documents):", "",
              "| language | documents | before | 95 % CI | after | 95 % CI | diff | 95 % CI |",
              "|---|---|---|---|---|---|---|---|"]
    for lang, a in res["val_bpb"].items():
        lines.append(f"| {lang} | {a['items']} | {a['before']} | {a['before_ci']} | {a['after']} | {a['after_ci']} | "
                     f"{a['diff']} | {a['diff_ci']} |")
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
