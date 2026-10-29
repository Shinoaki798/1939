"""Diagnostic for cross-lingual v2 (not a decision input): where in the target does the matched-vs-mismatched
German effect sit, and does an on-topic German context keep the model expecting German? Per target-position bin:
effects on copy-free tokens (as the primary metric) and the probability mass on German-marked tokens (an umlaut,
or >= 3 times in the German contexts and never in the English text of the items).

    python scripts/crosslingual_diag.py main_s335m gru_50m      # -> reports/crosslingual/diag_v2_position.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenizers import Tokenizer  # noqa: E402

from src.data.download import load_config, repo_path  # noqa: E402
from src.eval.crosslingual import CONDITIONS, contexts, effects, load_items, per_item, scored, token_bytes  # noqa: E402
from src.model.build import build  # noqa: E402

cfg = load_config(repo_path("config/paths.yaml"))
ecfg = yaml.safe_load(repo_path("config/eval.yaml").read_text(encoding="utf-8"))
c = ecfg["crosslingual"]
tok = Tokenizer.from_file(str(repo_path(yaml.safe_load(repo_path("config/tokenizer.yaml").read_text())["file"])))
items, sha = load_items(cfg, ecfg, tok)
ctx = contexts(items)
targets = [np.asarray(t) for t in items["target"]]
V = tok.get_vocab_size()
umlaut = np.array([any(ch in tok.decode([i]) for ch in "äöüßÄÖÜ") for i in range(V)])
de_n = np.bincount(np.concatenate([np.asarray(x) for x in items["de_ctx"]]), minlength=V)
en_n = np.bincount(np.concatenate([np.asarray(x) for x in items["en_ctx"]] + targets), minlength=V)
german = umlaut | ((de_n >= 3) & (en_n == 0))
print(f"German-marked tokens: {int(german.sum())} (umlaut {int(umlaut.sum())})", flush=True)
gm = torch.from_numpy(german)
tb = token_bytes(tok)
nbytes = [tb[t] for t in targets]
topics = items["topic"].to_numpy()
novel = scored(items, ctx, "novel")
bins = [(0, 4), (4, 16), (16, 64)]
SEP = tok.encode(c["joins"]["v2"], add_special_tokens=False).ids


@torch.no_grad()
def run_cond(model, ctxs, device, rows=16):
    nll, mass = [], []
    for a in range(0, len(targets), rows):
        seqs = [np.concatenate([[0], cc, SEP, t]) for cc, t in zip(ctxs[a:a + rows], targets[a:a + rows])]
        L = max(len(s) for s in seqs) - 1
        x = torch.zeros(len(seqs), L, dtype=torch.long)
        for r, s in enumerate(seqs):
            x[r, :len(s) - 1] = torch.from_numpy(s[:-1].astype(np.int64))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = model(x.to(device))
        for r, (s, cc) in enumerate(zip(seqs, ctxs[a:a + rows])):
            k = 1 + len(cc) + len(SEP)
            lp = torch.log_softmax((h[r, k - 1:len(s) - 1].to(model.output_weight.dtype) @ model.output_weight.t()).float(), -1)
            y = torch.from_numpy(s[k:].astype(np.int64)).to(device)
            nll.append((-lp.gather(1, y[:, None])[:, 0]).double().cpu().numpy())
            mass.append(lp.exp()[:, gm.to(device)].sum(1).double().cpu().numpy())
    return nll, mass


out = {}
for run in sys.argv[1:]:
    run_dir = repo_path(cfg["runs"]) / run
    rc = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    ck_name = (run_dir / "checkpoints" / "latest.txt").read_text(encoding="utf-8").strip()
    model = build(rc["model"]).to("cuda").eval()
    model.load_state_dict(torch.load(run_dir / "checkpoints" / ck_name, map_location="cuda", weights_only=False)["model"])
    nll, mass = {}, {}
    for k in CONDITIONS:
        nll[k], mass[k] = run_cond(model, ctx[k], "cuda")
    res = {}
    for a, b in bins:
        masks = [m & (np.arange(len(m)) >= a) & (np.arange(len(m)) < b) for m in novel]
        e = effects(per_item(nll, masks, nbytes), topics, int(ecfg["bootstrap"]), int(ecfg["seed"]))
        res[f"pos{a}-{b}"] = {k: e[k] for k in ("de", "de_ci", "en", "en_ci", "scored_tokens")}
        res[f"pos{a}-{b}"]["german_mass"] = {k: round(float(np.mean([m[a:b].mean() for m in mass[k] if len(m) > a])), 4)
                                            for k in CONDITIONS}
    res["mean_nll_nats"] = {k: round(float(np.mean(np.concatenate(nll[k]))), 4) for k in CONDITIONS}
    out[run] = res
    print(run, json.dumps(res, indent=1), flush=True)
    del model
    torch.cuda.empty_cache()
(repo_path("reports") / "crosslingual" / "diag_v2_position.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
