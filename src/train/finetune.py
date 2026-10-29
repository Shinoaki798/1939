"""Fine-tune a copy of the finished main model on question-answer pairs (user 2026-10-22 and 10-29; HANDOFF §12;
CLAUDE.md rule 1; config/finetune.yaml; TASKS Phase 6).

Examples (src.data.qa_pairs): period_train and caqa_train, each as prompt (`qa_pairs.prompt_text`), answer,
<|endoftext|>. They are packed greedily into rows of 4,096 tokens, an example never crossing a row (the rest of
a row is <|endoftext|> padding); a row starts with <|endoftext|>, as every pretraining document follows one.
The loss is on the answer and its closing <|endoftext|> only, averaged over those tokens in each step. Every
epoch packs the examples in a fresh seeded order; step s reads rows [s*B, (s+1)*B), so the position in the data
is the step. Same optimiser, warmup + cosine schedule, bf16, clipping and checkpoints as src.train.train; the
dev loss (period_dev + caqa_dev, nats per scored token) is logged every dev_every_steps. The base checkpoint is
read, never written.

    python -m src.train.finetune --name ft_main_s335m_qa
    python -m src.train.finetune --name ft_main_s335m_qa --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.download import load_config, repo_path
from src.data.qa_pairs import encode, load_pairs
from src.model.build import build
from src.model.loss import loss_sum
from src.train.train import gpu_status, lr_at, save_checkpoint, write_provenance

TRAIN_SETS, DEV_SETS = ("period_train", "caqa_train"), ("period_dev", "caqa_dev")


def examples(df, sets, tok, context_tokens: int) -> tuple[list, int]:
    """(prompt ids, answer ids) of every pair in sets; also the count dropped because a cut passage lost the answer."""
    out, dropped = [], 0
    for r in df[df["set"].isin(sets)].itertuples():
        e = encode(tok, r.question, r.answer, r.context if isinstance(r.context, str) else None, context_tokens)
        if e is None:
            dropped += 1
        else:
            out.append(e)
    return out, dropped


def pack(exs: list, C: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Inputs (n, C) and targets (n, C) of packed rows, targets -1 except the answers and their closing
    <|endoftext|>; also the number of examples longer than a row (skipped)."""
    rows, skipped = [], 0
    t, s, pos = np.zeros(C + 1, np.int64), np.zeros(C + 1, bool), 1
    for p, a in exs:
        n = len(p) + len(a) + 1
        if n > C:
            skipped += 1
            continue
        if pos + n > C + 1:
            rows.append((t, s))
            t, s, pos = np.zeros(C + 1, np.int64), np.zeros(C + 1, bool), 1
        t[pos:pos + len(p)] = p
        t[pos + len(p):pos + len(p) + len(a)] = a
        s[pos + len(p):pos + n] = True                     # the answer and the <|endoftext|> after it (t is 0 there)
        pos += n
    if pos > 1:
        rows.append((t, s))
    x = np.stack([t[:-1] for t, _ in rows]).astype(np.uint16)
    y = np.stack([np.where(s[1:], t[1:], -1) for t, s in rows]).astype(np.int32)
    return x, y, skipped


def epochs(exs: list, C: int, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    xs, ys, info = [], [], []
    for e in range(n):
        order = np.random.default_rng(seed + e).permutation(len(exs))
        x, y, skipped = pack([exs[i] for i in order], C)
        xs.append(x)
        ys.append(y)
        info.append({"epoch": e, "rows": len(x), "scored_tokens": int((y >= 0).sum()), "skipped_too_long": skipped})
    return np.concatenate(xs), np.concatenate(ys), info


@torch.no_grad()
def dev_loss(fwd, weight, x: np.ndarray, y: np.ndarray, device: str, rows: int) -> float:
    tot, n = 0.0, 0
    for i in range(0, len(x), rows):
        xb = torch.from_numpy(x[i:i + rows].astype(np.int64)).to(device)
        yb = torch.from_numpy(y[i:i + rows].astype(np.int64)).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            tot += float(loss_sum(fwd(xb), weight, yb))
        n += int((yb >= 0).sum())
    return tot / max(1, n)


def finetune(run: dict, run_dir: Path, base_state: dict | None, x: np.ndarray, y: np.ndarray, dev: tuple,
             device: str, resume: bool, stop_after: int | None = None) -> dict:
    torch.manual_seed(run["seed"])
    model = build(run["model"]).to(device)
    cuda = device == "cuda"
    fwd = torch.compile(model) if cuda and run["model"]["type"] == "transformer" else model
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": run["weight_decay"]}, {"params": rest, "weight_decay": 0.0}],
                            lr=run["lr"], betas=tuple(run["betas"]), fused=cuda)
    step = 0
    if resume:
        name = (run_dir / "checkpoints" / "latest.txt").read_text(encoding="utf-8").strip()
        ck = torch.load(run_dir / "checkpoints" / name, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        step = int(ck["step"])
        print(f"resumed from {name} at step {step:,}", flush=True)
    else:
        model.load_state_dict(base_state)
    B, micro = run["batch_rows"], run["micro_batch_rows"]
    max_steps = len(x) // B
    if B % micro or max_steps == 0:
        raise SystemExit(f"batch {B} rows / micro-batch {micro}, or too few rows ({len(x):,})")
    log = open(run_dir / "log.jsonl", "a", encoding="utf-8")
    if step == 0:
        model.eval()
        rec = {"step": 0, "dev_loss": round(dev_loss(fwd, model.output_weight, *dev, device, micro), 5)}
        model.train()
        log.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    t_last, last = time.time(), {}
    while step < max_steps:
        lr = lr_at(step, max_steps, run)
        for g in opt.param_groups:
            g["lr"] = lr
        xs = torch.from_numpy(x[step * B:(step + 1) * B].astype(np.int64))
        ys = torch.from_numpy(y[step * B:(step + 1) * B].astype(np.int64))
        n_scored = int((ys >= 0).sum())
        total = torch.zeros((), device=device)
        for m in range(B // micro):
            xb = xs[m * micro:(m + 1) * micro].to(device, non_blocking=True)
            yb = ys[m * micro:(m + 1) * micro].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cuda):
                loss = loss_sum(fwd(xb), model.output_weight, yb, compile=cuda) / max(1, n_scored)
            loss.backward()
            total += loss.detach()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), run["grad_clip"]))
        loss_v = float(total)
        if not (math.isfinite(loss_v) and math.isfinite(gnorm)):
            log.write(json.dumps({"step": step, "error": "non-finite loss or gradient", "loss": loss_v,
                                  "grad_norm": gnorm}) + "\n")
            log.close()
            raise SystemExit(f"step {step}: non-finite loss {loss_v} or grad norm {gnorm}; nothing saved")
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        rec = {}
        if step % 10 == 0 or step == max_steps:
            now = time.time()
            rec = {"step": step, "loss": round(loss_v, 5), "lr": lr, "grad_norm": round(gnorm, 4),
                   "scored_tokens": n_scored, "s_per_step": round((now - t_last) / 10, 2) if step % 10 == 0 else None,
                   "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            t_last = now
        if step % run["log_gpu_temp_every_steps"] == 0:
            rec.update(gpu_status())
        if step % run["dev_every_steps"] == 0 or step == max_steps:
            model.eval()
            rec["dev_loss"] = round(dev_loss(fwd, model.output_weight, *dev, device, micro), 5)
            model.train()
        if step % run["checkpoint_every_steps"] == 0 or step == max_steps:
            rec["checkpoint"] = save_checkpoint(run_dir, model, opt, step).name
        if rec:
            rec.setdefault("step", step)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            last = rec
            print(json.dumps(rec), flush=True)
        if stop_after and step >= stop_after:
            break
    log.close()
    return {"step": step, "max_steps": max_steps, "last": last}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name: runs/<name>/")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", type=int, help="stop after this step (resume tests)")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    cfg = load_config(repo_path("config/paths.yaml"))
    fcfg = yaml.safe_load(repo_path("config/finetune.yaml").read_text(encoding="utf-8"))
    run_dir = repo_path(cfg["runs"]) / args.name
    if args.resume:
        run = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    elif (run_dir / "checkpoints").exists():
        raise SystemExit(f"{run_dir} already has checkpoints; use --resume or another --name")
    else:
        b = fcfg["base"]
        base_run = yaml.safe_load((repo_path(cfg["runs"]) / b["run"] / "config.yaml").read_text(encoding="utf-8"))
        run = {"name": args.name, "seed": int(fcfg["seed"]), "base": dict(b), "model": base_run["model"],
               "data": fcfg["data"], **fcfg["train"]}
    tcfg = yaml.safe_load(repo_path("config/tokenizer.yaml").read_text(encoding="utf-8"))
    tok = Tokenizer.from_file(str(repo_path(tcfg["file"])))
    df, pairs_sha = load_pairs(cfg)
    C, ctx_tok = int(run["model"]["context"]), int(run["data"]["caqa"]["context_tokens"])
    train_ex, dropped = examples(df, TRAIN_SETS, tok, ctx_tok)
    dev_ex, dev_dropped = examples(df, DEV_SETS, tok, ctx_tok)
    x, y, info = epochs(train_ex, C, int(run["epochs"]), int(run["seed"]))
    dx, dy, _ = pack(dev_ex, C)
    print(f"{len(train_ex):,} train examples ({dropped} dropped: answer cut from the passage), {len(x):,} rows, "
          f"{len(x) // run['batch_rows']:,} steps; dev {len(dev_ex):,} examples, {len(dx):,} rows", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    base_state = None
    if not args.resume:
        ck_path = repo_path(cfg["runs"]) / run["base"]["run"] / "checkpoints" / run["base"]["checkpoint"]
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        base_state = ck["model"]
        h = hashlib.sha256()
        with open(ck_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        write_provenance(run_dir, run, {
            "pairs_file": json.loads((repo_path(cfg["data_root"]) / "finetune" / "MANIFEST.json").read_text())["file"],
            "pairs_sha256": pairs_sha, "base_checkpoint": str(ck_path.relative_to(repo_path("."))),
            "base_checkpoint_step": int(ck["step"]), "base_checkpoint_sha256": h.hexdigest(),
            "train_examples": len(train_ex), "train_dropped_answer_cut": dropped, "dev_examples": len(dev_ex),
            "dev_dropped_answer_cut": dev_dropped, "epochs": info, "dev_rows": len(dx),
            "dev_scored_tokens": int((dy >= 0).sum())})
    out = finetune(run, run_dir, base_state, x, y, (dx, dy), device, args.resume, args.stop_after)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
