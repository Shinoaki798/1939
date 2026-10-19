"""Shared training loop for every neural model: the recurrent baselines, the main Transformer and the twin
(TASKS Phases 4-5; CLAUDE.md reproducibility conventions and hardware rules).

    python -m src.train.train --name lstm_50m --model lstm_50m --lr 1e-3            # new run
    python -m src.train.train --name lstm_50m --resume                              # continue it

Settings come from config/train_defaults.yaml, overridden on the command line; the resolved run config is
written to runs/<name>/config.yaml together with data_manifest.json, seed.txt, git_sha.txt and env.txt.
On --resume the stored config is used and the command-line settings are ignored.

Data: step s reads rows [s*R, (s+1)*R) of the stream (R = batch_tokens / context), so the position in the
data is the step itself. Each row of 4,096 tokens is split into micro-batches of micro_batch_rows rows whose
gradients are accumulated. bf16 autocast, AdamW (decay on matrices only), linear warmup then cosine decay
to min_lr_ratio, gradient clipping. Every checkpoint_every_steps the model, the optimiser and the step are
saved atomically to runs/<name>/checkpoints/ (the last two are kept); every eval_every_steps the fixed Val slices
are scored (bits per byte). runs/<name>/log.jsonl gets a line every 10 steps, with the GPU temperature,
power and memory every log_gpu_temp_every_steps. A non-finite loss stops the run before anything is saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from src.data.download import load_config, repo_path
from src.model.build import build, model_spec
from src.model.loss import loss_sum, token_nll


class Stream:
    """Read-only view of a written stream (src.data.stream): contiguous tokens across shards."""

    def __init__(self, root: Path):
        path = root / "MANIFEST.json"
        man = json.loads(path.read_text(encoding="utf-8"))
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.context = int(man["config"]["context"])
        self.rows = int(man["rows"])
        self.maps = [np.memmap(root / s["file"], dtype="<u2", mode="r") for s in man["shards"]]
        self.first = np.array([s["first_token"] for s in man["shards"]] + [man["tokens"]], dtype=np.int64)

    def span(self, a: int, b: int) -> np.ndarray:
        out = np.empty(b - a, dtype=np.int64)
        i, pos = int(np.searchsorted(self.first, a, side="right")) - 1, a
        while pos < b:
            take = min(b, int(self.first[i + 1])) - pos
            lo = pos - int(self.first[i])
            out[pos - a:pos - a + take] = self.maps[i][lo:lo + take]
            pos, i = pos + take, i + 1
        return out

    def batch(self, row: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Rows [row, row+n): inputs and next-token targets, each (n, context)."""
        C = self.context
        t = torch.from_numpy(self.span(row * C, (row + n) * C + 1))
        return t[:-1].view(n, C), t[1:].view(n, C)


class ValSlice:
    """A fixed Val slice (src.data.stream --val): documents back to back, each preceded by <|endoftext|>."""

    def __init__(self, root: Path, lang: str, context: int):
        self.lang, self.context = lang, context
        self.tokens = torch.from_numpy(np.fromfile(root / f"{lang}.bin", dtype="<u2").astype(np.int64))
        idx = pq.read_table(root / f"{lang}.idx.parquet")
        self.off = idx.column("offset").to_numpy().astype(np.int64)
        self.n = idx.column("n_tokens").to_numpy().astype(np.int64)
        self.bytes = idx.column("n_bytes").to_numpy().astype(np.int64)
        self.rows = (len(self.tokens) - 1) // context

    @torch.no_grad()
    def evaluate(self, fwd, weight: torch.Tensor, device: str, rows_per_batch: int) -> dict:
        """Mean loss per target (nats) over whole rows, and bits per byte over the documents the rows cover:
        each document's tokens are predicted from what precedes them in the slice; <|endoftext|> targets
        are not counted in bits per byte."""
        C = self.context
        nll = torch.empty(self.rows * C, dtype=torch.float64)
        for r in range(0, self.rows, rows_per_batch):
            k = min(rows_per_batch, self.rows - r)
            t = self.tokens[r * C:(r + k) * C + 1]
            x, y = t[:-1].view(k, C).to(device), t[1:].view(k, C).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                h = fwd(x)
            nll[r * C:(r + k) * C] = token_nll(h, weight, y).double().view(-1).cpu()
        csum = torch.cat([torch.zeros(1, dtype=torch.float64), nll.cumsum(0)]).numpy()
        end = self.off + self.n - 1                       # one past the last target index of each document
        ok = end <= self.rows * C
        doc = csum[end[ok]] - csum[self.off[ok] - 1]
        return {"loss": float(nll.mean()), "bpb": float(doc.sum() / math.log(2) / self.bytes[ok].sum()),
                "documents": int(ok.sum())}


def lr_at(step: int, max_steps: int, run: dict) -> float:
    if step < run["warmup_steps"]:
        return run["lr"] * (step + 1) / run["warmup_steps"]
    p = (step - run["warmup_steps"]) / max(1, max_steps - run["warmup_steps"])
    return run["lr"] * (run["min_lr_ratio"] + (1 - run["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def gpu_status() -> dict:
    q = "temperature.gpu,power.draw,memory.used,utilization.gpu"
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=10)
        t, p, m, u = (v.strip() for v in r.stdout.strip().split(","))
        return {"gpu_temp_c": int(t), "gpu_power_w": float(p), "gpu_mem_mib": int(m), "gpu_util": int(u)}
    except Exception:  # noqa: BLE001 - no GPU (tests) or nvidia-smi hiccup: log nothing
        return {}


def write_run_files(run_dir: Path, run: dict, stream: Stream, val_root: Path | None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")
    vm = val_root / "MANIFEST.json" if val_root else None
    (run_dir / "data_manifest.json").write_text(json.dumps({
        "stream": run["stream"], "stream_manifest_sha256": stream.sha256, "rows": stream.rows,
        "val_manifest_sha256": hashlib.sha256(vm.read_bytes()).hexdigest() if vm and vm.exists() else None,
    }, indent=1), encoding="utf-8")
    (run_dir / "seed.txt").write_text(f"{run['seed']}\n", encoding="utf-8")
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_path(".")).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                               cwd=repo_path(".")).stdout.strip()
        (run_dir / "git_sha.txt").write_text(sha + (" (uncommitted changes)" if dirty else "") + "\n", encoding="utf-8")
    except OSError:
        (run_dir / "git_sha.txt").write_text("unknown\n", encoding="utf-8")
    env = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "platform": platform.platform()}
    (run_dir / "env.txt").write_text(json.dumps(env, indent=1) + "\n", encoding="utf-8")


def save_checkpoint(run_dir: Path, model, opt, step: int, keep: int = 2) -> Path:
    ck = run_dir / "checkpoints"
    ck.mkdir(exist_ok=True)
    path = ck / f"step_{step:06d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "torch_rng": torch.get_rng_state()}, tmp)
    os.replace(tmp, path)
    (ck / "latest.txt").write_text(path.name + "\n", encoding="utf-8")
    for old in sorted(ck.glob("step_*.pt"))[:-keep]:
        old.unlink()
    return path


def train(run: dict, run_dir: Path, stream: Stream, vals: list[ValSlice], device: str, resume: bool,
          stop_after: int | None = None) -> dict:
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
    C = stream.context
    R = run["batch_tokens"] // C
    micro = run["micro_batch_rows"]
    accum = R // micro
    max_steps = int(run["tokens"] // run["batch_tokens"])
    if R % micro or stream.rows < max_steps * R:
        raise SystemExit(f"batch {R} rows / micro-batch {micro}, or stream too short ({stream.rows:,} rows "
                         f"for {max_steps:,} steps x {R} rows)")
    log = open(run_dir / "log.jsonl", "a", encoding="utf-8")
    t_last, last = time.time(), {}
    while step < max_steps:
        lr = lr_at(step, max_steps, run)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = stream.batch(step * R, R)
        total = torch.zeros((), device=device)
        for m in range(accum):
            xb = x[m * micro:(m + 1) * micro].to(device, non_blocking=True)
            yb = y[m * micro:(m + 1) * micro].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cuda):
                loss = loss_sum(fwd(xb), model.output_weight, yb, compile=cuda) / (micro * C)
            (loss / accum).backward()
            total += loss.detach() / accum
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
                   "tok_s": round(10 * run["batch_tokens"] / (now - t_last)) if step % 10 == 0 else None,
                   "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            t_last = now
        if step % run["log_gpu_temp_every_steps"] == 0:
            rec.update(gpu_status())
        if vals and (step % run["eval_every_steps"] == 0 or step == max_steps):
            model.eval()
            for v in vals:
                r = v.evaluate(fwd, model.output_weight, device, micro if run["model"]["type"] == "transformer" else 16)
                rec.update({f"val_{v.lang}_loss": round(r["loss"], 5), f"val_{v.lang}_bpb": round(r["bpb"], 5)})
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


def resolve(args, defaults: dict, ladder: dict) -> dict:
    run = dict(defaults)
    for k in ("lr", "tokens", "warmup_steps", "micro_batch_rows", "stream", "seed"):
        v = getattr(args, k)
        if v is not None:
            run[k] = v
    run["name"] = args.name
    run["model"] = model_spec(args.model, ladder)
    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name: runs/<name>/")
    ap.add_argument("--model", help="rung or baseline name in config/model_ladder.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--tokens", type=float)
    ap.add_argument("--warmup-steps", dest="warmup_steps", type=int)
    ap.add_argument("--micro-batch-rows", dest="micro_batch_rows", type=int)
    ap.add_argument("--stream")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--stop-after", type=int, help="stop after this step (resume tests)")
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    run_dir = repo_path(cfg["runs"]) / args.name
    if args.resume:
        run = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    else:
        if (run_dir / "checkpoints").exists():
            raise SystemExit(f"{run_dir} already has checkpoints; use --resume or another --name")
        defaults = yaml.safe_load(repo_path("config/train_defaults.yaml").read_text(encoding="utf-8"))
        ladder = yaml.safe_load(repo_path("config/model_ladder.yaml").read_text(encoding="utf-8"))
        run = resolve(args, defaults, ladder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    shards = repo_path(cfg["shards"])
    stream = Stream(shards / run["stream"])
    vals = [ValSlice(shards / "val", lang, stream.context) for lang in ("en", "de") if (shards / "val" / f"{lang}.bin").exists()]
    if not args.resume:
        write_run_files(run_dir, run, stream, shards / "val")
    out = train(run, run_dir, stream, vals, device, args.resume, args.stop_after)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
