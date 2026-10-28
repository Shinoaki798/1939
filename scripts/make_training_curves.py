"""Training loss and Val bpb of a run against a baseline trained on the same stream (same token order).

Left: training loss (nats/token, rolling median over 25 log lines) by tokens seen. Right: Val bpb, English
solid and German dashed, with the baseline's final Val as thin reference lines.

    python scripts/make_training_curves.py --run main_s335m --compare gru_50m \
        --out reports/figures/main_s335m_training_curves          # writes .pdf and .png
    --log PATH reads a copy of the run's log.jsonl (e.g. fetched from the 5080 while it trains)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BATCH_TOKENS = 262_144
COLORS = {"gru_50m": "#2a78d6", "lstm_50m": "#eb6834", "rnn_50m": "#1baf7a", "main_s335m": "#e87ba4"}
LABELS = {"gru_50m": "GRU", "lstm_50m": "LSTM", "rnn_50m": "RNN", "main_s335m": "Transformer"}
INK, MUTED, GRID, SURFACE = "#222222", "#5f5f5a", "#e4e3df", "#fcfcfb"


def read_log(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def rolling_median(y: np.ndarray, w: int) -> np.ndarray:
    return np.array([np.median(y[max(0, i - w + 1): i + 1]) for i in range(len(y))])


def style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("tokens seen (billions)", color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_facecolor(SURFACE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main_s335m")
    ap.add_argument("--log", help="log.jsonl to read instead of runs/<run>/log.jsonl")
    ap.add_argument("--compare", default="gru_50m")
    ap.add_argument("--out", default="reports/figures/main_s335m_training_curves")
    args = ap.parse_args()
    run = read_log(Path(args.log) if args.log else REPO / "runs" / args.run / "log.jsonl")
    base = read_log(REPO / "runs" / args.compare / "log.jsonl")
    last = max(r["step"] for r in run if "loss" in r)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)

    for name, rows, lim in ((args.compare, base, last), (args.run, run, last)):
        tr = [r for r in rows if "loss" in r and r["step"] <= lim]
        x = np.array([r["step"] for r in tr]) * BATCH_TOKENS / 1e9
        y = rolling_median(np.array([r["loss"] for r in tr]), 25)
        a.plot(x, y, color=COLORS[name], lw=2, label=LABELS[name])
        a.annotate(f"{LABELS[name]} {y[-1]:.2f}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                   va="center", fontsize=9, color=INK)
    a.set_ylim(2.8, 6.5)
    a.set_ylabel("training loss (nats/token, rolling median)", color=MUTED, fontsize=9)
    style(a, f"Training loss, same stream, first {last * BATCH_TOKENS / 1e9:.2f}B tokens")

    val = [r for r in run if "val_en_bpb" in r]
    x = np.array([r["step"] for r in val]) * BATCH_TOKENS / 1e9
    final = [r for r in base if "val_en_bpb" in r][-1]
    for key, ls, lang in (("val_en_bpb", "-", "English"), ("val_de_bpb", (0, (5, 3)), "German")):
        y = np.array([r[key] for r in val])
        b.plot(x, y, color=COLORS[args.run], lw=2, ls=ls, marker="o", ms=4)
        b.annotate(f"{LABELS[args.run]} {lang} {y[-1]:.3f}", (x[-1], y[-1]), xytext=(6, 0),
                   textcoords="offset points", va="center", fontsize=9, color=INK)
        b.axhline(final[key], color=COLORS[args.compare], lw=1, ls=ls)
        b.text(0.45, final[key] + 0.012, f"{LABELS[args.compare]} final {lang} {final[key]:.3f} (12B tokens)",
               fontsize=8, color=MUTED, ha="center", transform=b.get_yaxis_transform())
    b.set_xlim(0, x[-1] * 1.45)
    b.set_ylabel("Val bits per byte (lower is better)", color=MUTED, fontsize=9)
    style(b, f"Val bpb every 500 steps (step {val[-1]['step']:,} of 45,776)")

    fig.tight_layout()
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE)
    print(out.with_suffix(".png"))


if __name__ == "__main__":
    main()
