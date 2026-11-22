"""Draw progress_report/figures/results.{pdf,png}: (a) Val English bpb during training, (b) English bpb by year.

Drawn at the printed size (5.5 in = the text width): font sizes are the sizes on the page, none below 7.5 pt.

Reads runs/<run>/log.jsonl (val_en_bpb every 500 steps; every run sees 2^18 tokens per step) and
reports/<run>/bpb_by_year.json (bands: bootstrap 95 % CI over documents).

    python progress_report/figures/make_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = [("main_s335m", "Transformer 349M", "#e87ba4"), ("gru_50m", "GRU 50M", "#2a78d6"),
        ("lstm_50m", "LSTM 50M", "#eb6834"), ("rnn_50m", "RNN 50M", "#1baf7a"), ("ngram_kn5", "KN 5-gram", "#eda100")]
TOKENS_PER_STEP = 2 ** 18
INK, GRID = "#222222", "#e4e3df"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "axes.titlesize": 8.5, "axes.titleweight": "bold", "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "axes.linewidth": 0.6})
fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 1.72), gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.2})

for run, label, color in RUNS:
    log = REPO / "runs" / run / "log.jsonl"
    if not log.exists():
        continue
    pts = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()]
    pts = [(p["step"] * TOKENS_PER_STEP / 1e9, p["val_en_bpb"]) for p in pts if "val_en_bpb" in p]
    a.plot([x for x, _ in pts], [y for _, y in pts], color=color, lw=1.2, label=label)
    nudge = {"lstm_50m": 0.06, "gru_50m": -0.06}.get(run, 0.0)
    a.text(12.3, pts[-1][1] + nudge, f"{pts[-1][1]:.3f}", color=color, va="center", fontsize=7.5)
a.set_xlim(0, 15.2)
a.set_xticks([0, 4, 8, 12])
a.set_ylim(1.1, 2.4)
a.set_xlabel("tokens seen (billions)")
a.set_ylabel("Val English bits per byte")
a.set_title("(a) learning curves", loc="left")
a.grid(color=GRID, lw=0.6)

for run, label, color in RUNS:
    rows = json.loads((REPO / "reports" / run / "bpb_by_year.json").read_text(encoding="utf-8"))["rows"]
    en = sorted((r for r in rows if r["lang"] == "en"), key=lambda r: r["year"])
    yrs = [r["year"] for r in en]
    b.fill_between(yrs, [r["ci_lo"] for r in en], [r["ci_hi"] for r in en], color=color, alpha=0.2, lw=0)
    b.plot(yrs, [r["bpb"] for r in en], color=color, lw=1.2, label=label)
b.axvline(1939.5, color=INK, ls="--", lw=0.9)
b.text(1939.0, 2.2, "cutoff\n1939-06-30", fontsize=7.5, va="top", ha="right")
b.set_xlim(1899, 1956)
b.set_xlabel("publication year of the held-out article")
b.set_title("(b) held-out English bpb by year", loc="left")
b.grid(color=GRID, lw=0.6)
handles, labels = b.get_legend_handles_labels()                       # one legend for both panels
fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8, handlelength=1.5,
           columnspacing=1.2, bbox_to_anchor=(0.53, 1.0))
b.set_ylim(1.0, 2.25)
fig.subplots_adjust(left=0.085, right=0.99, bottom=0.215, top=0.755)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"results.{ext}", dpi=220)
print("wrote", OUT / "results.pdf")
