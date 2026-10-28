"""Per-year bpb of the four baselines (reports/<run>/bpb_by_year.json), English scored and German reported.

Lines are the per-year bpb, bands the bootstrap 95 % CI over documents; the dashed rule is the cutoff.

    python scripts/make_baselines_figure.py --out reports/figures/baselines_v1_bpb_by_year   # .pdf and .png
    python scripts/make_baselines_figure.py --with-main --out reports/figures/main_vs_baselines_bpb_by_year
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# Fixed categorical order (one hue per model, never by rank); checked for colour-vision deficiency.
RUNS = [("gru_50m", "GRU", "#2a78d6"), ("lstm_50m", "LSTM", "#eb6834"),
        ("rnn_50m", "RNN", "#1baf7a"), ("ngram_kn5", "KN 5-gram", "#eda100")]
MAIN = ("main_s335m", "Transformer", "#e87ba4")
INK, MUTED, GRID, SURFACE = "#222222", "#5f5f5a", "#e4e3df", "#fcfcfb"
CUTOFF = 1939.5        # 1939-06-30


def load(run: str) -> dict[str, list[dict]]:
    rows = json.loads((REPO / "reports" / run / "bpb_by_year.json").read_text())["rows"]
    return {lang: sorted((r for r in rows if r["lang"] == lang), key=lambda r: r["year"]) for lang in ("en", "de")}


def spread(ys: list[float], gap: float) -> list[float]:
    """Label heights at least `gap` apart, each as close to its line end as the others allow."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = [0.0] * len(ys)
    prev = -1e9
    for i in order:
        out[i] = max(ys[i], prev + gap)
        prev = out[i]
    return out


def panel(ax, data: dict[str, dict], lang: str, title: str, last_year: int, runs: list) -> None:
    ends = []
    for run, label, color in runs:
        rows = data[run][lang]
        x = [r["year"] for r in rows]
        ax.fill_between(x, [r["ci_lo"] for r in rows], [r["ci_hi"] for r in rows], color=color, alpha=0.18, lw=0)
        ax.plot(x, [r["bpb"] for r in rows], color=color, lw=2, label=label, solid_capstyle="round")
        ends.append((x[-1], rows[-1]["bpb"], label))
    lo, hi = ax.get_ylim()
    for (x, _, label), y in zip(ends, spread([e[1] for e in ends], 0.055 * (hi - lo))):
        ax.annotate(label, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.axvline(CUTOFF, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(CUTOFF + 0.6, hi - 0.02 * (hi - lo), "cutoff\n1939-06-30", va="top", fontsize=8, color=MUTED)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlim(1899, last_year + 7)
    ax.set_xlabel("publication year", color=MUTED, fontsize=9)
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
    ap.add_argument("--out", default="reports/figures/baselines_v1_bpb_by_year")
    ap.add_argument("--with-main", action="store_true", help="add the main Transformer run")
    args = ap.parse_args()
    out = REPO / args.out
    runs = [MAIN] + RUNS if args.with_main else RUNS
    data = {run: load(run) for run, _, _ in runs}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [63, 53]})
    panel(axes[0], data, "en", "English (American Stories, scored)", 1955, runs)
    panel(axes[1], data, "de", "German (DDB / Europeana, reported)", 1945, runs)
    axes[0].set_ylabel("bits per byte (lower is better)", color=MUTED, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(runs), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE)
    print(out.with_suffix(".png"))


if __name__ == "__main__":
    main()
