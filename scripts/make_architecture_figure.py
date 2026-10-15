"""Architecture figure of the main Transformer: every layer with its size and parameter count.

One row, drawn at 11 in wide like docs/figures/overview.pdf and scaled to the text width in the proposal.

    python scripts/make_architecture_figure.py --out docs/figures/architecture   # writes .pdf and .png
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

V, D, L, HEADS, FF, CTX = 49152, 1024, 24, 16, 2688, 4096       # config/model_ladder.yaml (s335m), tokenizer
EMB = V * D
ATTN = 4 * D * D
FFN = 3 * D * FF
BLOCK = ATTN + FFN + 2 * D
TOTAL = EMB + L * BLOCK + D

W, H = 11.0, 1.62
INK, MUTED, GROUP = "#222222", "#4a4a4a", "#b07a3c"
FILL = {"emb": "#dbe8f5", "norm": "#e6e6e6", "attn": "#d9ecd9", "ffn": "#e8dff2"}


def m(n: int) -> str:
    return f"{n / 1e6:.1f}M"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="path without extension")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "pdf.fonttype": 42})
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    cy, bh = 0.60, 0.80                       # centre line and height of the layer boxes
    skip_y, gtop, gbot = 1.12, H - 0.04, 0.04

    def box(x, w, title, lines, kind):
        ax.add_patch(FancyBboxPatch((x, cy - bh / 2), w, bh, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=FILL[kind], ec=INK, lw=0.8))
        n = len(lines)
        ax.text(x + w / 2, cy + 0.11 * n, title, ha="center", va="center", fontsize=11, weight="bold", color=INK)
        for i, s in enumerate(lines):
            ax.text(x + w / 2, cy + 0.11 * n - 0.235 * (i + 1), s, ha="center", va="center", fontsize=10,
                    color=MUTED)
        return x + w

    def arrow(x0, x1, y=cy):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=9, lw=0.8, color=INK,
                                     shrinkA=0, shrinkB=0))

    def plus(cx):
        ax.add_patch(Circle((cx, cy), 0.11, fc="white", ec=INK, lw=0.8))
        ax.text(cx, cy - 0.01, "+", ha="center", va="center", fontsize=11, color=INK)

    # token ids in
    ax.text(0.33, cy + 0.10, f"{CTX:,}\ntoken ids", ha="center", va="bottom", fontsize=10, color=MUTED,
            linespacing=1.0)
    arrow(0.02, 0.66)
    x = box(0.66, 1.45, "Token embedding", [f"{V:,} × {D:,}", f"{m(EMB)} ({EMB / TOTAL:.0%})"], "emb")

    # decoder block, repeated 24 times
    g0 = x + 0.26
    arrow(x, g0 + 0.20)
    res1 = g0 + 0.10
    x = box(g0 + 0.20, 0.86, "RMSNorm", [f"{D:,}"], "norm")
    arrow(x, x + 0.16)
    x = box(x + 0.16, 1.62, "Causal self-attention", [f"{HEADS} heads × {D // HEADS}, RoPE",
                                                       f"Q, K, V, O: {m(ATTN)}"], "attn")
    p1 = x + 0.27
    arrow(x, p1 - 0.11)
    plus(p1)
    res2 = p1 + 0.24
    arrow(p1 + 0.11, res2 + 0.14)
    x = box(res2 + 0.14, 0.86, "RMSNorm", [f"{D:,}"], "norm")
    arrow(x, x + 0.16)
    x = box(x + 0.16, 1.58, "SwiGLU feed-forward", [f"{D:,} → {FF:,} → {D:,}", f"{m(FFN)}"], "ffn")
    p2 = x + 0.27
    arrow(x, p2 - 0.11)
    plus(p2)
    g1 = p2 + 0.24
    for xa, xb in ((res1, p1), (res2, p2)):     # residual paths over the two sub-layers
        ax.plot([xa, xa, xb], [cy, skip_y, skip_y], color=INK, lw=0.8)
        ax.add_patch(FancyArrowPatch((xb, skip_y), (xb, cy + 0.11), arrowstyle="-|>", mutation_scale=9, lw=0.8,
                                     color=INK, shrinkA=0, shrinkB=0))
    ax.add_patch(FancyBboxPatch((g0, gbot), g1 - g0, gtop - gbot, boxstyle="round,pad=0,rounding_size=0.10",
                                fc="none", ec=GROUP, lw=0.9, ls="--"))
    ax.text((g0 + g1) / 2, gtop - 0.17, f"Decoder block × {L}  (pre-norm, residual):  {m(BLOCK)} each,  "
            f"{m(L * BLOCK)} in all ({L * BLOCK / TOTAL:.0%})", ha="center", va="center", fontsize=10.5,
            weight="bold", color=GROUP)

    arrow(p2 + 0.11, g1 + 0.26)
    x = box(g1 + 0.26, W - 0.02 - (g1 + 0.26), "Output layer", [f"RMSNorm, {D:,} → {V:,}",
                                                                "tied to embedding; softmax"], "emb")
    assert x <= W, x
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=200 if ext == "png" else None)
    print(f"-> {args.out}.pdf / .png; total {TOTAL:,} parameters; group {g0:.2f}..{g1:.2f}, output box from {g1 + 0.26:.2f}")


if __name__ == "__main__":
    main()
