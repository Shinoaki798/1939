"""Draw docs/figures/overview.{pdf,png}: one model, three measurements (hypothesised shapes).

Content follows the reviewer's figure of 2026-10-05 (docs_update.zip); this script
redraws it so that no label overlaps or is clipped.

    python docs/figures/make_overview.py
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent
RED, GREEN, ORANGE, INK, GREY = "#c0504d", "#5b9a68", "#cf963f", "#222222", "#888888"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})

fig = plt.figure(figsize=(11.0, 2.25))
gs = fig.add_gridspec(2, 3, width_ratios=[4.4, 2.9, 2.5], height_ratios=[1, 1],
                      left=0.025, right=0.99, bottom=0.21, top=0.88, wspace=0.3, hspace=1.0)

# (1) one model, one boundary --------------------------------------------------------
ax = fig.add_subplot(gs[:, 0])
ax.set_xlim(1897, 1957)
ax.set_ylim(0, 10)
for s in ("left", "top", "right"):
    ax.spines[s].set_visible(False)
ax.set_yticks([])
ax.set_xticks([1900, 1910, 1920, 1930, 1939, 1945, 1955])
ax.set_title("(1) One model, one boundary", loc="left")
ax.text(1896.5, 9.6, "US press, English\n(American Stories +)", va="top", fontsize=9.5, linespacing=1.0)
ax.text(1896.5, 7.2, "German press, original\n(Europeana, DDB)", va="top", fontsize=9.5, linespacing=1.0)
box = FancyBboxPatch((1923.3, 6.95), 16.4, 1.9, boxstyle="round,pad=0.3", fc="#f3efe2", ec=INK, lw=1.1)
ax.add_patch(box)
ax.text(1931.5, 7.9, "ONE decoder-only\nTransformer", ha="center", va="center", fontsize=10,
        fontweight="bold", linespacing=1.1)
ax.annotate("", xy=(1922.8, 8.4), xytext=(1919.6, 8.95), arrowprops=dict(arrowstyle="->", color=INK))
ax.annotate("", xy=(1922.8, 7.4), xytext=(1919.6, 6.75), arrowprops=dict(arrowstyle="->", color=INK))
ax.annotate("", xy=(1944.0, 7.9), xytext=(1940.3, 7.9), arrowprops=dict(arrowstyle="->", color=INK))
ax.text(1944.6, 7.9, "held-out\nbits/byte\nby year", va="center", fontsize=9.5, linespacing=1.0)

ax.text(1900.5, 4.35, "hypothesised bits/byte: flat, then a step", fontsize=10.5)
xs = [1900 + i * 0.5 for i in range(79)]
ax.plot(xs, [3.45 + 0.07 * math.sin(2 * math.pi * (x - 1900) / 9) for x in xs], color=GREEN, lw=2.2)
ax.plot([1939.05, 1939.6, 1945, 1955], [3.5, 4.75, 4.55, 4.65], color=RED, lw=2.2)

# TRAIN bar shaded by the recency weight exp(-(1939 - year) / 5)
g, white = to_rgb(GREEN), (1.0, 1.0, 1.0)
for y0 in [1900 + 0.25 * i for i in range(156)]:
    w = 0.25 + 0.75 * math.exp(-(1939 - y0) / 5)
    ax.add_patch(Rectangle((y0, 0.3), 0.26, 1.7, ec="none",
                           color=tuple(w * c + (1 - w) * b for c, b in zip(g, white))))
ax.add_patch(Rectangle((1939.7, 0.3), 15.3, 1.7, color=RED, ec="none", alpha=0.9))
ax.text(1919.5, 1.15, "TRAIN  ≤ 1939-06-30\n(recency-weighted)", ha="center", va="center",
        color=INK, fontsize=10.5, fontweight="bold", linespacing=1.05)
ax.text(1947.4, 1.15, "TEST", ha="center", va="center", color="white", fontsize=11.5, fontweight="bold")
ax.annotate("embargo\nJul–Aug 1939", xy=(1939.35, 2.05), xytext=(1948.0, 3.0), fontsize=9.5,
            ha="center", va="center", arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))

# (2) RQ1 ------------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[:, 1])
labels = ["C1 coinage", "C2 compositional", "C3 new sense", "C4 new assoc."]
vals, cols = [0.92, 0.3, 0.57, 0.88], [RED, GREEN, ORANGE, RED]
ax2.barh(range(4), vals, color=cols, height=0.62)
ax2.set_yticks(range(4))
ax2.set_yticklabels(labels)
ax2.invert_yaxis()
ax2.set_xlim(0, 1.05)
ax2.set_xticks([0, 0.92])
ax2.set_xticklabels(["0", r"$\Delta$c"])
ax2.set_xlabel("surprisal − matched control")
ax2.set_title("(2) RQ1: membrane, not wall", loc="left")

# (3) RQ2 ------------------------------------------------------------------------------
ax3 = fig.add_subplot(gs[0, 2])
eps = [0.1, 0.5, 2, 10, 50]
est = [0.32, 0.6, 2.1, 9.5, 52]
ax3.loglog([0.05, 80], [0.05, 80], ls=":", color=GREY, lw=1)
ax3.loglog(eps, est, "-o", color=INK, ms=4, lw=1.4)
ax3.axvline(0.5, color=RED, ls="--", lw=1.1)
ax3.text(0.07, 12, "detection\nlimit", color=RED, fontsize=9.5, linespacing=1.0)
ax3.set_xlim(0.05, 80)
ax3.set_ylim(0.05, 80)
ax3.set_xticks([0.1, 1, 10])
ax3.set_xticklabels(["0.1%", "1%", "10%"])
ax3.set_yticks([0.1, 1, 10])
ax3.set_yticklabels(["0.1", "1", "10"])
ax3.set_ylabel(r"est. $\varepsilon$")
ax3.set_title(r"(3) RQ2: est. vs true $\varepsilon$", loc="left")

# RQ3 ----------------------------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 2])
ax4.bar([0, 1], [0.21, 0.172], color=[GREEN, ORANGE], width=0.55)
ax4.axhline(0.25, color=GREY, ls="--", lw=1)
ax4.axhline(0.235, color=INK, ls=":", lw=1)
ax4.text(1.32, 0.252, "chance", color=GREY, fontsize=9.5, va="bottom", ha="right")
ax4.text(1.32, 0.232, "n-gram", color=INK, fontsize=9.5, va="top", ha="right")
ax4.set_xticks([0, 1])
ax4.set_xticklabels(["US context", "DE context"])
ax4.set_xlim(-0.5, 1.4)
ax4.set_ylim(0.12, 0.27)
ax4.set_yticks([0.15, 0.25])
ax4.set_ylabel("Brier ↓")
ax4.set_title("RQ3: foresight by press", loc="left")

fig.savefig(OUT / "overview.pdf")
fig.savefig(OUT / "overview.png", dpi=100)
print("wrote", OUT / "overview.pdf", OUT / "overview.png")
