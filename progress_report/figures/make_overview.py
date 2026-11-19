"""Draw progress_report/figures/overview.{pdf,png}: input -> model -> output, and the date split.

    python progress_report/figures/make_overview.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent
RED, GREEN, BLUE, INK, GREY = "#c0504d", "#5b9a68", "#4f6fa8", "#222222", "#888888"
FILL = {"in": "#eef3ea", "tok": "#f1f1f1", "model": "#f3ede3", "out": "#e9eef6"}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10.5,
                     "pdf.fonttype": 42})

fig = plt.figure(figsize=(8.6, 2.0))           # close to the 5.5 in text width, so the text stays legible
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 110)
ax.set_ylim(0, 20.5)
ax.axis("off")


def box(x, y, w, h, title, lines, fill, edge=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fill, ec=edge, lw=1.1))
    ax.text(x + w / 2, y + h - 1.15, title, ha="center", va="top", fontweight="bold", fontsize=10.5)
    ax.text(x + w / 2, y + h - 3.8, "\n".join(lines), ha="center", va="top", fontsize=9.4, linespacing=1.2)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.1, color=INK))


top, h = 7.3, 12.6
box(0.6, top, 24.4, h, "Input: text to 30 June 1939",
    ["US newspapers, German", "newspapers in the original,", "books, case law, science:", "12.0B tokens seen",
     "(English 9.0B, German 3.0B)"], FILL["in"])
box(27.8, top, 13.4, h, "Tokenizer", ["byte-level BPE,", "49,152 tokens,", "trained on the", "same text"], FILL["tok"])
box(44.0, top, 22.6, h, "Transformer", ["decoder-only, 349M", "parameters, 24 layers,", "random initialisation,",
                                         "predicts the next token"], FILL["model"])
box(69.4, top, 40.0, h, "Output: p(next token | text)", [""], FILL["out"])
arrow(25.25, top + h / 2, 27.55, top + h / 2)
arrow(41.45, top + h / 2, 43.75, top + h / 2)
arrow(66.85, top + h / 2, 69.15, top + h / 2)
uses = [("bits per byte of\nheld-out text,\nyear by year", "where\nknowledge ends"),
        ("forced choice\nbetween\nanswers", "what it knows\n(quiz, RQ1, RQ3)"),
        ("fine-tuned\ncopy answers\nquestions", "Q. ...?  A. ...")]
for i, (a, c) in enumerate(uses):
    x = 70.2 + i * 13.2
    ax.add_patch(FancyBboxPatch((x, top + 0.7), 12.2, 8.4, boxstyle="round,pad=0.2,rounding_size=0.6", fc="white",
                                ec=BLUE, lw=0.9))
    ax.text(x + 6.1, top + 8.5, a, ha="center", va="top", fontsize=8.6, linespacing=1.1)
    ax.text(x + 6.1, top + 1.0, c, ha="center", va="bottom", fontsize=8.4, style="italic", color=BLUE,
            linespacing=1.05)

# date split: x from 1900 to 1955 over [6, 104]
X = lambda yr: 6 + (yr - 1900) / 55 * 98          # noqa: E731
y0, hh = 2.2, 2.6
ax.add_patch(Rectangle((X(1900), y0), X(1939.5) - X(1900), hh, fc=GREEN, ec="none", alpha=0.85))
ax.add_patch(Rectangle((X(1939.5), y0), X(1939.67) - X(1939.5), hh, fc=GREY, ec="none"))
ax.add_patch(Rectangle((X(1939.67), y0), X(1955) - X(1939.67), hh, fc=RED, ec="none", alpha=0.85))
ax.text(X(1919.5), y0 + hh / 2, "TRAIN  1900 ... 1939-06-30  (2 % of every year held out)",
        ha="center", va="center", color="white", fontsize=9.2, fontweight="bold")
ax.text(X(1947.5), y0 + hh / 2, "TEST only  1939-09 ... 1955", ha="center", va="center", color="white", fontsize=9.2,
        fontweight="bold")
ax.annotate("embargo Jul-Aug 1939: never trained or scored", (X(1939.58), y0), (X(1939.58) - 1.0, 0.25),
            ha="right", va="bottom", fontsize=8.6, color=GREY, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
for yr in (1900, 1920, 1939, 1955):
    ax.text(X(yr) + (0.0 if yr != 1939 else 0.6), y0 + hh + 0.35, str(yr), ha="center", va="bottom", fontsize=8.6,
            color=INK)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"overview.{ext}", dpi=220)
print("wrote", OUT / "overview.pdf")
