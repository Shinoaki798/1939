"""Draw progress_report/figures/overview.{pdf,png}: input -> model -> output, and the date split.

Drawn at the printed size (5.5 in = the text width), so every font size below is the size on the page
(nothing smaller than 7.5 pt; the proposal's figure was marked down for small, crowded text).

    python progress_report/figures/make_overview.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent
RED, GREEN, BLUE, INK, GREY = "#c0504d", "#4f8a5b", "#3f5f9a", "#222222", "#6b6b6b"
FILL = {"in": "#eef3ea", "tok": "#f1f1f1", "model": "#f3ede3", "out": "#e9eef6"}
W, H = 5.5, 1.62
TITLE, BODY, SMALL = 8.5, 8.0, 7.5

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "pdf.fonttype": 42})
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
texts = []                                          # (text, box) pairs checked for overflow below
pairs = []                                          # (left, right) labels on one line, must not touch


def box(x, w, title, lines, fill, y=0.60, h=0.99):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fill, ec=INK, lw=0.8)
    ax.add_patch(patch)
    texts.append((ax.text(x + w / 2, y + h - 0.07, title, ha="center", va="top", fontsize=TITLE,
                          fontweight="bold"), (x, w)))
    texts.append((ax.text(x + w / 2, y + h - 0.26, "\n".join(lines), ha="center", va="top", fontsize=BODY,
                          linespacing=1.15), (x, w)))


def arrow(x0, x1, y=0.60 + 0.99 / 2):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=8, lw=0.8, color=INK,
                                 shrinkA=0, shrinkB=0))


box(0.02, 1.36, "Input", ["text up to 30 June 1939:", "US newspapers, German", "newspapers in the original,",
                          "books, law, science"], FILL["in"])
box(1.53, 0.80, "Tokenizer", ["byte-level BPE,", "49,152 tokens,", "trained on the", "same text"], FILL["tok"])
box(2.48, 0.94, "Transformer", ["decoder-only,", "349M parameters,", "random init.,", "12.0B tokens seen"],
    FILL["model"])
box(3.57, 1.91, "Output: p(next token | text)", [], FILL["out"])
arrow(1.38, 1.53)
arrow(2.33, 2.48)
arrow(3.42, 3.57)
uses = [("bpb per year", "where knowledge ends"), ("forced choice", "what it knows"),
        ("fine-tuned copy", "answers questions")]
for i, (a, c) in enumerate(uses):
    y = 1.30 - i * 0.215
    left = ax.text(3.65, y, a, ha="left", va="center", fontsize=BODY)
    right = ax.text(5.42, y, c, ha="right", va="center", fontsize=BODY, style="italic", color=BLUE)
    texts += [(left, (3.57, 1.91)), (right, (3.57, 1.91))]
    pairs.append((left, right))

# date split: 1900 ... 1955 over [0.15, 5.35]
X = lambda yr: 0.15 + (yr - 1900) / 55 * 5.2      # noqa: E731
y0, hh = 0.17, 0.21
ax.add_patch(Rectangle((X(1900), y0), X(1939.5) - X(1900), hh, fc=GREEN, ec="none"))
ax.add_patch(Rectangle((X(1939.5), y0), X(1939.67) - X(1939.5), hh, fc=GREY, ec="none"))
ax.add_patch(Rectangle((X(1939.67), y0), X(1955) - X(1939.67), hh, fc=RED, ec="none"))
ax.text(X(1919.75), y0 + hh / 2, "TRAIN  1900 – 1939-06-30  (2 % of every year held out)", ha="center",
        va="center", color="white", fontsize=BODY, fontweight="bold")
ax.text(X(1947.3), y0 + hh / 2, "TEST only  1939-09 – 1955", ha="center", va="center", color="white",
        fontsize=BODY, fontweight="bold")
ax.annotate("embargo Jul–Aug 1939: never trained on or scored", (X(1939.58), y0), (X(1939.58) - 0.08, 0.005),
            ha="right", va="bottom", fontsize=SMALL, color=GREY,
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7, shrinkA=0, shrinkB=0))
for yr in (1900, 1920, 1939, 1955):
    ax.text(X(yr) + (0.04 if yr == 1939 else 0.0), y0 + hh + 0.03, str(yr), ha="center", va="bottom",
            fontsize=SMALL, color=INK)

fig.canvas.draw()
r = fig.canvas.get_renderer()
inch = fig.dpi
for t, (x, w) in texts:                            # every label must sit inside its box with a margin
    bb = t.get_window_extent(r)
    assert bb.x0 / inch >= x + 0.03 and bb.x1 / inch <= x + w - 0.03, (t.get_text(), bb.x0 / inch, bb.x1 / inch)
for left, right in pairs:
    assert left.get_window_extent(r).x1 / inch + 0.08 <= right.get_window_extent(r).x0 / inch, left.get_text()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"overview.{ext}", dpi=220)
print("wrote", OUT / "overview.pdf")
