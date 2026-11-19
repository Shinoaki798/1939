"""Draw progress_report/figures/architecture.{pdf,png}: the primary Transformer, every layer with its size.

Redrawn for the progress report at the printed size (5.5 in = the text width), so every font size is the size on
the page (none below 7.5 pt). Two rows: embedding and output layer on top, the decoder block below. The proposal's
one-row version (scripts/make_architecture_figure.py, 11 in wide, printed at half size) is left unchanged.

    python progress_report/figures/make_architecture.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent
V, D, L, HEADS, FF, CTX = 49152, 1024, 24, 16, 2688, 4096       # config/model_ladder.yaml (s335m), tokenizer
EMB = V * D
ATTN = 4 * D * D
FFN = 3 * D * FF
BLOCK = ATTN + FFN + 2 * D
TOTAL = EMB + L * BLOCK + D
assert TOTAL == 349_225_984

W, H = 5.5, 1.46
TITLE, BODY = 8.0, 7.5
INK, MUTED, GROUP = "#222222", "#3d3d3d", "#9a6528"
FILL = {"emb": "#dbe8f5", "norm": "#e6e6e6", "attn": "#d9ecd9", "ffn": "#e8dff2"}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "pdf.fonttype": 42})
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
checks = []                                       # (text, x0, x1): text must fit inside [x0, x1]


def m(n: int) -> str:
    return f"{n / 1e6:.1f}M"


def box(x, w, cy, h, title, lines, kind):
    ax.add_patch(FancyBboxPatch((x, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=FILL[kind],
                                ec=INK, lw=0.7))
    n = len(lines)
    top = cy + (0.135 * n) / 2
    checks.append((ax.text(x + w / 2, top, title, ha="center", va="center", fontsize=TITLE, weight="bold",
                           color=INK), x, x + w))
    for i, s in enumerate(lines):
        checks.append((ax.text(x + w / 2, top - 0.135 * (i + 1), s, ha="center", va="center", fontsize=BODY,
                               color=MUTED), x, x + w))
    return x + w


def arrow(p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=0.7, color=INK, shrinkA=0,
                                 shrinkB=0))


def line(xs, ys):
    ax.plot(xs, ys, color=INK, lw=0.7, solid_capstyle="butt")


R = 0.075                                          # radius of the residual "+" nodes


def plus(cx, cy):
    ax.add_patch(Circle((cx, cy), R, fc="white", ec=INK, lw=0.7))
    ax.text(cx, cy - 0.005, "+", ha="center", va="center", fontsize=8.5, color=INK)


# top row: token ids -> embedding ... output layer
ty, th = 1.20, 0.44
ax.text(0.24, ty, f"{CTX:,}\ntoken ids", ha="center", va="center", fontsize=BODY, color=MUTED, linespacing=1.05)
arrow((0.47, ty), (0.55, ty))
emb_x1 = box(0.55, 0.98, ty, th, "Token embedding", [f"{V:,} × {D:,}", f"{m(EMB)}  ({EMB / TOTAL:.0%})"], "emb")
out_x0 = 4.20
box(out_x0, W - 0.02 - out_x0, ty, th, "Output layer", [f"RMSNorm, {D:,} → {V:,}", "tied to embedding; softmax"],
    "emb")

# bottom row: one decoder block, repeated 24 times; residual paths run below the boxes
cy, bh = 0.48, 0.40
res_y = 0.105
x_in = 0.80                                        # the embedding's output comes down here
line([x_in, x_in], [ty - th / 2, cy])
x = 0.98
arrow((x_in, cy), (x, cy))
tap1 = 0.88
x = box(x, 0.62, cy, bh, "RMSNorm", [f"{D:,}"], "norm")
arrow((x, cy), (x + 0.11, cy))
x = box(x + 0.11, 1.08, cy, bh, "Causal self-attention", [f"{HEADS} heads × {D // HEADS}, RoPE",
                                                           f"Q, K, V, O: {m(ATTN)}"], "attn")
p1 = x + 0.11 + R
arrow((x, cy), (p1 - R, cy))
plus(p1, cy)
tap2 = p1 + R + 0.07
arrow((p1 + R, cy), (tap2 + 0.10, cy))
x = box(tap2 + 0.10, 0.62, cy, bh, "RMSNorm", [f"{D:,}"], "norm")
arrow((x, cy), (x + 0.11, cy))
x = box(x + 0.11, 1.15, cy, bh, "SwiGLU feed-forward", [f"{D:,} → {FF:,} → {D:,}", m(FFN)], "ffn")
p2 = x + 0.11 + R
arrow((x, cy), (p2 - R, cy))
plus(p2, cy)
for xa, xb in ((tap1, p1), (tap2, p2)):            # residual connections around the two sub-layers
    line([xa, xa, xb], [cy, res_y, res_y])
    arrow((xb, res_y), (xb, cy - R))
g0, g1 = 0.66, p2 + R + 0.08
ax.add_patch(FancyBboxPatch((g0, 0.03), g1 - g0, 0.77, boxstyle="round,pad=0,rounding_size=0.06", fc="none",
                            ec=GROUP, lw=0.8, ls="--"))
arrow((p2, cy + R), (p2, ty - th / 2))             # into the output layer
gx = (emb_x1 + out_x0) / 2
checks.append((ax.text(gx, 1.27, f"Decoder block × {L}  (pre-norm)", ha="center", va="center", fontsize=TITLE,
                       weight="bold", color=GROUP), emb_x1, out_x0))
checks.append((ax.text(gx, 1.12, f"{m(BLOCK)} parameters each, {m(L * BLOCK)} in all ({L * BLOCK / TOTAL:.0%})",
                       ha="center", va="center", fontsize=BODY, color=GROUP), emb_x1, out_x0))
assert g1 <= W - 0.01 and out_x0 < p2 < W - 0.02, (g1, p2)

fig.canvas.draw()
r = fig.canvas.get_renderer()
for t, x0, x1 in checks:
    bb = t.get_window_extent(r)
    assert bb.x0 / fig.dpi >= x0 + 0.03 and bb.x1 / fig.dpi <= x1 - 0.03, (t.get_text(), bb.x0 / fig.dpi,
                                                                           bb.x1 / fig.dpi, x0, x1)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"architecture.{ext}", dpi=220)
print(f"wrote {OUT / 'architecture.pdf'}; {TOTAL:,} parameters; block group {g0:.2f}..{g1:.2f}")
