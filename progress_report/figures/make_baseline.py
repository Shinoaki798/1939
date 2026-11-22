"""Draw progress_report/figures/baseline.{pdf,png}: the recurrent baselines (vanilla RNN, LSTM, GRU).

Same style and printed size as make_architecture.py (5.5 in = the text width, so every font size is the size on the
page, none below 7.5 pt). Structure from src/model/recurrent.py: token embedding of width h -> 2 recurrent layers
(input and hidden width h) -> output through the same embedding matrix (tied), softmax; each 4,096-token row starts
from a zero state. Widths and parameter counts from config/model_ladder.yaml (baselines), checked against the formula.

    python progress_report/figures/make_baseline.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent
V, CTX, NL = 49152, 4096, 2
GATES = {"RNN": 1, "LSTM": 4, "GRU": 3}
HIDDEN = {"RNN": 960, "LSTM": 800, "GRU": 832}


def params(kind: str) -> int:                     # src/model/recurrent.py param_count
    h = HIDDEN[kind]
    return V * h + NL * GATES[kind] * (2 * h * h + 2 * h)


assert [round(params(k) / 1e5) for k in ("RNN", "LSTM", "GRU")] == [509, 496, 492]

W, H = 5.5, 0.74
TITLE, BODY = 8.0, 7.5
INK, MUTED, LOOP = "#222222", "#3d3d3d", "#9a6528"
FILL = {"emb": "#dbe8f5", "rec": "#f5e6d3"}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "pdf.fonttype": 42})
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
checks = []                                       # (text, x0, x1): text must fit inside [x0, x1]


def box(x, w, cy, h, title, lines, kind):
    ax.add_patch(FancyBboxPatch((x, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=FILL[kind],
                                ec=INK, lw=0.7))
    top = cy + (0.135 * len(lines)) / 2
    checks.append((ax.text(x + w / 2, top, title, ha="center", va="center", fontsize=TITLE, weight="bold",
                           color=INK), x, x + w))
    for i, s in enumerate(lines):
        checks.append((ax.text(x + w / 2, top - 0.135 * (i + 1), s, ha="center", va="center", fontsize=BODY,
                               color=MUTED), x, x + w))
    return x + w


def arrow(p0, p1, **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=0.7, color=kw.pop("color", INK),
                                 shrinkA=0, shrinkB=0, **kw))


cy, bh, gap = 0.24, 0.44, 0.12
ax.text(0.24, cy, f"{CTX:,}\ntoken ids", ha="center", va="center", fontsize=BODY, color=MUTED, linespacing=1.05)
x = 0.49
arrow((x, cy), (x + 0.08, cy))
x = box(x + 0.08, 0.92, cy, bh, "Token embedding", [f"{V:,} × h"], "emb")
rec = []
for name in ("Recurrent layer 1", "Recurrent layer 2"):
    arrow((x, cy), (x + gap, cy))
    x0 = x + gap
    x = box(x0, 1.22, cy, bh, name, ["RNN / LSTM / GRU", "h → h"], "rec")
    rec.append((x0, x))
arrow((x, cy), (x + gap, cy))
x = box(x + gap, W - 0.02 - (x + gap), cy, bh, "Output layer", [f"h → {V:,}, tied to", "embedding; softmax"], "emb")
assert x <= W - 0.01
for x0, x1 in rec:                                 # the state is fed back from one token to the next
    arrow((x1 - 0.22, cy + bh / 2), (x0 + 0.22, cy + bh / 2), color=LOOP,
          connectionstyle="arc3,rad=0.30")
lx0, lx1 = rec[0][0], rec[1][1]
checks.append((ax.text((lx0 + lx1) / 2, H - 0.075, "state carried token to token; zero at the start of every "
                       f"{CTX:,}-token row", ha="center", va="center", fontsize=BODY, color=LOOP), 0.55, W - 0.55))

fig.canvas.draw()
r = fig.canvas.get_renderer()
for t, x0, x1 in checks:
    bb = t.get_window_extent(r)
    assert bb.x0 / fig.dpi >= x0 + 0.03 and bb.x1 / fig.dpi <= x1 - 0.03, (t.get_text(), bb.x0 / fig.dpi,
                                                                           bb.x1 / fig.dpi, x0, x1)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"baseline.{ext}", dpi=220)
print(f"wrote {OUT / 'baseline.pdf'}")
