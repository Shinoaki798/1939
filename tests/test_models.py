"""src.model: parameter counts, no access to future tokens, chunked loss equals the plain loss, RoPE is relative."""
import math

import pytest
import torch
import torch.nn.functional as F
import yaml

from src.data.download import repo_path
from src.model.build import build, model_spec
from src.model.loss import loss_sum, token_nll
from src.model.recurrent import param_count
from src.model.transformer import TransformerConfig, apply_rope, rope_tables

LADDER = yaml.safe_load(repo_path("config/model_ladder.yaml").read_text(encoding="utf-8"))


def _params(spec):
    with torch.device("meta"):
        return sum(p.numel() for p in build(spec).parameters())


@pytest.mark.parametrize("name,expected", [("rnn_50m", 50_876_160), ("lstm_50m", 49_574_400), ("gru_50m", 49_211_136),
                                           ("s335m", 349_225_984)])
def test_parameter_counts_match_the_ladder(name, expected):
    spec = model_spec(name, LADDER)
    assert _params(spec) == expected
    if spec["type"] != "transformer":
        assert param_count(spec["type"], spec["vocab_size"], spec["hidden"], spec["n_layer"]) == expected
        assert abs(expected - 50e6) < 1.5e6


TINY = [{"type": "transformer", "vocab_size": 50, "context": 16, "n_layer": 2, "d_model": 32, "n_head": 4},
        {"type": "rnn", "vocab_size": 50, "hidden": 16, "n_layer": 2},
        {"type": "lstm", "vocab_size": 50, "hidden": 16, "n_layer": 2},
        {"type": "gru", "vocab_size": 50, "hidden": 16, "n_layer": 2}]


@pytest.mark.parametrize("spec", TINY, ids=lambda s: s["type"])
def test_no_position_sees_a_later_token(spec):
    torch.manual_seed(0)
    m = build(spec).eval()
    x = torch.randint(0, 50, (2, 16))
    x2 = x.clone()
    x2[:, 10:] = (x2[:, 10:] + 7) % 50
    with torch.no_grad():
        h, h2 = m(x), m(x2)
    assert h.shape == (2, 16, m.output_weight.shape[1])
    assert torch.allclose(h[:, :10], h2[:, :10], atol=1e-6)
    assert not torch.allclose(h[:, 10:], h2[:, 10:])


def test_chunked_loss_equals_plain_cross_entropy():
    torch.manual_seed(0)
    h = torch.randn(3, 11, 8, requires_grad=True)
    w = torch.randn(50, 8, requires_grad=True)
    t = torch.randint(0, 50, (3, 11))
    t[0, :2] = -1                                        # ignored positions
    a = loss_sum(h, w, t, chunk=7)
    b = F.cross_entropy((h @ w.t()).view(-1, 50), t.view(-1), reduction="sum", ignore_index=-1)
    assert torch.allclose(a, b, atol=1e-4)
    ga = torch.autograd.grad(a, (h, w))
    gb = torch.autograd.grad(b, (h, w))
    assert all(torch.allclose(x, y, atol=1e-4) for x, y in zip(ga, gb))
    per = token_nll(h, w, t, chunk=5)
    assert per.shape == t.shape and float(per[0, :2].abs().sum()) == 0.0
    assert torch.allclose(per.sum(), b.detach(), atol=1e-4)


def test_rope_scores_depend_only_on_relative_position():
    cfg = TransformerConfig(vocab_size=10, context=64, n_layer=1, d_model=16, n_head=2)
    cos, sin = rope_tables(cfg)
    torch.manual_seed(0)
    q, k = torch.randn(8), torch.randn(8)

    def score(m, n):
        x = torch.zeros(1, 64, 1, 8)
        y = torch.zeros(1, 64, 1, 8)
        x[0, m, 0], y[0, n, 0] = q, k
        return float((apply_rope(x, cos, sin)[0, m, 0] * apply_rope(y, cos, sin)[0, n, 0]).sum())
    assert math.isclose(score(5, 2), score(40, 37), rel_tol=1e-4)
    assert not math.isclose(score(5, 2), score(5, 4), rel_tol=1e-3)
