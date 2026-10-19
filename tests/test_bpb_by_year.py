"""src.eval.bpb_by_year: every token scored exactly once, uniform-model bpb, bootstrap CI brackets the value."""
import math

import numpy as np
import pandas as pd
import torch

from src.eval.bpb_by_year import neural_nll, pieces, summarize


def test_windows_score_every_token_once_with_history():
    docs = [np.arange(1, 4), np.arange(1, 12), np.arange(1, 9)]
    seen = {i: [] for i in range(3)}
    for i, x, y in pieces(docs, context=4, stride=2):
        assert len(x) == len(y) <= 4
        scored = np.flatnonzero(y >= 0)
        if seen[i]:
            assert scored[0] >= 2                     # a later window: context - stride tokens of history
        else:
            assert x[0] == 0                          # the first window starts with <|endoftext|>
        seen[i] += y[y >= 0].tolist()
    for i, d in enumerate(docs):
        assert seen[i] == d.tolist()


class Uniform(torch.nn.Module):
    output_weight = torch.zeros(50, 4)

    def forward(self, x):
        return torch.zeros(*x.shape, 4)


def test_uniform_model_gives_log2_vocab_per_token():
    docs = [np.array([5, 6, 7]), np.arange(1, 12)]
    nll = neural_nll(Uniform(), docs, context=4, stride=2, batch_tokens=8, device="cpu")
    assert np.allclose(nll, [3 * math.log(50), 11 * math.log(50)])
    d = pd.DataFrame({"year": [1930, 1930], "n_tokens": [3, 11], "n_bytes": [7, 21]})
    rows = summarize(d, nll, "en", 200, 0)
    assert rows[0]["bpb"] == round(14 * math.log2(50) / 28, 5)
    assert rows[0]["ci_lo"] <= rows[0]["bpb"] <= rows[0]["ci_hi"]
