"""Recurrent baselines (HANDOFF §6.2, instructor's progression RNN -> LSTM -> GRU -> Transformer).

Token embedding -> 2 recurrent layers (cuDNN RNN with tanh, LSTM or GRU; input and hidden width h) ->
output through the same embedding matrix (tied weights, so the embedding width equals h). Every model is
about 50M parameters; the width is solved from the parameter count (config/model_ladder.yaml baselines):
  params = V*h + n_layer * g * (2*h*h + 2*h),  g = 1 (RNN), 3 (GRU), 4 (LSTM).
Each 4,096-token row starts from a zero state, as each Transformer row starts from no context.
"""
from __future__ import annotations

import torch
import torch.nn as nn

CELLS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
GATES = {"rnn": 1, "gru": 3, "lstm": 4}


def param_count(kind: str, vocab: int, hidden: int, n_layer: int = 2) -> int:
    return vocab * hidden + n_layer * GATES[kind] * (2 * hidden * hidden + 2 * hidden)


class RecurrentLM(nn.Module):
    def __init__(self, kind: str, vocab_size: int, hidden: int, n_layer: int = 2):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(vocab_size, hidden)
        self.rnn = CELLS[kind](hidden, hidden, num_layers=n_layer, batch_first=True)
        nn.init.normal_(self.emb.weight, std=0.02)       # small logits at the start (the output is tied)
        if kind == "lstm":                               # forget-gate bias 1 (Jozefowicz et al. 2015)
            for name, p in self.rnn.named_parameters():
                if name.startswith("bias_hh"):
                    with torch.no_grad():
                        p[hidden:2 * hidden].fill_(1.0)

    @property
    def output_weight(self) -> torch.Tensor:
        return self.emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Token ids (B, T) -> hidden states (B, T, h) before the output layer."""
        x = self.emb(idx)
        if x.is_cuda and torch.is_autocast_enabled("cuda"):
            # Autocast runs cuDNN RNNs in float16 whatever dtype is asked for, and there is no loss scaling
            # here. Run the recurrence in the autocast dtype (bfloat16) instead: bf16 copies of the fp32
            # weights, so gradients still reach the fp32 parameters.
            dt = torch.get_autocast_dtype("cuda")
            with torch.autocast("cuda", enabled=False):
                w = {n: p.to(dt) for n, p in self.rnn.named_parameters()}
                out, _ = torch.func.functional_call(self.rnn, w, (x.to(dt),))
            return out
        out, _ = self.rnn(x)
        return out
