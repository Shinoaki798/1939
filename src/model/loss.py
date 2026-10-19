"""Output layer and next-token cross-entropy shared by every neural model.

The output matrix is the (tied) embedding, shape (vocab, d). For one 4,096-token row the logits alone are
4,096 x 49,152 values, so they are computed a chunk of positions at a time and never kept: in training each
chunk is recomputed in the backward pass (activation checkpointing). Losses are in nats; targets < 0 are
ignored (padding in evaluation).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

CHUNK = 16384


def _chunk_sum(h: torch.Tensor, w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy((h @ w.t()).float(), t, reduction="sum", ignore_index=-1)


_compiled = None


def loss_sum(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, compile: bool = False,
             chunk: int = CHUNK) -> torch.Tensor:
    """Summed cross-entropy of hidden states (..., d) against targets (...) through weight (V, d)."""
    global _compiled
    fn = _chunk_sum
    if compile:
        if _compiled is None:
            _compiled = torch.compile(_chunk_sum)
        fn = _compiled
    h, t = hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1)
    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, t.numel(), chunk):
        part = (h[i:i + chunk], weight, t[i:i + chunk])
        total = total + (checkpoint(fn, *part, use_reentrant=False) if torch.is_grad_enabled() else fn(*part))
    return total


@torch.no_grad()
def token_nll(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk: int = CHUNK) -> torch.Tensor:
    """Per-position cross-entropy in nats, same shape as targets; 0 where the target is ignored."""
    h, t = hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1)
    out = [F.cross_entropy((h[i:i + chunk] @ weight.t()).float(), t[i:i + chunk], reduction="none", ignore_index=-1)
           for i in range(0, t.numel(), chunk)]
    return torch.cat(out).view(targets.shape)
