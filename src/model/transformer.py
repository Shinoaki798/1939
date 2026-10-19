"""The decoder-only Transformer of the main run and the twin (CLAUDE.md rules 1-2, HANDOFF §6.1).

Token embedding -> n_layer pre-norm blocks -> RMSNorm -> output through the embedding matrix (tied).
Block: x + Attention(RMSNorm(x)), then x + SwiGLU(RMSNorm(x)).
  Attention: one projection to queries, keys and values for n_head heads of width d/n_head; rotary
    position embedding (RoPE) on queries and keys; causal scaled dot-product attention (each position sees
    itself and earlier positions only; PyTorch's fused kernel); output projection.
  SwiGLU: (SiLU(x W_gate) * (x W_up)) W_down with hidden width ffn_hidden (8d/3 rounded down to 64).
No biases anywhere. Initialisation N(0, 0.02); the two projections that write into the residual stream
(attention output, W_down) are scaled by 1/sqrt(2 n_layer), as in GPT-2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int = 49152
    context: int = 4096
    n_layer: int = 24
    d_model: int = 1024
    n_head: int = 16
    ffn_hidden: int = 0            # 0: 8 d / 3 rounded down to a multiple of 64
    rope_base: float = 10000.0

    def __post_init__(self):
        if not self.ffn_hidden:
            self.ffn_hidden = max(64, 64 * int(8 * self.d_model / 3 / 64))


def rope_tables(cfg: TransformerConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of position x frequency, shape (context, 1, head_dim / 2)."""
    half = cfg.d_model // cfg.n_head // 2
    inv_freq = 1.0 / cfg.rope_base ** (torch.arange(half, dtype=torch.float32) / half)
    angle = torch.outer(torch.arange(cfg.context, dtype=torch.float32), inv_freq)[:, None, :]
    return angle.cos(), angle.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each (first-half, second-half) pair of a head's features by its position's angle; x (B, T, H, D)."""
    a, b = x.chunk(2, dim=-1)
    T = x.shape[1]
    c, s = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    return torch.cat([a * c - b * s, a * s + b * c], dim=-1)


class Block(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        d = cfg.d_model
        self.n_head = cfg.n_head
        self.norm1 = nn.Parameter(torch.ones(d))
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.norm2 = nn.Parameter(torch.ones(d))
        self.gate_up = nn.Linear(d, 2 * cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, d, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(F.rms_norm(x, (D,), self.norm1)).view(B, T, 3, self.n_head, D // self.n_head).unbind(2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        gate, up = self.gate_up(F.rms_norm(x, (D,), self.norm2)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = nn.Parameter(torch.ones(cfg.d_model))
        cos, sin = rope_tables(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * cfg.n_layer) if name.endswith(("proj.weight", "down.weight")) else 0.02
                nn.init.normal_(p, std=std)

    @property
    def output_weight(self) -> torch.Tensor:
        return self.emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Token ids (B, T), T <= context -> final normalised hidden states (B, T, d)."""
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin)
        return F.rms_norm(x, (x.shape[-1],), self.norm)
