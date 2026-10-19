"""Model specs from config/model_ladder.yaml and construction from a spec (one dict, stored in every run's
config.yaml, so a checkpoint can be rebuilt without the ladder)."""
from __future__ import annotations

import torch.nn as nn

from src.model.recurrent import CELLS, RecurrentLM
from src.model.transformer import Transformer, TransformerConfig


def model_spec(name: str, ladder: dict) -> dict:
    """A rung name (s90m ... s335m) or a recurrent baseline name (rnn_50m, lstm_50m, gru_50m)."""
    for r in ladder["rungs"]:
        if r["name"] == name:
            return {"name": name, "type": "transformer", "vocab_size": ladder["vocab_size"],
                    "context": ladder["context_length"], "n_layer": r["n_layer"], "d_model": r["d_model"],
                    "n_head": r["n_head"], **ladder.get("transformer", {})}
    for b in ladder["baselines"]:
        if b["name"] == name and b["type"] in CELLS:
            return {"name": name, "type": b["type"], "vocab_size": ladder["vocab_size"], "hidden": b["hidden"],
                    "n_layer": b["n_layer"]}
    raise SystemExit(f"no neural model named {name} in config/model_ladder.yaml")


def build(spec: dict) -> nn.Module:
    if spec["type"] == "transformer":
        keys = TransformerConfig.__dataclass_fields__
        return Transformer(TransformerConfig(**{k: v for k, v in spec.items() if k in keys}))
    return RecurrentLM(spec["type"], spec["vocab_size"], spec["hidden"], spec.get("n_layer", 2))
