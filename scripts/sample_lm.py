"""Scratch: sample continuations from a finished recurrent baseline on the CPU (not project code).

    python sample_lm.py <run name> "<prompt>" ...
Each prompt follows <|endoftext|> (id 0), as every document does in training; generation stops at <|endoftext|>.
One greedy continuation and three samples (temperature 0.8, top-k 40), 60 tokens each.
"""
import json
import sys

import torch
import yaml

sys.path.insert(0, "/home/an/1939")
from tokenizers import Tokenizer  # noqa: E402

from src.data.download import repo_path  # noqa: E402
from src.model.build import build  # noqa: E402

torch.set_num_threads(6)
name, prompts = sys.argv[1], sys.argv[2:]
run = yaml.safe_load(open(repo_path(f"runs/{name}/config.yaml")))
ck_name = open(repo_path(f"runs/{name}/checkpoints/latest.txt")).read().strip()
model = build(run["model"]).eval()
ck = torch.load(repo_path(f"runs/{name}/checkpoints/{ck_name}"), map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])
tok = Tokenizer.from_file(str(repo_path(yaml.safe_load(open(repo_path("config/tokenizer.yaml")))["file"])))


@torch.no_grad()
def continue_(prompt, n=60, temp=0.8, top_k=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = [0] + tok.encode(prompt, add_special_tokens=False).ids
    out, state = model.rnn(model.emb(torch.tensor([ids])))
    h, gen = out[:, -1], []
    for _ in range(n):
        logits = (h @ model.emb.weight.t()).float()[0]
        if temp == 0:
            nxt = int(logits.argmax())
        else:
            v, i = logits.topk(top_k)
            nxt = int(i[torch.multinomial(torch.softmax(v / temp, -1), 1, generator=g)])
        if nxt == 0:
            break
        gen.append(nxt)
        o, state = model.rnn(model.emb(torch.tensor([[nxt]])), state)
        h = o[:, -1]
    return tok.decode(gen)


print(json.dumps({"run": name, "checkpoint": ck_name, "step": ck["step"]}))
for p in prompts:
    print("=" * 100)
    print("PROMPT:", repr(p), "tokens:", tok.encode(p, add_special_tokens=False).tokens)
    print("[greedy]", repr(continue_(p, temp=0)))
    for s in range(3):
        print(f"[sample {s + 1}]", repr(continue_(p, seed=s + 1)))
