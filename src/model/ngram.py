"""Modified Kneser-Ney 5-gram baseline, the non-neural floor (HANDOFF §6.2; decisions 2026-10-19).

KenLM with its default estimation (lmplz: modified Kneser-Ney, no pruning) on the first `train_tokens`
tokens of the main stream (config/model_ladder.yaml, ngram_kn5), stored as a binary trie (exact, no
quantisation). The full 12B-token stream would hold ~13.6B distinct n-grams, far beyond the box's memory.
Token ids are written as words, one document per line (the stream's <|endoftext|> separates documents),
and piped straight into lmplz; the ARPA output is gzipped on the fly and deleted once the binary is built.

    python -m src.model.ngram --name ngram_kn5 [--tokens 1e9] [--memory 50%]
Output in runs/<name>/: config.yaml, data_manifest.json, git_sha.txt, env.txt, lmplz.log, model.binary.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np
import yaml

from src.data.download import load_config, repo_path
from src.train.train import Stream

IDS = [f"{i}".encode() for i in range(65536)]


def export(stream: Stream, tokens: int, out, eos: int = 0, chunk: int = 1 << 24) -> dict:
    """Write the first `tokens` tokens as one document per line; returns document and token counts."""
    docs = written = 0
    carry = np.zeros(0, dtype=np.int64)
    for a in range(0, tokens, chunk):
        t = np.concatenate([carry, stream.span(a, min(a + chunk, tokens))])
        cut = np.flatnonzero(t == eos)
        start = 0
        lines = []
        for c in cut:
            if c > start:
                lines.append(b" ".join([IDS[i] for i in t[start:c].tolist()]))
                written += c - start
                docs += 1
            start = c + 1
        carry = t[start:]
        if lines:
            out.write(b"\n".join(lines) + b"\n")
    if len(carry):                                   # the document cut by the token limit
        out.write(b" ".join([IDS[i] for i in carry.tolist()]) + b"\n")
        written += len(carry)
        docs += 1
    return {"documents": int(docs), "tokens": int(written)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="ngram_kn5")
    ap.add_argument("--tokens", type=float)
    ap.add_argument("--memory", default="50%", help="lmplz -S and build_binary -S")
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    ladder = yaml.safe_load(repo_path("config/model_ladder.yaml").read_text(encoding="utf-8"))
    spec = next(b for b in ladder["baselines"] if b["name"] == args.name)
    tokens = int(args.tokens or spec["train_tokens"])
    run_dir = repo_path(cfg["runs"]) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    bindir = os.path.expanduser(cfg["kenlm_bin"])
    stream = Stream(repo_path(cfg["shards"]) / "main")
    tmp = repo_path(cfg["data_root"]) / "kenlm_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    arpa, binary = tmp / f"{args.name}.arpa.gz", run_dir / "model.binary"   # gzip: ~70 GB of ARPA text otherwise
    run = {"name": args.name, "model": spec, "train_tokens": tokens, "stream": "main", "memory": args.memory,
           "lmplz": f"lmplz -o {spec['order']} -S {args.memory} -T {tmp} | gzip -1 > <arpa.gz>",
           "build_binary": f"build_binary -S {args.memory} -T {tmp} trie <arpa.gz> model.binary"}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo_path(".")).stdout.strip()
    (run_dir / "git_sha.txt").write_text(sha + "\n", encoding="utf-8")
    t0 = time.time()
    with open(run_dir / "lmplz.log", "wb") as log:
        with open(arpa, "wb") as gz_out:
            p = subprocess.Popen([f"{bindir}/lmplz", "-o", str(spec["order"]), "-S", args.memory, "-T", str(tmp)],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
            z = subprocess.Popen(["gzip", "-1"], stdin=p.stdout, stdout=gz_out)
            p.stdout.close()                         # gzip holds the read end
            counts = export(stream, tokens, p.stdin)
            p.stdin.close()
            if p.wait() or z.wait():
                raise SystemExit(f"lmplz ({p.returncode}) or gzip ({z.returncode}) failed; see {run_dir / 'lmplz.log'}")
        log.write(f"\n--- build_binary after {time.time() - t0:.0f}s ---\n".encode())
        log.flush()
        b = subprocess.run([f"{bindir}/build_binary", "-S", args.memory, "-T", str(tmp), "trie", str(arpa), str(binary)],
                           stdout=log, stderr=log)
        if b.returncode:
            raise SystemExit(f"build_binary failed ({b.returncode}); see {run_dir / 'lmplz.log'}")
    arpa_bytes = arpa.stat().st_size
    arpa.unlink()
    (run_dir / "data_manifest.json").write_text(json.dumps({
        "stream": "main", "stream_manifest_sha256": stream.sha256, "first_tokens": tokens, **counts,
        "arpa_bytes": arpa_bytes, "binary_bytes": binary.stat().st_size, "seconds": round(time.time() - t0)}, indent=1),
        encoding="utf-8")
    print(f"-> {binary} ({binary.stat().st_size / 2**30:.1f} GiB), {counts}", flush=True)


if __name__ == "__main__":
    main()
