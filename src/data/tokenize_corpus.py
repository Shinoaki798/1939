"""Tokenise the filtered corpus with the frozen 48k BPE (`config/tokenizer.yaml`; TASKS Phase 1).

Every filtered file of both languages, every split, in the file's row order: the text is normalised as
for training (src.data.normalize; American Stories and the other sources it does not list are left as
they are) and encoded without special tokens; the ids of the file's documents go back to back into one
little-endian uint16 array. Output per filtered file, named by the sha256 of the array:
  data/tokenized/<lang>/<sha12>.bin           the ids
  data/tokenized/<lang>/<sha12>.idx.parquet   one row per filtered row, same order: article_id, split,
                                              n_tokens, n_bytes (UTF-8 bytes of the normalised text), offset
data/tokenized/MANIFEST.json lists the tokenizer (path, sha256), and per filtered file its bin, index,
sha256, rows and tokens, with totals per language and split. Finished files are recorded as they complete
(MANIFEST.partial.json), so a stopped run resumes; `--verify` decodes random documents back and compares
them with the normalised text.

    python -m src.data.tokenize_corpus [--lang en,de] [--workers 8] [--verify 200] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import random
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.data.download import load_config, repo_path
from src.data.normalize import normalize

LANGS = ("en", "de")
INDEX_SCHEMA = pa.schema([("article_id", pa.string()), ("split", pa.string()), ("n_tokens", pa.int32()),
                          ("n_bytes", pa.int32()), ("offset", pa.int64())])
BATCH = 512
_TOK = None


def tokenizer_config() -> dict:
    tc = yaml.safe_load(repo_path("config/tokenizer.yaml").read_text(encoding="utf-8"))
    path = repo_path(tc["file"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != tc["sha256"]:
        raise SystemExit(f"{path}: sha256 {digest} does not match config/tokenizer.yaml ({tc['sha256']})")
    return {"file": str(path), "sha256": digest, "vocab_size": tc["vocab_size"]}


def _init(tok_file: str, threads: int) -> None:
    os.environ["RAYON_NUM_THREADS"] = str(threads)          # before the tokenizers thread pool starts
    global _TOK
    from tokenizers import Tokenizer
    _TOK = Tokenizer.from_file(tok_file)


def encode_file(path: str, lang: str, out_dir: str, tok=None) -> dict:
    """Encode one filtered file; returns the output names, sha256 and counts."""
    tok = tok or _TOK
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    tmp_bin, tmp_idx = out / f".tmp_{stem}.bin", out / f".tmp_{stem}.idx.parquet"
    rows = {k: [] for k in INDEX_SCHEMA.names}
    offset, by_split = 0, Counter()
    h = hashlib.sha256()
    with open(tmp_bin, "wb") as fb:
        for b in pq.ParquetFile(path).iter_batches(batch_size=BATCH, columns=["article_id", "split", "source", "text"]):
            aid, split, src, text = (b.column(c).to_pylist() for c in ("article_id", "split", "source", "text"))
            texts = [normalize(t or "", s, lang) for t, s in zip(text, src)]
            for a, sp, t, e in zip(aid, split, texts, tok.encode_batch(texts, add_special_tokens=False)):
                ids = np.asarray(e.ids, dtype=np.uint16)
                buf = ids.tobytes()
                fb.write(buf)
                h.update(buf)
                for k, v in (("article_id", a), ("split", sp), ("n_tokens", len(ids)),
                             ("n_bytes", len(t.encode("utf-8"))), ("offset", offset)):
                    rows[k].append(v)
                offset += len(ids)
                by_split[sp] += len(ids)
    digest = h.hexdigest()
    pq.write_table(pa.table(rows, schema=INDEX_SCHEMA), tmp_idx, compression="zstd")
    final_bin, final_idx = out / f"{digest[:12]}.bin", out / f"{digest[:12]}.idx.parquet"
    os.replace(tmp_bin, final_bin)
    os.replace(tmp_idx, final_idx)
    return {"bin": f"{lang}/{final_bin.name}", "index": f"{lang}/{final_idx.name}", "sha256": digest,
            "index_sha256": hashlib.sha256(final_idx.read_bytes()).hexdigest(), "rows": len(rows["article_id"]),
            "tokens": offset, "tokens_by_split": dict(by_split)}


def _job(job: tuple) -> tuple[str, dict]:
    lang, rel, path, out_dir = job
    return rel, encode_file(path, lang, out_dir)


def verify(cfg: dict, manifest: dict, n: int, tok_file: str) -> dict:
    """Decode n random documents per language and compare with their normalised text."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_file)
    root, troot = repo_path(cfg["filtered"]), repo_path(cfg["tokenized"])
    rng = random.Random(20261009)
    result = {}
    for lang in LANGS:
        items = [(rel, o) for rel, o in manifest["outputs"].items() if rel.startswith(lang + "/") and o["rows"]]
        ok = bad = 0
        for rel, o in rng.sample(items, min(len(items), 20)):
            idx = pq.read_table(troot / o["index"])
            ids = np.memmap(troot / o["bin"], dtype="<u2", mode="r")
            src = pq.read_table(root / rel, columns=["source", "text"])
            for i in rng.sample(range(idx.num_rows), min(n // 20 + 1, idx.num_rows)):
                off, k = idx.column("offset")[i].as_py(), idx.column("n_tokens")[i].as_py()
                text = normalize(src.column("text")[i].as_py() or "", src.column("source")[i].as_py(), lang)
                if tok.decode(ids[off:off + k].tolist(), skip_special_tokens=False) == unicodedata.normalize("NFC", text):
                    ok += 1
                else:
                    bad += 1
        result[lang] = {"checked": ok + bad, "round_trip_ok": ok, "mismatch": bad}
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="en,de")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2, help="tokenizer threads per worker")
    ap.add_argument("--verify", type=int, default=200, help="documents per language decoded back at the end")
    ap.add_argument("--dry-run", action="store_true", help="encode the first file of each language; write nothing")
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    tcfg = tokenizer_config()
    troot = repo_path(cfg["tokenized"])
    partial_path = troot / "MANIFEST.partial.json"
    partial = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.exists() else {}
    if partial.get("tokenizer_sha256") not in (None, tcfg["sha256"]):
        raise SystemExit("MANIFEST.partial.json was made with another tokenizer; move it aside first")
    done = partial.get("outputs", {})
    jobs, inputs = [], {}
    for lang in args.lang.split(","):
        froot = repo_path(cfg["filtered"]) / lang
        fm_path = froot / "MANIFEST.json"
        inputs[lang] = hashlib.sha256(fm_path.read_bytes()).hexdigest()
        fm = json.loads(fm_path.read_text(encoding="utf-8"))
        for o in sorted(fm["outputs"].values(), key=lambda o: o["file"]):
            rel = f"{lang}/{o['file']}"
            if rel in done and done[rel].get("filtered_sha256") == o["sha256"]:
                continue
            jobs.append((lang, rel, str(froot / o["file"]), str(troot / lang), o["sha256"]))
    print(f"{len(jobs):,} files to encode, {len(done):,} already done; tokenizer {tcfg['sha256'][:12]}", flush=True)
    if args.dry_run:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tcfg["file"])
        for lang in args.lang.split(","):
            j = next(j for j in jobs if j[0] == lang)
            t = pq.read_table(j[2], columns=["source", "text", "n_words"]).slice(0, 2000)
            texts = [normalize(x or "", s, lang) for x, s in zip(t.column("text").to_pylist(), t.column("source").to_pylist())]
            n = sum(len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False))
            print(f"  {lang} {j[1]}: {n:,} tokens for {sum(t.column('n_words').to_pylist()):,} words "
                  f"({n / max(1, sum(t.column('n_words').to_pylist())):.3f} tokens/word)", flush=True)
        return
    t0 = dt.datetime.now()
    partial.update(tokenizer_sha256=tcfg["sha256"])
    partial.setdefault("outputs", {})
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=_init, initargs=(tcfg["file"], args.threads)) as ex:
        futs = {ex.submit(_job, j[:4]): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            rel, out = fut.result()
            out["filtered_sha256"] = futs[fut][4]
            partial["outputs"][rel] = out
            if k % 200 == 0 or k == len(jobs):
                partial_path.write_text(json.dumps(partial), encoding="utf-8")
                print(f"{k:,}/{len(jobs):,} files, {(dt.datetime.now() - t0).seconds}s", flush=True)
    partial_path.write_text(json.dumps(partial), encoding="utf-8")
    totals = {lang: Counter() for lang in LANGS}
    for rel, o in partial["outputs"].items():
        totals[rel.split("/")[0]].update(o["tokens_by_split"])
    manifest = {"stage": "tokenized", "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "tokenizer": tcfg, "normalize": "src.data.normalize (training-only sources; American Stories untouched)",
                "dtype": "uint16 little-endian, documents back to back, no special tokens",
                "filtered_manifest_sha256": inputs, "outputs": partial["outputs"],
                "tokens_by_lang_split": {l: dict(c) for l, c in totals.items()},
                "seconds": (dt.datetime.now() - t0).seconds}
    if args.verify:
        manifest["verify"] = verify(cfg, manifest, args.verify, tcfg["file"])
        print("verify: " + json.dumps(manifest["verify"]), flush=True)
    (troot / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("tokens: " + json.dumps(manifest["tokens_by_lang_split"]), flush=True)
    print(f"-> {troot}", flush=True)


if __name__ == "__main__":
    main()
