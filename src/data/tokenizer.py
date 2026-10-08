"""Bilingual byte-level BPE (CLAUDE.md rule 10: trained on native pre-cutoff EN + DE training text only, no
pretrained tokenizer; HANDOFF §12 2026-10-05: 48k), and the 32k / 48k / 64k comparison (user, 2026-10-08).

  sample   a training sample from the drawn mixture (data/mixture): a selected document is taken with
           probability rate x count (its seen passes), by a seeded hash of its id, so the sample follows the
           seen mixture (language, period, source); documents with ocr_hit < --min-ocr are left out (keyed
           sources always kept); text normalised as the shards will be (src.data.normalize).
           -> data/tokenizer/sample/<sha12>.parquet (text, source, lang)
  train    one BPE run at the largest size. BPE picks merges greedily, so a run stopped after N merges learns
           exactly the first N merges of a longer run on the same data: smaller vocabularies are prefixes
           (checked when they are cut). Byte-level, NFC, digits split one by one, <|endoftext|>.
           -> data/tokenizer/bpe_<size>/tokenizer.json
  compare  bytes and words per token on held-out American Stories (the scored English text) and German
           held-out pages; the share of OCR junk among the word-initial tokens each size adds (tokens that
           begin no word of the OCR lexicons); probe terms that are a single token.
           -> reports/tokenizer_compare.md

    python -m src.data.tokenizer sample [--rate 0.04] [--min-ocr 0.85]
    python -m src.data.tokenizer train [--sizes 32768,49152,65536]
    python -m src.data.tokenizer compare [--sizes ...]
"""
from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data import ocr_quality as oq
from src.data.download import load_config, repo_path
from src.data.mixture import uniform
from src.data.normalize import normalize

LANGS = ("en", "de")
SEED = 20261008
SPECIALS = ["<|endoftext|>"]
SIZES = (32768, 49152, 65536)
D_MODEL, N_LAYER = 1024, 24                      # the main run (HANDOFF §12 2026-10-05)
SAMPLE_SCHEMA = pa.schema([("text", pa.string()), ("source", pa.string()), ("lang", pa.string())])

_CHOSEN: dict = {}


def tok_dir(cfg: dict) -> Path:
    return repo_path(cfg.get("tokenizer", "./data/tokenizer"))


# ------------------------------------------------------------------ sample

def _sample_file(job: tuple) -> dict:
    lang, path, rate, min_ocr = job
    chosen = _CHOSEN[lang]
    t = pq.read_table(path, columns=["article_id"])
    aid = t.column("article_id").to_numpy(zero_copy_only=False)
    _, u = uniform(aid, SEED)
    idx = np.flatnonzero(u < 2 * rate)                   # superset: count <= 2
    idx = np.array([i for i in idx if aid[i] in chosen and u[i] < rate * chosen[aid[i]]], dtype=np.int64)
    out = {"text": [], "source": [], "lang": [], "dropped_ocr": 0}
    if not len(idx):
        return out
    off = 0      # batch by batch: a whole file's text column can exceed 2 GB of string offsets
    for b in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["text", "source", "ocr_hit", "keyed"]):
        lo, hi = np.searchsorted(idx, off), np.searchsorted(idx, off + b.num_rows)
        if hi > lo:
            t = b.take(pa.array(idx[lo:hi] - off))
            for text, src, hit, keyed in zip(*(t.column(c).to_pylist() for c in ("text", "source", "ocr_hit", "keyed"))):
                if not keyed and hit is not None and not math.isnan(hit) and hit < min_ocr:
                    out["dropped_ocr"] += 1
                    continue
                out["text"].append(normalize(text or "", src, lang))
                out["source"].append(src)
                out["lang"].append(lang)
        off += b.num_rows
    return out


def sample(cfg: dict, rate: float, min_ocr: float, workers: int) -> None:
    mix = repo_path(cfg["mixture"])
    mm = json.loads((mix / "MANIFEST.json").read_text(encoding="utf-8"))
    jobs = []
    for lang in LANGS:
        sel = pq.read_table(mix / mm["outputs"][lang]["file"], columns=["article_id", "count"])
        ids = sel.column("article_id").to_numpy(zero_copy_only=False)
        cnt = sel.column("count").to_numpy(zero_copy_only=False)
        _, u = uniform(ids, SEED)
        keep = u < rate * cnt
        _CHOSEN[lang] = dict(zip(ids[keep].tolist(), cnt[keep].tolist()))
        root = repo_path(cfg["filtered"]) / lang
        fm = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        jobs += [(lang, str(root / o["file"]), rate, min_ocr) for o in fm["outputs"].values()]
        print(f"{lang}: {len(_CHOSEN[lang]):,} documents drawn", flush=True)
    out = tok_dir(cfg) / "sample"
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / ".tmp_sample.parquet"
    w = pq.ParquetWriter(tmp, SAMPLE_SCHEMA, compression="zstd")
    stats = {"docs": {l: 0 for l in LANGS}, "bytes": {l: 0 for l in LANGS}, "dropped_ocr": 0}
    with ProcessPoolExecutor(max_workers=workers) as ex:                 # fork: workers see _CHOSEN
        for r in ex.map(_sample_file, jobs, chunksize=16):
            stats["dropped_ocr"] += r.pop("dropped_ocr")
            if r["text"]:
                w.write_table(pa.table(r, schema=SAMPLE_SCHEMA))
                for text, lang in zip(r["text"], r["lang"]):
                    stats["docs"][lang] += 1
                    stats["bytes"][lang] += len(text.encode("utf-8"))
    w.close()
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    final = out / f"{digest[:12]}.parquet"
    os.replace(tmp, final)
    for old in out.glob("*.parquet"):
        if old != final:
            old.unlink()
    (out / "MANIFEST.json").write_text(json.dumps({
        "stage": "tokenizer_sample", "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "file": final.name, "sha256": digest, "rate": rate, "min_ocr": min_ocr, "seed": SEED,
        "mixture_manifest_sha256": hashlib.sha256((mix / "MANIFEST.json").read_bytes()).hexdigest(),
        **stats}, indent=1), encoding="utf-8")
    print(f"-> {final} {json.dumps(stats)}", flush=True)


def sample_path(cfg: dict) -> Path:
    d = tok_dir(cfg) / "sample"
    return d / json.loads((d / "MANIFEST.json").read_text(encoding="utf-8"))["file"]


# ------------------------------------------------------------------ train

def new_tokenizer():
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True),
                                                 pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])
    tok.decoder = decoders.ByteLevel()
    return tok


def train_bpe(texts, vocab_size: int, length: int | None = None):
    from tokenizers import pre_tokenizers, trainers
    tok = new_tokenizer()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer, length=length)
    return tok


def truncate(tok, size: int):
    """The tokenizer a run stopped at `size` would have learned: the first merges of a longer run."""
    from tokenizers import Tokenizer
    j = json.loads(tok.to_str())
    vocab, merges = j["model"]["vocab"], j["model"]["merges"]
    base = len(vocab) - len(merges)
    pairs = [m.split(" ") if isinstance(m, str) else m for m in merges]
    if any(vocab["".join(p)] != base + k for k, p in enumerate(pairs)):
        raise ValueError("merge ids are not sequential; train each size separately")
    j["model"]["vocab"] = {t: i for t, i in vocab.items() if i < size}
    j["model"]["merges"] = merges[:size - base]
    return Tokenizer.from_str(json.dumps(j))


def iter_texts(path: Path, batch: int = 2000):
    for b in pq.ParquetFile(path).iter_batches(batch_size=batch, columns=["text"]):
        yield from b.column(0).to_pylist()


def train(cfg: dict, sizes: tuple[int, ...]) -> None:
    sp = sample_path(cfg)
    n = pq.ParquetFile(sp).metadata.num_rows
    t0 = dt.datetime.now()
    tok = train_bpe(iter_texts(sp), max(sizes), length=n)
    print(f"trained {max(sizes):,} on {n:,} documents in {(dt.datetime.now() - t0).seconds}s", flush=True)
    for size in sizes:
        t = tok if size == max(sizes) else truncate(tok, size)
        d = tok_dir(cfg) / f"bpe_{size}"
        d.mkdir(parents=True, exist_ok=True)
        t.save(str(d / "tokenizer.json"))
        meta = {"vocab_size": t.get_vocab_size(), "sample": sp.name,
                "sha256": hashlib.sha256((d / "tokenizer.json").read_bytes()).hexdigest(),
                "prefix_of": None if size == max(sizes) else f"bpe_{max(sizes)}",
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        (d / "MANIFEST.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"-> {d} {meta['vocab_size']:,}", flush=True)


# ------------------------------------------------------------------ compare

def heldout_texts(cfg: dict, lang: str, sources: set[str], n: int) -> list[str]:
    """Up to n held-out documents dated <= cutoff (normalised like training text), by seeded hash."""
    root = repo_path(cfg["filtered"]) / lang
    fm = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    out = []
    files = sorted(o["file"] for o in fm["outputs"].values() if o["file"].split("/")[0] in sources)
    rate = None
    for f in files:
        t = pq.read_table(root / f, columns=["article_id", "split", "date", "source", "text"])
        sp = t.column("split").to_numpy(zero_copy_only=False)
        date = t.column("date").to_numpy(zero_copy_only=False)
        m = (sp == "holdout") & (date < "1939-07")
        if not m.any():
            continue
        if rate is None:
            rate = min(1.0, 4.0 * n / (m.sum() * len(files)))
        _, u = uniform(t.column("article_id").to_numpy(zero_copy_only=False), SEED + 1)
        for i in np.flatnonzero(m & (u < rate)):
            out.append(normalize(t.column("text")[i].as_py() or "", t.column("source")[i].as_py(), lang))
        if len(out) >= n:
            break
    return out[:n]


def byte_decoder() -> dict[str, int]:
    from tokenizers import pre_tokenizers
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    # GPT-2 byte-to-unicode map, rebuilt (printable bytes map to themselves)
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, k = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + k)
            k += 1
    m = {chr(c): b for b, c in zip(bs, cs)}
    assert set(m) == set(alphabet)
    return m


def token_text(token: str, dec: dict[str, int]) -> str:
    return bytes(dec[c] for c in token if c in dec).decode("utf-8", errors="replace")


def compare(cfg: dict, sizes: tuple[int, ...], n_en: int, n_de: int, out_md: str) -> None:
    from tokenizers import Tokenizer
    toks = {s: Tokenizer.from_file(str(tok_dir(cfg) / f"bpe_{s}" / "tokenizer.json")) for s in sizes}
    evals = {"en: American Stories held-out (scored text)": heldout_texts(cfg, "en", {"american_stories"}, n_en),
             "de: DDB + Europeana held-out": heldout_texts(cfg, "de", {"ddb_newspapers_de", "europeana_newspapers_de"}, n_de)}
    rows = []
    for name, texts in evals.items():
        nbytes = sum(len(t.encode("utf-8")) for t in texts)
        nwords = sum(len(t.split()) for t in texts)
        for s, tok in toks.items():
            ntok = sum(len(e.ids) for e in tok.encode_batch(texts))
            rows.append((name, s, len(texts), nbytes / ntok, ntok / nwords))
    lex = sorted({w.lower() for lang in LANGS for w in oq.load_lexicon(cfg, lang)})
    dec = byte_decoder()

    def starts_a_word(piece: str) -> bool:
        i = bisect.bisect_left(lex, piece)
        return i < len(lex) and lex[i].startswith(piece)

    added, prev = [], 0
    vocab = {i: t for t, i in toks[max(sizes)].get_vocab().items()}
    for s in sorted(sizes):
        words = [token_text(vocab[i], dec) for i in range(prev, s)]
        words = [w for w in words if w.startswith(" ") and len(w) >= 4 and w[1:].isalpha()]
        junk = [w for w in words if not starts_a_word(w[1:].lower())]
        added.append((s, s - prev, len(words), len(junk), junk[:25]))
        prev = s
    probes = []
    for f in ("probes/rq1_seed.csv", "probes/c1_screen.csv"):
        with open(repo_path(f), encoding="utf-8") as fh:
            probes += [r["term"] for r in csv.DictReader(fh)]
    single = {s: sorted({p for p in probes for form in (" " + p, " " + p.capitalize())
                         if len(tok.encode(form).ids) == 1}) for s, tok in toks.items()}
    body = 12 * N_LAYER * D_MODEL ** 2
    out = [f"# Tokenizer comparison ({dt.date.today().isoformat()})", "",
           "Byte-level BPE trained once on the mixture-weighted sample (`data/tokenizer/sample`, "
           "MANIFEST there); the smaller vocabularies are prefixes of the largest run's merges. Held-out "
           "documents only (never in the tokenizer sample); text normalised as in training.", "",
           "| eval set | vocab | docs | bytes / token | tokens / word |", "|---|---|---|---|---|"]
    out += [f"| {n} | {s:,} | {d:,} | {b:.3f} | {t:.3f} |" for n, s, d, b, t in rows]
    out += ["", "| vocab | embedding params (tied, d 1024) | share of a 350M model | output layer vs 24-layer body (FLOPs/token) |",
            "|---|---|---|---|"]
    out += [f"| {s:,} | {s * D_MODEL / 1e6:.1f}M | {s * D_MODEL / 350e6:.1%} | {s * D_MODEL / body:.1%} |" for s in sizes]
    out += ["", "Tokens each size adds over the previous one; word-initial = a space then >= 3 letters; junk = "
            "begins no word of the EN or DE OCR lexicon:", "", "| vocab | tokens added | word-initial | junk | examples of junk |",
            "|---|---|---|---|---|"]
    out += [f"| {s:,} | {a:,} | {w:,} | {j:,} ({j / max(w, 1):.1%}) | {' '.join(x.strip() for x in ex)} |"
            for s, a, w, j, ex in added]
    out += ["", "Probe terms that are a single token (must be none): " +
            "; ".join(f"{s:,}: {', '.join(v) or 'none'}" for s, v in single.items()), ""]
    Path(out_md).write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sample", "train", "compare"])
    ap.add_argument("--rate", type=float, default=0.04, help="sample: share of each selected pass taken")
    ap.add_argument("--min-ocr", type=float, default=0.85)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    ap.add_argument("--n-en", type=int, default=20000)
    ap.add_argument("--n-de", type=int, default=3000)
    ap.add_argument("--out", default="reports/tokenizer_compare.md")
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    sizes = tuple(sorted(int(s) for s in args.sizes.split(",")))
    if args.cmd == "sample":
        sample(cfg, args.rate, args.min_ocr, args.workers)
    elif args.cmd == "train":
        train(cfg, sizes)
    else:
        compare(cfg, sizes, args.n_en, args.n_de, args.out)


if __name__ == "__main__":
    main()
