"""Fine-tuning and evaluation pairs for the fine-tuned copy of the main model (user 2026-10-22 and 10-29;
HANDOFF §12; CLAUDE.md rules 1, 3, 5; config/finetune.yaml).

Period pairs. The question-and-answer columns of American Stories (syndicated "Answers to Questions" columns
of the 1920s-30s: "Q. ...? A. ..."), so question and answer are both written in the period. An article with
>= min_pairs pairs is split at line-initial "Q." / "A." markers (also "Q," "Q:" ...); a pair is a question of
question_chars characters ending in "?" directly followed by an answer of answer_chars characters (cut at the
next marker, or where a question with a misread marker runs into it: "q. How ...?"), white space collapsed
("O." counts as "Q.", a common OCR misreading). Court transcripts print
"Q. ... A. ..." too: an article in which >= testimony_share of the questions read as cross-examination
(`TESTIMONY`: "did / were / have / had / do you", or a leading pronoun or conjunction) is dropped. Sets by the
article's split and date:
    period_train / period_dev   split train, dated <= cutoff (dev: dev_share of the articles, by seeded hash)
    period_pre                  split holdout or val, dated <= cutoff                   (evaluation)
    period_post                 split holdout or post, dated >= test_start: never trained on, the knowledge
                                boundary                                                (evaluation)
The embargo window is never read. Within a set a repeated question is kept once (columns ran in many papers);
an evaluation question sharing >= overlap of its word 3-grams with a train question is dropped (3-grams in
more than 2,000 train questions are not looked up; they still count in the share's denominator).

CAQA pairs. ChroniclingAmericaQA (passage of a 1800-1920 page + machine-written question -> answer span):
seeded samples of train.json (caqa_train / caqa_dev) and test.json (caqa_test); passages are cut to
context_tokens at encoding time (`encode`), and an example whose answer the cut removes is dropped there.

Output: data/finetune/<sha256[:12]>.parquet (set, id, date, question, answer, context) + MANIFEST.json. The
evaluation sets are frozen with the file's sha256 in probes/CHECKSUMS before any model is scored on them.

    python -m src.data.qa_pairs --dry-run       # counts only
    python -m src.data.qa_pairs                 # write
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from src.data.download import load_config, repo_path
from src.data.mixture import uniform

MARK = re.compile(r"(?:^|\n)[ \t]*([QAO])[.,:][ \t]+")          # "O." is OCR for "Q."
QTAIL = re.compile(r"\s[QqOo@e][.,:]\s+[A-Z\"][^?]{3,300}\?")  # a question run into the answer ("q. How ...?")
TESTIMONY = re.compile(r"\b(did|were|have|had|do) you\b|^(and|now|then|well|so|but|you|he|she|they|isn't|didn't|"
                       r"wasn't|weren't)\b", re.IGNORECASE)
COLUMN = r"(^|\s)Q[.,:]\s[^?]{8,400}\?\s{0,3}A[.,:]\s"          # fast Arrow pre-filter for column articles
DF_CAP = 2000


def pairs(text: str, q_chars: tuple[int, int], a_chars: tuple[int, int]) -> list[tuple[str, str]]:
    """(question, answer) pairs of one column article."""
    marks = [(m.group(1), m.start(), m.end()) for m in MARK.finditer(text)] + [("", len(text), len(text))]
    out = []
    for (k, _, e), (k2, s2, e2), (_, s3, _) in zip(marks, marks[1:], marks[2:] + [("", len(text), len(text))]):
        if k not in ("Q", "O") or k2 != "A":
            continue
        q, a = " ".join(text[e:s2].split()), " ".join(text[e2:s3].split())
        m = QTAIL.search(a)
        if m:                                       # the next question, its marker misread mid-line
            a = a[:m.start()]
        if q.rstrip("\"'”’ ").endswith("?") and q_chars[0] <= len(q) <= q_chars[1] and \
                a_chars[0] <= len(a) <= a_chars[1]:
            out.append((q, a))
    return out


def cross_exam_share(questions: list[str]) -> float:
    """Share of questions phrased as cross-examination (court transcripts also print "Q. ... A. ...")."""
    return sum(bool(TESTIMONY.search(q)) for q in questions) / max(1, len(questions))


def words(q: str) -> tuple[str, ...]:
    return tuple(re.sub(r"[^a-z]+", " ", q.lower()).split())


def shingles(q: str) -> set:
    w = words(q)
    return {w[i:i + 3] for i in range(len(w) - 2)} if len(w) >= 3 else {w}


def near_duplicates(questions: list[str], train_questions: list[str], overlap: float) -> np.ndarray:
    """True for each question sharing >= overlap of its word 3-grams with some train question."""
    post = collections.defaultdict(list)
    for j, q in enumerate(train_questions):
        for s in shingles(q):
            post[s].append(j)
    out = np.zeros(len(questions), dtype=bool)
    for i, q in enumerate(questions):
        sh = shingles(q)
        hits = collections.Counter()
        for s in sh:
            if len(post.get(s, ())) <= DF_CAP:
                hits.update(post.get(s, ()))
        out[i] = bool(hits) and max(hits.values()) >= overlap * len(sh)
    return out


def first_per_question(df: pd.DataFrame) -> pd.DataFrame:
    key = df["question"].map(lambda q: " ".join(words(q)))
    return df.assign(_k=key).sort_values(["date", "id"]).drop_duplicates("_k").drop(columns="_k")


def period_pairs(cfg: dict, c: dict, seed: int) -> pd.DataFrame:
    cutoff, test_start = str(cfg["cutoff"]), str(cfg["test_start"])
    troot, froot = repo_path(cfg["tokenized"]), repo_path(cfg["filtered"])
    man = json.loads((troot / "MANIFEST.json").read_text(encoding="utf-8"))
    rows, court = [], collections.Counter()
    for key in sorted(man["outputs"]):
        if not key.startswith(f"en/{c['source']}/"):
            continue
        for b in pq.ParquetFile(froot / key).iter_batches(batch_size=50_000, columns=["article_id", "split", "date", "text"]):
            n = pc.count_substring_regex(b.column("text"), COLUMN).to_numpy(zero_copy_only=False)
            keep = n >= int(c["min_pairs"])
            if not keep.any():
                continue
            for r in b.filter(pa.array(keep)).to_pylist():
                d = r["date"]
                if r["split"] == "train" and d <= cutoff:
                    s = "period_train"
                elif r["split"] in ("holdout", "val") and d <= cutoff:
                    s = "period_pre"
                elif r["split"] in ("holdout", "post") and d >= test_start:
                    s = "period_post"
                else:
                    continue                                    # embargo, or a train / val article after it
                ps = pairs(r["text"], tuple(c["question_chars"]), tuple(c["answer_chars"]))
                if len(ps) >= int(c["min_pairs"]) and cross_exam_share([q for q, _ in ps]) >= float(c["testimony_share"]):
                    court[s] += len(ps)
                elif len(ps) >= int(c["min_pairs"]):
                    rows += [{"set": s, "id": f"{r['article_id']}#{k}", "article": r["article_id"], "date": d,
                              "question": q, "answer": a} for k, (q, a) in enumerate(ps)]
    df = pd.DataFrame(rows)
    df = pd.concat([first_per_question(g) for _, g in df.groupby("set")], ignore_index=True)
    train = df[df["set"] == "period_train"]
    ev = df["set"] != "period_train"
    dup = near_duplicates(df.loc[ev, "question"].tolist(), train["question"].tolist(), float(c["overlap"]))
    df = df.drop(index=df.index[ev][dup])
    u = uniform(df["article"].to_numpy(dtype=object), seed)[1]
    df.loc[(df["set"] == "period_train").to_numpy() & (u < float(c["dev_share"])), "set"] = "period_dev"
    print(f"period: {int(dup.sum()):,} evaluation pairs dropped as near-duplicates of train questions; pairs in "
          f"court-testimony articles dropped: {dict(court)}", flush=True)
    return df.drop(columns="article").assign(context=None)


def caqa_pairs(cfg: dict, c: dict, seed: int) -> pd.DataFrame:
    root = repo_path(c["dir"])
    rng = np.random.default_rng(seed)
    out = []
    for split, n, name in (("train", int(c["train"]), "caqa_train"), ("test", int(c["eval"]), "caqa_test")):
        items = json.loads((root / f"{split}.json").read_text(encoding="utf-8"))
        take = sorted(rng.choice(len(items), size=min(n, len(items)), replace=False).tolist())
        for i in take:
            r = items[i]
            m = re.search(r"_(\d{4})(\d{2})(\d{2})_", r["para_id"])
            date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""
            if date and date > str(cfg["cutoff"]):
                continue
            out.append({"set": name, "id": r["query_id"], "date": date, "question": r["question"].strip(),
                        "answer": r["answer"].strip(), "context": r["context"]})
        print(f"caqa {split}: {len(items):,} items, sampled {len(take):,}", flush=True)
        del items
    df = pd.DataFrame(out)
    u = uniform(df["id"].to_numpy(dtype=object), seed)[1]
    df.loc[(df["set"] == "caqa_train").to_numpy() & (u < float(c["dev_share"])), "set"] = "caqa_dev"
    return df


def prompt_text(question: str, context: str | None = None) -> str:
    """The column layout of American Stories: 'Q. question?' then a paragraph break and ' A.'."""
    return (f"{context}\n\n " if context else "") + f"Q. {question}\n\n A."


def encode(tok, question: str, answer: str, context: str | None = None,
           context_tokens: int | None = None) -> tuple[list[int], list[int]] | None:
    """Prompt ids and answer ids (' ' + answer) of one example; None when cutting the passage removes the answer."""
    if context and context_tokens:
        ids = tok.encode(context, add_special_tokens=False).ids
        if len(ids) > context_tokens:
            context = tok.decode(ids[:context_tokens])
            if answer not in context:
                return None
    return (tok.encode(prompt_text(question, context), add_special_tokens=False).ids,
            tok.encode(" " + answer, add_special_tokens=False).ids)


def load_pairs(cfg: dict) -> tuple[pd.DataFrame, str]:
    root = repo_path(cfg["data_root"]) / "finetune"
    man = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    path = root / man["file"]
    return pq.read_table(path).to_pandas(), hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    fcfg = yaml.safe_load(repo_path("config/finetune.yaml").read_text(encoding="utf-8"))
    seed, d = int(fcfg["seed"]), fcfg["data"]
    df = pd.concat([period_pairs(cfg, d["period"], seed), caqa_pairs(cfg, d["caqa"], seed)], ignore_index=True)
    counts = df.groupby("set").agg(pairs=("id", "size"), first=("date", "min"), last=("date", "max"))
    print(counts.to_string(), flush=True)
    if args.dry_run:
        return
    root = repo_path(cfg["data_root"]) / "finetune"
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / "pairs.tmp.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
    sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
    path = root / f"{sha[:12]}.parquet"
    tmp.replace(path)
    (root / "MANIFEST.json").write_text(json.dumps({
        "file": path.name, "sha256": sha, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": d, "seed": seed, "cutoff": str(cfg["cutoff"]), "test_start": str(cfg["test_start"]),
        "sources": {"period": "data/filtered en/american_stories (tokenized MANIFEST outputs)",
                    "caqa": "ChroniclingAmericaQA revision 15e58335, train.json / test.json"},
        "counts": {s: {"pairs": int(r.pairs), "first": r.first, "last": r.last} for s, r in counts.iterrows()},
    }, indent=1), encoding="utf-8")
    print(f"wrote {path} ({len(df):,} pairs, sha256 {sha[:12]})")


if __name__ == "__main__":
    main()
