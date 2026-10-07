"""Split membership: a fixed hash of the parent document id (HANDOFF §5.2; CLAUDE.md rules 3-5).

Every document of the deduplicated pool, in either language, falls in one of 100 buckets by a blake2b
hash of its parent id (chunk suffixes `_cNNN` removed, so all chunks of one book, issue or volume share
a split). No pass over the pool is needed and the result is the same on every machine.

  holdout   buckets 0-1 (2 %), every year 1900-1955: never trained on; the per-year bits-per-byte curve.
            Test-A = held-out 1930-01-01..1938-12-31, Test-B = 1939-09-01..1945-12-31,
            Test-C = 1946-01-01..1955-12-31. Only American Stories is scored (rule 5).
  val       buckets 2-3 (2 %), dated <= 1939-06-30: early stopping and learning rate.
  train     the rest dated <= 1939-06-30.
  embargo   1939-07-01..1939-08-31: never trained, never scored; RQ3 conditioning contexts only.
  post      the rest dated >= 1939-09-01: never trained (probe contexts may be drawn from it).

The holdout is drawn from the deduplicated pool before OCR filtering and tokenisation (rule 4).
"""

from __future__ import annotations

import hashlib
import re

HOLDOUT_BUCKETS = range(0, 2)
VAL_BUCKETS = range(2, 4)
TEST_SETS = (("A", "1930-01-01", "1938-12-31"), ("B", "1939-09-01", "1945-12-31"), ("C", "1946-01-01", "1955-12-31"))
_CHUNK = re.compile(r"_c\d{3}$")


def parent_id(article_id: str) -> str:
    return _CHUNK.sub("", article_id)


def bucket(article_id: str) -> int:
    d = hashlib.blake2b(f"split:{parent_id(article_id)}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "big") % 100


def split_of(article_id: str, date_class: str) -> str:
    """date_class from src.data.select: pre, embargo or post."""
    if date_class == "embargo":
        return "embargo"
    b = bucket(article_id)
    if b in HOLDOUT_BUCKETS:
        return "holdout"
    if date_class == "post":
        return "post"
    return "val" if b in VAL_BUCKETS else "train"


def test_set(date: str, split: str) -> str | None:
    if split != "holdout":
        return None
    for name, lo, hi in TEST_SETS:
        if lo <= date <= hi:
            return name
    return None
