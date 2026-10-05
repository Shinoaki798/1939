"""Per-article keep/drop rules applied after ingest.

Each rule is a pure function of columns already present in the ingested parquet,
so a rule change never requires re-reading the raw archives, and the yield of
every rule can be reported by year in reports/.

Language rule (calibrated on 12,000 scans of 1923, 2026-10-05):
  - English papers: median en_share 0.60, 5th pct 0.35; 2.4 % of their articles
    lose the stopword vote, nearly all address lists / small ads.
  - Spanish papers (El Mundo, El Imparcial): median en_share 0.02, 0.3 % voted "en".
  - Czech paper (Svet): median en_share 0.17, 2 % voted "en".
  An article-level vote alone lets a few foreign rows through, so the article
  rule is paired with a title-level rule: a title (LCCN) whose articles that
  year are mostly not English is dropped whole. Missing an English article
  costs nothing (English is far beyond the training budget); letting foreign
  text in would contaminate the English-only tokenizer (CLAUDE.md rule 10).
"""

from __future__ import annotations

import math

MIN_EN_SHARE = 0.3          # article level
MIN_TITLE_EN_FRACTION = 0.5  # title level, per (lccn, year)


def is_native_english(lang: str, en_share: float) -> bool:
    """Article-level rule. "und" (fewer than 5 stopwords; mostly <20-word fragments) is dropped."""
    if lang != "en" or en_share is None or math.isnan(en_share):
        return False
    return en_share >= MIN_EN_SHARE


def english_titles(lccn_year_lang: "list[tuple[str, int, str]]") -> set[tuple[str, int]]:
    """Title-level rule. Input: (lccn, year, lang) per article. Returns the (lccn, year)
    pairs whose fraction of lang == "en" articles is at least MIN_TITLE_EN_FRACTION."""
    total: dict[tuple[str, int], int] = {}
    english: dict[tuple[str, int], int] = {}
    for lccn, year, lang in lccn_year_lang:
        key = (lccn, year)
        total[key] = total.get(key, 0) + 1
        if lang == "en":
            english[key] = english.get(key, 0) + 1
    return {k for k, n in total.items() if english.get(k, 0) / n >= MIN_TITLE_EN_FRACTION}
