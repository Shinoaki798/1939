"""Text normalisation of training-only sources (user decision 2026-10-08, HANDOFF §12), applied when the
training shards are written, identically to every split of a source. American Stories (the scored source)
and every source not listed here are left as they are. English line-break hyphens ("gladi- ators") are not
touched (they also occur in the scored text).

  europeana_newspapers_de  the upstream text repeats a word at every hyphenated line break ("eingetroffen.
                           eingetroffen."): an immediately repeated token of >= 4 letters loses its copy.
  ddb_newspapers_de,       pages tokenised with a space before punctuation ("Halt . „ Die Sache “ ,"):
  royal_society_corpus     no space before , . ; : ! ? ) ] or a closing quote, none after an opening one.
  every German document    ⸗ (U+2E17, the Fraktur line-end hyphen): joined when a lowercase word follows
                           ("Alka⸗ lien" -> "Alkalien") unless that word is a conjunction (a suspended
                           compound: "Unterhaltungs⸗ und" -> "Unterhaltungs- und"); any other ⸗ -> "-".
  congressional_record     the source prints every comma as a period: ". " before a lowercase letter
                           becomes ", " unless the word before it is a single letter ("p. m.") or a
                           common abbreviation ("etc.").

    python -m src.data.normalize --sample 3      # before/after of changed passages from the filtered data
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter

DUPLICATE = re.compile(r"(?<!\S)((?=\S*\w{4})\S+)\s+\1(?!\S)")
SPACE_BEFORE = re.compile(r"(?<=\S) +(?=[,.;:!?)\]](?:\s|$|[,.;:!?)\]\"'“”»]))")
QUOTES = {"de": (re.compile(r"([„‚(\[]) +"), re.compile(r"(?<=\S) +([“‘])(?=\s|$|[,.;:!?)])")),
          "en": (re.compile(r"([“‘(\[]) +"), re.compile(r"(?<=\S) +([”’])(?=\s|$|[,.;:!?)])"))}
OBLIQUE_JOIN = re.compile(r"⸗\s+(?=([a-zäöüß]\w*))")
CONJUNCTIONS = {"und", "oder", "bzw", "sowie", "bis", "noch", "als", "wie", "u"}
CR_PERIOD = re.compile(r"(\w+)\. (?=[a-z])")
CR_ABBREVIATIONS = {"etc", "viz", "vs", "cf", "ibid", "seq", "inst", "ult", "prox", "sec", "secs", "sess", "stat",
                    "vol", "pp", "ch", "cong", "no", "nos", "art", "par", "subsec", "pt", "mr", "mrs", "dr",
                    "messrs", "hon", "esq", "jr", "sr", "co", "corp", "bros", "ft", "lbs", "oz", "gal", "approx"}

TOKENISED_SOURCES = {"ddb_newspapers_de": "de", "royal_society_corpus": "en"}


def dedupe_repeats(text: str, stats: Counter) -> str:
    out, n = DUPLICATE.subn(r"\1", text)
    stats["repeats_removed"] += n
    return out


def detokenise(text: str, lang: str, stats: Counter) -> str:
    text, n = SPACE_BEFORE.subn("", text)
    opening, closing = QUOTES[lang]
    text, a = opening.subn(r"\1", text)
    text, b = closing.subn(r"\1", text)
    stats["spaces_removed"] += n + a + b
    return text


def oblique_hyphens(text: str, stats: Counter) -> str:
    def join(m: re.Match) -> str:
        if m.group(1).lower() in CONJUNCTIONS:
            stats["oblique_to_hyphen"] += 1
            return "- "
        stats["oblique_joined"] += 1
        return ""
    text = OBLIQUE_JOIN.sub(join, text)
    stats["oblique_to_hyphen"] += text.count("⸗")
    return text.replace("⸗", "-")


def cr_commas(text: str, stats: Counter) -> str:
    def fix(m: re.Match) -> str:
        w = m.group(1)
        if len(w) < 2 or w.lower() in CR_ABBREVIATIONS or w.isdigit():
            return m.group(0)
        stats["commas_restored"] += 1
        return w + ", "
    return CR_PERIOD.sub(fix, text)


def normalize(text: str, source: str, lang: str, stats: Counter | None = None) -> str:
    """The normalised text of one document of `source` in language `lang`."""
    stats = stats if stats is not None else Counter()
    if source == "europeana_newspapers_de":
        text = dedupe_repeats(text, stats)
    if source in TOKENISED_SOURCES:
        text = detokenise(text, TOKENISED_SOURCES[source], stats)
    if lang == "de" and "⸗" in text:
        text = oblique_hyphens(text, stats)
    if source == "congressional_record":
        text = cr_commas(text, stats)
    return text


def main() -> None:
    import pyarrow.parquet as pq

    from src.data.download import load_config, repo_path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=3, help="changed passages to show per source")
    ap.add_argument("--seed", type=int, default=20261008)
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    rng = random.Random(args.seed)
    targets = {"en": ["congressional_record", "royal_society_corpus"],
               "de": ["europeana_newspapers_de", "ddb_newspapers_de", "voelkischer_beobachter_de", "meyers6_ia"]}
    for lang, sources in targets.items():
        root = repo_path(cfg["filtered"]) / lang
        m = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        for src in sources:
            files = [root / o["file"] for o in m["outputs"].values() if o["file"].startswith(src + "/")]
            stats, words, shown = Counter(), 0, 0
            for f in rng.sample(files, min(4, len(files))):
                t = pq.read_table(f, columns=["text"])
                for i in rng.sample(range(t.num_rows), min(100, t.num_rows)):
                    x = t.column("text")[i].as_py() or ""
                    words += len(x.split())
                    y = normalize(x, src, lang, stats)
                    if y != x and shown < args.sample:
                        a = next(k for k in range(min(len(x), len(y))) if x[k] != y[k])
                        lo = max(0, a - 80)
                        print(f"--- {lang} {src}\n  before: {' '.join(x[lo:lo + 220].split())}\n"
                              f"  after:  {' '.join(y[lo:lo + 200].split())}")
                        shown += 1
            per = {k: round(v * 1000 / max(words, 1), 2) for k, v in stats.items()}
            print(f"=== {lang} {src}: {words:,} words sampled; changes per 1,000 words {per}")


if __name__ == "__main__":
    main()
