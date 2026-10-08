"""Corpus audit (reports/audit_v1.md; HANDOFF §12 2026-10-05 audit spec, 2026-10-06/07 additions).

Reads the stage MANIFESTs (selected, dedup, para_dedup, filtered) and the metadata columns of the
filtered parquet; writes one markdown report. Token counts are ESTIMATES (words x TOKENS_PER_WORD)
until the tokenizer exists; every table says so.

Sections: pipeline yield per source; unique training text by year x language and by period; held-out,
Val and Test-A/B/C sizes (American Stories, the only scored source); document dedup per source at the
per-source thresholds and at a uniform 0.80; paragraph dedup (2a held-out protection hits per source
and cutoff; 2b reprint removals); OCR gate outcomes per source x period; C1 hits per term x year x
split; science bucket by language x decade x source with licence and keyed/OCR share; the seen-token
mixture against the caps of CLAUDE.md rule 7 with any shortfall.

    python -m src.data.audit [--out reports/audit_v1.md]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from src.data.download import load_config, repo_path

TOKENS_PER_WORD = {"en": 1.35, "de": 1.6}       # estimate until the 48k BPE exists
BUDGET = 10e9                                   # seen tokens (rule 7)
CAPS = {"pre1920": 0.08, "1920s": 0.35, "german_per_period": 0.25, "books_per_period": 0.12, "legal": 0.10,
        "science_per_language": 0.10}
META = ["source", "category", "bucket", "keyed", "year", "period", "split", "test_set", "n_words", "train_ok",
        "gate", "c1_term"]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fmt(n: float, unit: str = "") -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B{unit}"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M{unit}"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k{unit}"
    return f"{n:.0f}{unit}"


def table(head: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]


def scan_filtered(cfg: dict, lang: str) -> dict:
    """Aggregates over the filtered parquet metadata of one language."""
    m = load(repo_path(cfg["filtered"]) / lang / "MANIFEST.json")
    agg = defaultdict(Counter)
    for o in m.get("outputs", {}).values():
        f = repo_path(cfg["filtered"]) / lang / o["file"]
        t = pq.read_table(f, columns=META).to_pydict()
        for i in range(len(t["source"])):
            s, y, sp, w, ok = t["source"][i], t["year"][i], t["split"][i], t["n_words"][i], t["train_ok"][i]
            cat = t["category"][i]
            per = t["period"][i]
            if ok:
                agg["train_words_year"][y] += w if sp == "train" else 0
                agg["val_words"]["all"] += w if sp == "val" else 0
                key = "science" if t["bucket"][i] == "science" else per
                agg["unique_by_period_cat"][f"{key}|{cat}"] += w if sp == "train" else 0
                if t["bucket"][i] == "science":
                    agg["science_decade_source"][f"{(y // 10) * 10}s|{s}"] += w if sp == "train" else 0
                    agg["science_keyed"]["keyed" if t["keyed"][i] else "ocr"] += w if sp == "train" else 0
            if s == "american_stories":
                if sp == "holdout":
                    agg["as_holdout_year_docs"][y] += 1
                    agg["as_holdout_year_words"][y] += w
                    if t["test_set"][i]:
                        agg["as_test_docs"][t["test_set"][i]] += 1
                        agg["as_test_words"][t["test_set"][i]] += w
                elif sp == "val":
                    agg["as_val"]["docs"] += 1
                    agg["as_val"]["words"] += w
    return {"manifest": m, "agg": agg}


def mixture(unique: dict[str, dict[str, float]]) -> list[list]:
    """Seen-token plan per language from unique train tokens (estimates). unique[lang][key] with key in
    1930-39.06|<cat>, 1920-29|<cat>, 1900-19|<cat>, <1900|<cat>, science|science."""
    def period_total(lang_vals: dict[str, dict[str, float]], period: str, epochs: float) -> dict[str, float]:
        """Seen tokens of one period across languages with the German, books and legal caps applied."""
        en = {c: v * epochs for k, v in lang_vals["en"].items() if k.startswith(period + "|") for c in [k.split("|")[1]]}
        de = sum(v * epochs for k, v in lang_vals["de"].items() if k.startswith(period + "|"))
        free = sum(v for c, v in en.items() if c not in ("books", "legal"))
        books, legal = en.get("books", 0), en.get("legal", 0)
        total = free + books + legal + de
        for _ in range(50):                                   # caps are shares of the period total
            b, l, d = min(books, CAPS["books_per_period"] * total), min(legal, CAPS["legal"] * total), \
                min(de, CAPS["german_per_period"] * total)
            new = free + b + l + d
            if abs(new - total) < 1:
                break
            total = new
        return {"en_free": free, "books": b, "legal": l, "de": d, "total": total,
                "available": free + books + legal + de}
    rows = []
    p30 = period_total(unique, "1930-39.06", 2.0)
    sci = {lang: unique[lang].get("science|science", 0) for lang in ("en", "de")}
    rest = BUDGET - p30["total"]
    p20 = period_total(unique, "1920-29", 1.0)
    p20_seen = min(p20["total"], CAPS["1920s"] * BUDGET, max(rest, 0))
    pre = period_total({lang: {k.replace("1900-19", "pre").replace("<1900", "pre"): v for k, v in unique[lang].items()}
                        for lang in unique}, "pre", 1.0)
    pre_seen = min(pre["total"], CAPS["pre1920"] * BUDGET, max(rest - p20_seen, 0))
    general = p30["total"] + p20_seen + pre_seen
    f20 = p20_seen / p20["total"] if p20["total"] else 0.0
    fpre = pre_seen / pre["total"] if pre["total"] else 0.0
    by_lang = {"en": sum(f * (p["en_free"] + p["books"] + p["legal"]) for p, f in ((p30, 1.0), (p20, f20), (pre, fpre))),
               "de": sum(f * p["de"] for p, f in ((p30, 1.0), (p20, f20), (pre, fpre)))}
    share = CAPS["science_per_language"] / (1 - CAPS["science_per_language"])     # science <= 10 % of the language
    sci_seen = {lang: min(2 * sci[lang], share * by_lang[lang]) for lang in sci}
    total = min(BUDGET, general + sum(sci_seen.values()))
    rows.append(["1930-1939.06 (x2)", fmt(p30["available"], " tok"), fmt(p30["total"], " tok"),
                 f"{100 * p30['total'] / total:.1f} %", "repeated twice; German <= 25 %, books <= 12 %, legal <= 10 %"])
    rows.append(["1920-1929 (x1)", fmt(p20["total"], " tok"), fmt(p20_seen, " tok"), f"{100 * p20_seen / total:.1f} %",
                 "fills the remainder, <= 35 %"])
    rows.append(["pre-1920 (x1)", fmt(pre["total"], " tok"), fmt(pre_seen, " tok"), f"{100 * pre_seen / total:.1f} %",
                 "<= 8 %"])
    for lang in ("en", "de"):
        rows.append([f"science {lang} (<= x2)", fmt(sci[lang], " tok"), fmt(sci_seen[lang], " tok"),
                     f"{100 * sci_seen[lang] / total:.1f} %", "<= 10 % of the language's seen tokens"])
    rows.append(["total", "", fmt(total, " tok"), "100 %",
                 "budget 10B" + ("" if total >= BUDGET * 0.999 else f"; SHORT by {fmt(BUDGET - total, ' tok')}")])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reports/audit_v1.md")
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    sel_root = repo_path(cfg["selected"])
    out = [f"# Corpus audit v1 ({dt.date.today().isoformat()})", "",
           "Generated by `python -m src.data.audit` from the stage MANIFESTs. Words are counted; tokens are "
           f"estimates (words x {TOKENS_PER_WORD}) until the tokenizer exists.", ""]

    # 1. pipeline yield
    out += ["## 1. Pipeline yield per source (documents / words)", ""]
    rows = []
    for m in sorted(sel_root.glob("*/MANIFEST.json")):
        man = load(m)
        st = Counter()
        for e in man["files"].values():
            st.update(e["stats"])
        kept = sum(v for k, v in st.items() if k.startswith("keep_"))
        langd = sum(v for k, v in st.items() if k.startswith("drop_lang"))
        rows.append([man["source"], f"{st['rows_in']:,}", f"{kept:,}", f"{langd:,}", f"{st['drop_date']:,}",
                     f"{st['drop_pre1920_sample']:,}", f"{st['drop_ejc_not_science']:,}"])
    out += table(["source", "ingested", "selected", "language drop", "date drop", "pre-1920 sample drop", "EJC non-science"], rows)

    unique: dict[str, dict[str, float]] = {}
    for lang in ("en", "de"):
        f = scan_filtered(cfg, lang)
        agg, man = f["agg"], f["manifest"]
        tpw = TOKENS_PER_WORD[lang]
        unique[lang] = {k: v * tpw for k, v in agg["unique_by_period_cat"].items()}
        out += [f"## 2. {lang}: unique training text", ""]
        rows = [[k.split("|")[0], k.split("|")[1], fmt(v, " w"), fmt(v * tpw, " tok")]
                for k, v in sorted(agg["unique_by_period_cat"].items())]
        out += table(["period", "category", "words", "tokens (est.)"], rows)
        yrs = sorted(agg["train_words_year"])
        out += ["By year (train split, after every filter):", ""]
        out += table(["year", "words", "tokens (est.)"],
                     [[y, fmt(agg["train_words_year"][y], " w"), fmt(agg["train_words_year"][y] * tpw, " tok")]
                      for y in yrs if agg["train_words_year"][y]])
        p2539 = sum(agg["train_words_year"][y] for y in yrs if 1925 <= y <= 1939)
        out += [f"1925-1939.06 total: {fmt(p2539, ' words')} ({fmt(p2539 * tpw, ' tokens est.')}); "
                f"Val: {fmt(agg['val_words']['all'], ' words')}.", ""]
        if lang == "en":
            out += ["## 3. Held-out and test sets (American Stories, the only scored source)", ""]
            out += table(["set", "articles", "words"],
                         [[f"Test-{k}", f"{agg['as_test_docs'][k]:,}", fmt(agg['as_test_words'][k])] for k in "ABC"] +
                         [["Val", f"{agg['as_val']['docs']:,}", fmt(agg['as_val']['words'])]])
            out += ["Per-year held-out (2 %):", ""]
            out += table(["year", "articles", "words"], [[y, f"{agg['as_holdout_year_docs'][y]:,}",
                                                         fmt(agg['as_holdout_year_words'][y])]
                                                        for y in sorted(agg["as_holdout_year_docs"])])
        # document dedup
        dm = load(repo_path(cfg["dedup"]) / lang / "MANIFEST.json")
        out += [f"## 4. {lang}: document dedup (thresholds per source; uniform 0.80 for comparison)", ""]
        rows = []
        for s, c in sorted(dm.get("per_source", {}).items()):
            d = c.get("dropped_exact", 0) + c.get("dropped_near", 0)
            rows.append([s, dm["threshold_per_source"].get(s), f"{c['docs']:,}", f"{d:,}",
                         f"{100 * d / max(c['docs'], 1):.2f} %", f"{100 * c.get('dropped_words', 0) / max(c['words'], 1):.2f} %",
                         f"{100 * c.get('dropped_at_0.80', 0) / max(c['docs'], 1):.2f} %"])
        out += table(["source", "threshold", "documents", "dropped", "docs %", "words %", "docs % at 0.80"], rows)
        # paragraph dedup
        pd = repo_path(cfg["data_root"]) / "para_dedup" / lang
        for name, title in (("heldout", "2a held-out protection (hits per cutoff; applied at 0.5)"),
                            ("reprint", "2b reprints within the pool (applied at 0.5)")):
            pm = load(pd / f"MANIFEST_{name}.json")
            if not pm:
                out += [f"### {lang}: {title}: not run", ""]
                continue
            out += [f"### {lang}: {title}", ""]
            rows = []
            for s, c in sorted(pm["per_source"].items()):
                hits = " / ".join(f"{k[5:]}: {v:,}" for k, v in sorted(c.items()) if k.startswith("hits@"))
                words = c.get("words@0.5", 0)
                rows.append([s, f"{c.get('docs', 0):,}", f"{c.get('blocks', 0):,}", hits, fmt(words, " w")])
            out += table(["source", "training docs", "blocks", "blocks hit", "words removed at 0.5"], rows)
        # OCR gates
        out += [f"## 5. {lang}: OCR gates and C1 screen (train split; held-out never gated)", ""]
        if man:
            gp = man["gate_params"]
            out += [f"Gates: hit rate >= {gp['hit_threshold']}, word share >= {gp['word_share_threshold']}, "
                    f"tokens >= {gp['min_doc_tokens_item_level']} (item-level) / {gp['min_doc_tokens']} (pages); "
                    f"lexicon {gp['lexicon_file']} ({gp['lexicon_version']}); histograms in reports/ocr_gates_{lang}.md.", ""]
            rows = []
            for s, c in sorted(man["per_source"].items()):
                for per in ("<1900", "1900-19", "1920-29", "1930-39.06"):
                    w = c.get(f"{per}|train|words", 0)
                    if not w:
                        continue
                    cells = [f"{100 * c.get(f'{per}|train|{r}_words', 0) / w:.1f} %" for r in ("ok", "short", "hit_rate", "word_share", "c1")]
                    rows.append([s, per, fmt(w, " w")] + cells + [fmt(c.get(f"{per}|para_removed_words", 0), " w")])
            out += table(["source", "period", "train words", "kept", "short", "hit rate", "word share", "C1", "paragraphs removed"], rows)
            out += ["C1 hits, kept in training (rule 6 as revised 2026-10-07; term: year|split -> documents; "
                    "the C1 column above is the share of words in documents with a hit):", ""]
            for term, c in sorted(man.get("c1_hits", {}).items()):
                out.append(f"- {term}: {sum(c.values())} hits; " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
            out.append("")
        # science
        out += [f"## 6. {lang}: science bucket (train words after filters)", ""]
        rows = [[k.split("|")[0], k.split("|")[1], fmt(v, " w")] for k, v in sorted(agg["science_decade_source"].items())]
        out += table(["decade", "source", "words"], rows)
        ks = agg["science_keyed"]
        out += [f"Keyed {fmt(ks['keyed'], ' w')} vs OCR {fmt(ks['ocr'], ' w')} "
                f"({100 * ks['keyed'] / max(ks['keyed'] + ks['ocr'], 1):.1f} % keyed).", ""]
    # licences
    sources = load_config(repo_path(cfg["sources"]))
    out += ["## 7. Licence per science source", ""]
    out += table(["source", "licence"], [[n, str(v.get("licence", ""))[:140]] for n, v in sources.items()
                                         if isinstance(v, dict) and v.get("bucket") == "science"])
    out += ["## 8. Seen-token mixture against the caps (estimates)", ""]
    out += table(["slice", "available (unique x epochs)", "planned seen", "share", "rule"], mixture(unique))
    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
