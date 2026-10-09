"""Corpus audit (reports/audit_v1.md; HANDOFF §12 2026-10-05 audit spec, 2026-10-06/07 additions).

Reads the stage MANIFESTs (selected, dedup, para_dedup, filtered) and the metadata columns of the
filtered parquet; writes one markdown report. Token counts are ESTIMATES (words x TOKENS_PER_WORD)
until the tokenizer exists; every table says so.

Sections: pipeline yield per source; unique training text by year x language and by period; held-out,
Val and Test-A/B/C sizes (American Stories, the only scored source); document dedup per source at the
per-source thresholds and at a uniform 0.80; paragraph dedup (2a held-out protection hits per source
and cutoff; 2b reprint removals); OCR gate outcomes per source x period; C1 hits per term x year x
split; science bucket by language x decade x source with licence and keyed/OCR share; the training mixture
(src.data.mixture): unique vs seen tokens per period x language x category, caps and any shortfall.

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


def mixture_section(cfg: dict) -> list[str]:
    """Section 8 from data/mixture/MANIFEST.json (src.data.mixture): unique vs seen tokens per period x
    language x category, period totals against targets, cap checks, science per source."""
    m = load(repo_path(cfg["mixture"]) / "MANIFEST.json") if "mixture" in cfg else {}
    out = ["## 8. Training mixture: unique vs seen tokens (src.data.mixture)", ""]
    if not m:
        return out + ["Not drawn yet: run `python -m src.data.mixture`.", ""]
    p = m["params"]
    prof = p["profiles"][m["profile"]]
    source = m.get("token_source") or f"words x {p['tokens_per_word']} (placeholder until the BPE exists)"
    out += [f"Profile {m['profile']} (budget {fmt(prof['budget'], ' tok')}), drawn {m['created_at']}; tokens: "
            f"{source}. Second epoch only from "
            f"{p['repeat_from_year']}-01-01; within a period the order is by weight exp(-(1939 - year)/"
            f"{p['half_life_years']}); caps per period: German <= {p['caps']['german']:.0%}, books <= "
            f"{p['caps']['books']:.0%}, legal <= {p['caps']['legal_of_english']:.0%} of the period's English "
            f"(absolute: {p['caps'].get('legal_absolute', {})}). User decisions 2026-10-08 (HANDOFF §12)."
            + (f" The {prof['fill_to_budget']} target is raised from {fmt(prof['periods'][prof['fill_to_budget']], ' tok')} "
               f"to {fmt(m['period_targets_used'][prof['fill_to_budget']], ' tok')} to fill the budget left by the caps "
               f"(user, 2026-10-08)." if prof.get("fill_to_budget") else ""), ""]
    order = {"1930-39.06": 0, "1920-29": 1, "1900-19": 2, "science": 3}
    cells = sorted(m["cells"], key=lambda c: (order.get(c["period"], 9), c["lang"], c["category"]))
    out += table(["period", "lang", "category", "unique", "seen", "of which 2nd epoch", "seen / unique"],
                 [[c["period"], c["lang"], c["category"], fmt(c["unique_tokens"], " tok"), fmt(c["seen_tokens"], " tok"),
                   fmt(c["second_epoch_tokens"], " tok"),
                   f"{c['seen_tokens'] / c['unique_tokens']:.2f}" if c["unique_tokens"] else "-"] for c in cells])
    ck = m["checks"]
    rows = []
    for per in ("1930-39.06", "1920-29", "1900-19"):
        c = ck[per]
        rows.append([per, fmt(c["target"], " tok"), fmt(c["seen"], " tok"), fmt(c["second_epoch"], " tok"),
                     f"{c['german_share']:.1%}", f"{c['books_share']:.1%}", f"{c['legal_share_of_english']:.1%}",
                     fmt(c["legal_seen"], " tok")])
    out += table(["period", "target", "seen", "2nd epoch", "German share", "books share", "legal / English",
                  "legal seen"], rows)
    out += table(["science", "target", "seen", "share of the language's seen tokens"],
                 [[lang, fmt(ck[f"science_{lang}"]["target"], " tok"), fmt(ck[f"science_{lang}"]["seen"], " tok"),
                   f"{ck[f'science_{lang}']['share_of_language']:.1%}"] for lang in ("en", "de")])
    total, budget = m["total_seen"], prof["budget"]
    out += [f"Total seen: {fmt(total, ' tok')} of the {fmt(budget, ' tok')} budget"
            + ("" if total >= 0.999 * budget else f" (SHORT by {fmt(budget - total, ' tok')}: the run is shorter)")
            + f". Max count {ck['max_count']}; second-epoch rows before {p['repeat_from_year']}: "
            f"{ck['second_epoch_rows_before_repeat_year']}; general rows before 1900 selected: "
            f"{ck['general_pre1900_rows_selected']}.", ""]
    sci = sorted((s for s in m["sources"] if s["period"] == "science"), key=lambda s: (s["lang"], -s["seen_tokens"]))
    out += ["Science by source:", ""]
    out += table(["lang", "source", "unique", "seen"],
                 [[s["lang"], s["source"], fmt(s["unique_tokens"], " tok"), fmt(s["seen_tokens"], " tok")] for s in sci])
    out += [f"Selection files: " + ", ".join(f"`{o['file']}` ({o['rows']:,} rows, sha256 {o['sha256'][:12]})"
                                             for o in m["outputs"].values()), ""]
    tw = load(repo_path(cfg["mixture"]) / "MANIFEST_twin.json")
    if tw:
        main_seen = m.get("period_seen", {})
        out += ["### The English-only twin (compute-matched; user, 2026-10-09)", "",
                "Each period's German seen tokens are replaced by English of the same period (1930s: second "
                "epochs of 1930-33 English; 1920s and pre-1920: the period redrawn without German; science "
                "likewise); never a third epoch. Shortfall = what could not be replaced under these rules.", ""]
        out += table(["period", "main seen", "of which German", "twin seen", "shortfall"],
                     [[per, fmt(main_seen.get(per, 0), " tok"), fmt(tw["replaced_german"].get(per, 0), " tok"),
                       fmt(tw["period_seen"].get(per, 0), " tok"), fmt(tw["shortfall"].get(per, 0), " tok")]
                      for per in ("1930-39.06", "1920-29", "1900-19")]
                     + [["science", fmt(main_seen.get("science", 0), " tok"), "",
                         fmt(tw["period_seen"].get("science", 0), " tok"), fmt(tw["shortfall"].get("science", 0), " tok")],
                        ["total", fmt(m["total_seen"], " tok"), "", fmt(tw["total_seen"], " tok"),
                         fmt(m["total_seen"] - tw["total_seen"], " tok")]])
        if tw.get("filled"):
            out += ["1930s German replaced by: " + ", ".join(f"{k.replace('_', ' ')} {fmt(v, ' tok')}"
                                                             for k, v in tw["filled"].items()) + ".", ""]
        for k, v in tw.get("known_differences", {}).items():
            out += [f"Known difference: {k.replace('_', ' ')}: {fmt(v, ' tok')}.", ""]
        order = {"1930-39.06": 0, "1920-29": 1, "1900-19": 2, "science": 3}
        cells = {}
        for side, man in (("main", m), ("twin", tw)):
            for c in man["cells"]:
                cells.setdefault((c["period"], c["lang"], c["category"]), {})[side] = c["seen_tokens"]
        rows = [[p, l, c, fmt(v.get("main", 0), " tok"), fmt(v.get("twin", 0), " tok"),
                 ("+" if v.get("twin", 0) >= v.get("main", 0) else "-") + fmt(abs(v.get("twin", 0) - v.get("main", 0)), " tok")]
                for (p, l, c), v in sorted(cells.items(), key=lambda kv: (order.get(kv[0][0], 9), kv[0][1], kv[0][2]))
                if v.get("main", 0) or v.get("twin", 0)]
        out += ["Main and twin side by side (seen tokens per period x language x category):", ""]
        out += table(["period", "lang", "category", "main", "twin", "twin - main"], rows)
    return out


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

    for lang in ("en", "de"):
        f = scan_filtered(cfg, lang)
        agg, man = f["agg"], f["manifest"]
        tpw = TOKENS_PER_WORD[lang]
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
                f"Val (all sources, not only American Stories): {fmt(agg['val_words']['all'], ' words')}.", ""]
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
    out += mixture_section(cfg)
    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
