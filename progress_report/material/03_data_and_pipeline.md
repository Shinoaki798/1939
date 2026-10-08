# Data and pipeline (numbers as of 2026-10-08)

Tokens are estimates (words x 1.35 for English, x 1.6 for German) until the BPE is trained; they will
be re-derived then. Sources of the numbers: `reports/audit_v1.md`, `data/*/MANIFEST.json`.

## Pipeline

| stage | code | what it does | key result |
|---|---|---|---|
| download | `src/data/download.py`, `scripts/fetch_local.py` | checksummed downloads (5080 via proxy; 2080 unmetered) | ~40 sources |
| ingest | `src/data/ingest.py`, `ingest_extra.py` | one row per article / page / book / review with date and metadata | — |
| OCR score | `src/data/ocr_quality.py` | period-lexicon hit rate + word-like share per document | gates 0.75 / 0.70 |
| select | `src/data/select.py` | pool, language vote, date classes, pre-1920 recency sample, German typography | en 49.4M AS articles pre-cutoff |
| dedup | `src/data/dedup.py` | exact + MinHash (5-gram, 128 perm, LSH 32x4; OCR 0.60 / keyed 0.80), keep earliest | en 3.66M of 65.7M dropped |
| splits | `src/data/splits.py` | hash of the document id: 2 % per-year holdout, 2 % Val, Test-A/B/C | — |
| paragraph dedup | `src/data/para_dedup.py` | 2a: training paragraphs matching held-out American Stories removed; 2b: wire reprints, keep earliest | en 2a 16M words, 2b 150M; de 2b 155M |
| filter | `src/data/filter.py` | paragraph removal, chunking of long documents, OCR gates (train/val), C1 screen | en 36 GB, de 11 GB |
| audit | `src/data/audit.py` | `reports/audit_v1.md` | — |
| mixture | `src/data/mixture.py` | seen count (0/1/2) per document | 12.00B seen |
| normalise | `src/data/normalize.py` | training-only text fixes at shard time | — |

## Held-out and test sets (American Stories, the only scored source)

| set | articles | words |
|---|---|---|
| Test-A (1930-38 held-out) | 181,156 | 27.1M |
| Test-B (1939-09 … 1945-12) | 124,858 | 18.0M |
| Test-C (1946-55) | 120,602 | 18.3M |
| Val (≤ cutoff) | 931,573 | 137.5M |

## Unique training text after all cleaning (100M tokens, est.)

| period | EN newspapers | EN books | EN legal | EN Congress | EN total | DE (newspapers) | total |
|---|---|---|---|---|---|---|---|
| 1900-1919 | 17.3 | 5.2 | 0.0 | 0.2 | 22.7 | 7.5 | 30.2 |
| 1920-1929 | 52.1 | 14.8 | 3.3 | 1.1 | 71.2 | 17.3 | 88.4 |
| 1930-1939.06 | 27.9 | 0.2 | 5.9 | 1.0 | 35.0 | 12.1 | 47.2 |
| science | | | | | 17.0 | 4.7 | 21.7 |

English newspapers are American Stories in every period, plus Chronicling America pages in the 1930s
(1930s: American Stories 1.33B words, Chronicling America 0.73B words). English 1920-22 holds 1.1-1.2B
words per year against ~0.25B for 1925-29: Chronicling America long covered only public-domain years
(≤ 1922). German: DDB, Europeana, Völkischer Beobachter, Chronicling America German pages; the
German-language American Stories articles are 99 % OCR-gated (Fraktur read as Latin script).

## Training mixture (seen tokens, profile 12B)

| slice | seen | of which second epoch | caps (per period) |
|---|---|---|---|
| 1930-1939.06 (all once, 1934-01 … 1939-06 twice) | 6.60B | 2.12B | German 25.0 %, legal 0.35B (7.1 % of English), books 0.4 % |
| 1920-1929 (sampled, recency-ordered) | 3.45B | 0 | German 25.0 %, books 12.0 %, legal 7.8 % of English |
| 1900-1919 (sampled) | 0.80B | 0 | German 23.0 %, books 12.0 % |
| science EN (JSTOR, Nature, PNAS first) | 0.85B | 0 | 9.4 % of English |
| science DE (Annalen der Physik, JFM first) | 0.30B | 0 | 10.0 % of German |
| **total** | **12.00B** | | max 2 passes; no second epoch before 1934; no general text before 1900 |

## Text normalisation measured on the filtered data (changes per 1,000 words)

| source | fix | rate |
|---|---|---|
| Congressional Record | ". " + lowercase → ", " | 44 |
| Royal Society Corpus | space before punctuation removed | 128 |
| Europeana | repeated word at line breaks removed | 8 |
| DDB | spaces 118, ⸗ joined 11, ⸗ → "-" 6 | |
| Völkischer Beobachter | ⸗ joined 19, ⸗ → "-" 6 | |

## Tokenizer comparison (2026-10-08, `reports/tokenizer_compare.md`)

| vocab | EN bytes/token | EN tokens/word | DE bytes/token | DE tokens/word | embedding (d 1024) | output layer / body FLOPs | OCR junk in added tokens |
|---|---|---|---|---|---|---|---|
| 32,768 | 3.592 | 1.593 | 3.327 | 2.002 | 33.6M | 11.1 % | 0.0 % |
| 49,152 | 3.688 | 1.551 | 3.480 | 1.914 | 50.3M | 16.7 % | 0.3 % |
| 65,536 | 3.743 | 1.529 | 3.582 | 1.860 | 67.1M | 22.2 % | 0.7 % |
