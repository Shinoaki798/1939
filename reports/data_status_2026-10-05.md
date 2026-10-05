# Data collection status — 2026-10-05

Project: APS360 "The Shape of a Knowledge Boundary" — one decoder-only Transformer
trained from scratch on text published before the cutoff **1939-06-30** (embargo
1939-07-01…08-31). Design and rules: `docs/HANDOFF.md` (wins on conflict; decisions
log in §12), `CLAUDE.md`, `docs/TASKS.md`. Branch with all data work: `data/corpus-v1`.

## Where things are

| What | Where |
|---|---|
| Repo (local) | `C:\Users\27409\Desktop\1939` ↔ github.com/Shinoaki798/1939 |
| Repo + data (remote RTX 5080 box, WSL2) | `/home/an/1939` (SSH alias `gpu`) |
| English raw / ingested | `data/raw/<source>/`, `data/ingested/<source>/` |
| German raw / ingested (kept apart) | `data/foreign/de/raw/<source>/`, `data/foreign/de/ingested/<source>/` |
| Approved sources, pinned revisions | `config/sources.yaml` + `config/<source>_files.tsv` (size + sha256/md5 per file) |
| Code | `src/data/{download,ingest,ingest_extra,filters,prune_raw}.py`; `scripts/` for the remote box |

Every downloaded file was verified against a published checksum (HF LFS sha256 or the
publisher's md5); each stage writes a `MANIFEST.json`. All sources are text (OCR or
hand-keyed), no page images.

## Inventory (pre-cutoff portion, raw: before dedup / OCR filter / language filter)

Pre-cutoff = issue date ≤ 1939-06-30 where a day date exists, else year ≤ 1938.
Sources of the numbers (remote box): ingested sources from `logs/ingested_stats.json`
(computed from the ingested parquet); raw sources from `logs/census_raw.json`
(`scripts/census_raw.py`; JSTOR recounted with a fixed member filter).
Words = whitespace tokens; expect ≈1.3 BPE tokens per English word.

### English — ≈ 51.3 B words

| Source | Genre / span | Units | Words | Status |
|---|---|---|---|---|
| American Stories (backbone; only source of scored sets) | US newspapers, article level, 1900–1939-06 | 204 M articles | **27.89 B** (26.31 B voted English) | ingested |
| ↳ same, 1939-09…1955 (test only, never trained) | | | 2.08 B | ingested |
| common-pile/pre_1929_books | US PD books, ≤1928 | 137 k books | **11.11 B** | raw |
| storytracer/LoC-PD-Books | US PD books, kept ≤1938 | 129 k books | **7.87 B** | raw |
| biglam/hmd_newspapers | UK newspapers 1800–1896 | 3.06 M articles | **2.39 B** | ingested (raw pruned) |
| JSTOR Early Journal Content | academic journals, ≤1922 | 452 k articles | **1.12 B** | raw |
| Congressional Record (Congresses 43–76) | US floor speeches 1873–1939-06 | 7.54 M speeches | **0.53 B** | ingested (raw pruned) |
| ECCO-TCP | 18th-c. British texts, hand-keyed | 3.1 k | 0.11 B | raw |
| Evans-TCP | early American imprints, hand-keyed | 5.0 k | 0.10 B | raw |
| Royal Society Corpus 6.0 open | Phil. Trans. 1665–1920 | 17.5 k | 0.08 B | raw |
| NCSE (Vintage-LLM) | UK periodicals 1806–1890 (VLM re-OCR) | 445 k boxes (dups removed) | 0.07 B | raw |

### German — ≈ 17.6 B words (stored apart; **use undecided**)

| Source | Genre / span | Units | Words | Notes |
|---|---|---|---|---|
| storytracer/German-PD-Newspapers (DDB) | newspapers, page level, day dates | 5.33 M pages | **12.75 B** | Fraktur OCR, noisy; no OCR score |
| biglam/europeana_newspapers (de-1900…de-1930) | newspapers 1900–1939-06 | 1.77 M pages | **4.66 B** | has `mean_ocr` (~0.55 in the 1930s); overlaps DDB |
| Deutsches Textarchiv (2026-02-10) | hand-keyed TEI, literature/science/news | ~5.0 k | ~0.21 B | year from filename for now; TEI header at ingest |

## Caveats

- Counts are raw. Wire-service reprints (American Stories), book overlap
  (pre_1929 ↔ LoC), and DDB ↔ Europeana overlap will shrink them; MinHash dedup runs on
  the full pool before splitting (CLAUDE.md rule 4).
- American Stories contains foreign-language titles; filter = article stopword vote +
  per-title-year English share (`src/data/filters.py`). ~5 % of scans lack an `lccn`
  block; kept, LCCN taken from the filename.
- Year-only sources can be misdated (books, serials) — main contamination risk;
  plan an anachronism screen with the RQ1 C1 coinages as a backstop.
- Non-commercial licences: JSTOR EJC, Royal Society Corpus (record in manifests).
- Compute, not data, is the binding constraint: even after heavy filtering the English
  pool far exceeds the ~6.7 B tokens of the 335 M rung; the user allows a larger model.

## Open decision for the reviewer: how German text is used

Facts that bound the choice:
- **Rules in force** (`CLAUDE.md`): 5 translated text never scored; 6 any translated
  sentence with an RQ1 probe term is dropped whole; 10 the BPE is trained on
  native-English pre-cutoff text only; 11 translation can never block graded work.
  HANDOFF §5.5 plans sentence-level NMT (NLLB-200, pinned) → English, training-only.
- **Proposal** (`docs/proposal.tex`, due 2026-10-16, not yet submitted) describes DE/FR
  press "machine-translated into English". RQ3 uses US/DE/FR conditioning contexts.
- **Hardware/schedule**: one RTX 5080 (remote) does processing, baselines (10-17…10-24)
  and the Transformer run (10-25…11-02); translation must never compete with training.
  The local PC has an RTX 2080 SUPER 8 GB. v1 freeze 10-10, v2 freeze 10-24.
- **Scale**: 17.6 B German words raw; NMT throughput on one GPU is on the order of a few
  B words before 10-24, so a translation route uses only a fraction.

Options to decide between (or a variant):
- **A. Translate → English (current plan).** No rule changes; tokenizer stays English;
  needs a translation pipeline + the three defences; only a slice of the German data fits.
- **B. Train on German directly.** All German usable, no translator bias; requires
  changing rule 10 (multilingual BPE, larger vocab), the proposal wording (still
  editable), and a cap on the German share so English capacity is kept; all scored
  sets stay English.
- **C. Keep German out of the Transformer** (e.g. RQ3 contexts only, or not at all).

Please return: the choice, the reason, and which rules/docs must change.

## Next steps (corpus v1, freeze 2026-10-10)

1. Ingest adapters for the 10 raw sources; prune their raw files after verified ingest.
2. MinHash near-dedup on the full English pool → per-year 2 % holdout → OCR-quality
   column and threshold (user confirms) → splits with manifest.
3. 32 k BPE on native-English Train; tokenise; `reports/audit_v1.md`; choose model size.
