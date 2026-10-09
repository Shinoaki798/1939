# Operations log

Machines: **5080** = remote RTX 5080 box (Windows 11 + WSL2, SSH alias `gpu`, repo `/home/an/1939`);
**2080** = local PC (RTX 2080 SUPER, repo `C:\Users\27409\Desktop\1939`). Times EDT. Commits on branch
`data/corpus-v1` unless noted.

## 2026-10-01 — redesign

- Instructor feedback by email; `docs/HANDOFF.md` written as the single source of truth (earlier 8-page,
  six-run and multi-cutoff drafts abandoned).

## 2026-10-05 — downloads and ingest begin (5080)

- Repo scaffold; resumable, sha256-verified American Stories downloader; per-year article ingest;
  downloader runs outside the SSH session (WMI-launched `wsl.exe`).
- Multi-source downloader (proxy/mirror re-chosen per attempt), language filter for American Stories
  (foreign-language titles), adapters for Congressional Record, HMD, books, Chronicling America pages,
  Federal Register (839 issues 1936-03-14 … 1939-06-30, 2.98 GB), Caselaw Access Project, DDB,
  Europeana, Völkischer Beobachter.
- Inventory (`reports/data_status_2026-10-05.md`, raw, pre-cutoff): English ≈ 51.3B words (American
  Stories 27.89B, pre-1929 books 11.11B, LoC PD books 7.87B, …).
- German OCR scorer (period lexicon from DTA ≤ 1938) and quality report; threshold 0.75 chosen.
- CAP volume files overwrote each other (repeated volume numbers across reporters): renamed
  `<reporter>__<vol>`, 214 kept, 120 refetched.
- Proposal finalised (3 pages) and merged.

## 2026-10-06 — OCR gates, science bucket, downloads move to the 2080

- OCR gates confirmed (0.75 / 0.70 / 50 tokens; 20 for article-level sources); lexicon v2 for both
  languages; English OCR check (American Stories median hit rate 0.98).
- Science bucket built: archive.org catalog tool and 20 archive.org sources, JFM OAI harvester,
  Project Gutenberg selection, Dingler, USGS Professional Papers, Popular Science Monthly (Wikisource),
  science-book selection from the book sets (5,742 books, ~531M words).
- Chronicling America: 124-batch survey; batches fetched and ingested on the 2080 (the 5080's VPN is
  metered) and shipped as parquet via Baidu Netdisk; merged on the 5080.
- Proposal rewritten for the main + English-only twin design.
- Speed test: archive.org 2.8-5.4 s/file on the 2080 vs 7-20 s on the 5080 → all remaining downloads
  on the 2080; the 5080's download chains stopped.
- archive.org fixes: storage-replica fallback on HTTP 5xx, give up on 401/403 lending copies; JFM
  harvester rebuilt around per-walk page keys after resumption tokens were lost.
- 2080 jobs launched through WMI with hidden windows (Start-Process children died when the Claude app
  restarted; a visible window was closed by hand).

## 2026-10-07 — ingest finished, cleaning stages written

- 01:43 Windows Update restarted the 2080; jobs relaunched at 11:12.
- Ingested on the 2080: German archive.org sources, Gutenberg en (1,906 books, 122.5M words) and de
  (133, 9.0M), Dingler (43,811 articles, 71.5M words); hand-over packed: 23 sources, 799M words,
  2.28 GB tar. JFM stopped at 99.79 %: 137,372 unique reviews, 18.1M words, second tar (93 MB).
- Stages written with tests: `select` (pool, language, date classes, pre-1920 sample, German
  typography), `dedup` (exact + MinHash LSH, keep earliest), `splits`, `para_dedup` (2a / 2b),
  `filter` (paragraph removal, chunking, OCR gates, C1 screen), `audit`; chain `scripts/clean_run.sh`.
- Dedup experiment on the 2080 data (`reports/dedup_experiment_2080.md`) → per-source thresholds.
- ~22:30 both hand-overs merged on the 5080.

## 2026-10-07/08 night — selection and dedup on the 5080

- Selection: American Stories 219.7M rows → English pre-cutoff 49.4M articles / 7.16B words (after
  the pre-1920 sample), post-cutoff 12.8M / 1.87B; German-language American Stories 368k articles.
- MinHash signing with 8 workers (14 ran WSL out of memory).
- First matching run read 29 TB from a 33.6 GB memory-mapped signature file in 3.4 h → matching
  rebuilt (8-bit b-bit signatures in RAM, vectorised union-find, documents < 25 words exact-only,
  streamed band keys instead of 50 GB of temp files).
- English dedup: 65,709,836 documents, 3,656,425 dropped (902,637 exact, 2,753,788 near; 1,834,910 at
  a uniform 0.80). German dedup done.
- WSL memory raised to 48 GB + 16 GB swap; background apps closed on the 5080.

## 2026-10-08 — paragraph dedup, filter, audit, mixture

- 02:39 chain `heldout reprint_en …`: 2a (held-out protection) ran in 8 min; 2b crashed out of memory
  (one worker 22.5 GB). Fixed (per-language 2b sources, previous year streamed into a Bloom filter,
  reusable year shards); 2b English 22 min.
- Hardware monitoring installed on the 5080 (PawnIO driver + LibreHardwareMonitor library,
  `scripts/hw_temps.ps1`); hourly watcher with temperature / memory alarms.
- 03:43 chain stopped: CPU at 88.6 °C / 154 W with 14 filter workers, and German pages found to be one
  block each in paragraph dedup. Fixed (sentence-end blocks for single-line pages; 2a also on the German
  pool; 8 workers ≈ 80 °C); 2a/2b rerun.
- 04:30-05:05 SSD write counter rising ~160 GB/h: Windows paging because 8 filter workers held ~5 GB
  each. Filter now streams 20k-row groups (workers 1.4-1.7 GB); paging stopped.
- 05:08-05:40 filter English (31 min), 05:40-05:59 German (20 min); 06:00 `reports/audit_v1.md`.
- Mixture accounting bug in the audit (science counted twice, shares summed to 111 %) fixed.
- Stale half-written filter output (18 GB) deleted; Downloads hand-over leftovers (4.4 GB) moved to the
  Recycle Bin (user OK).
- Quality read of the filtered training text (`reports/mixture_samples_2026-10-08.md`) and artefact
  rates per source.
- `src/data/mixture.py` drawn: first 11.11B (caps), then with the 1920s fill 12.00B
  (`data/mixture/{en,de}/<sha12>.parquet`); audit §8 shows unique vs seen per period × language ×
  category.
- `src/data/normalize.py` (training-only text fixes) written and measured on the filtered data.
- `progress_report/material/` started (this folder).
- Review (user question: tokenizer size; is data collection over?). Findings: config/model_ladder.yaml
  (vocab 32768, context 1024), TASKS Phase 1 ("32k BPE, English only") and HANDOFF §5.4/§6.1 still carry
  the pre-2026-10-05 tokenizer and context; HF `tokenizers` is installed on the 5080, `sentencepiece` is
  not. RQ3 context supply in the filtered data: English 1939-06 73,440 American Stories articles, 08-18..31
  31,678; German 1939-06 ~2,200 pages (DDB 1,665, Europeana 563), 08-18..31 ~700 pages (DDB 447,
  Europeana 257; German-language American Stories is OCR-gated). ChroniclingAmericaQA (sanity
  criterion 5) is not in `config/sources.yaml` and has not been downloaded.
- ChroniclingAmericaQA registered as an evaluation-only source (`config/sources.yaml`, MIT, revision
  15e58335; dev.json 75 MB + test.json 75 MB, the 1.38 GB train.json skipped; dest `data/eval/`);
  `select.pool_sources` now also skips every "evaluation only" source. Downloaded on the 5080 through the
  VPN (user OK, VPN back on).
- ChroniclingAmericaQA downloaded on the 5080 (15:45-15:46, VPN, both files sha256-verified):
  test 24,084 / dev 24,111 questions; fields question, answer, context (corrected paragraph), raw_ocr,
  publication_date, url. Test by decade: 1800s-1890s 19,097, 1900s 2,253, 1910s 2,516, 1920s 218. Note
  for `sanity_caqa.py`: the 1900-1919 contexts come from the same Chronicling America scans as American
  Stories, so their text may be in training (fine for a sanity check, to be stated).
- ChroniclingAmericaQA train.json (1.38 GB, 62 s through the VPN, sha256-verified) stored as fine-tuning
  material for post-project exploration.
- Tokenizer comparison started on the 5080 (`scripts/tokenizer_run.sh`: sample → one 65,536 BPE run →
  32,768 / 49,152 as merge prefixes → `reports/tokenizer_compare.md`). Unit test confirms a prefix of a
  longer run equals a run stopped there (identical vocabulary and encodings).
- Tokenizer comparison done on the 5080 (16:00-16:03): sample 615,194 documents (EN 596,417 / 1.03 GB,
  DE 18,777 pages / 0.23 GB; 7,373 dropped at ocr_hit < 0.85); one 65,536 BPE run in 129 s; 32k/48k cut as
  prefixes. Held-out bytes per token, EN American Stories / DE: 32k 3.592 / 3.327, 48k 3.688 / 3.480, 64k
  3.743 / 3.582; tokens per word EN 1.593 / 1.551 / 1.529, DE 2.002 / 1.914 / 1.860 (the 1.35 / 1.6
  placeholders were ~15 % low). OCR junk among added word-initial tokens 0.0 / 0.3 / 0.7 %. Probe terms
  that are single tokens: blitz, occupation, resistance (+ collaboration at 64k), all C3 sense-shift words
  (ordinary pre-1939 words); no C1 coinage is a single token. `reports/tokenizer_compare.md`.
- Tokenizer frozen: `data/tokenizer/bpe_49152/tokenizer.json` (sha256 a5f5186e…), `config/tokenizer.yaml`;
  `config/model_ladder.yaml` updated (vocab 49,152, context 2048, selected s335m), TASKS/HANDOFF aligned.
  C1/C2 probe check passes (radar → rad|ar, Quisling → Qu|is|ling, Genozid → Gen|oz|id; 1939 → 1|9|3|9).

## 2026-10-09 — cleanup

- Local (2080): the 10-07 pipeline-experiment outputs `data/selected`, `data/dedup`, `data/filtered`,
  `data/para_dedup` (9.5 GB) removed; superseded by the full run on the 5080 (results in
  `reports/dedup_experiment_2080.md`). Every source fetched on the 2080 (15 English, 11 German) was
  confirmed present in the 5080's ingested data; the remaining local copies (`data/raw` except the SCOWL
  lexicon anchor, `data/foreign`, `data/ingested`: 77 GB) and the Desktop transfer packs (4.4 GB) are
  redundant and left for the user to delete with one command (the C: Recycle Bin holds only ~32 GB).
- 5080 cleanup candidates listed for the user (scratch: paragraph-dedup year shards 41 GB, held-out
  shingle array 2 GB, MinHash signature caches 34 GB; raw of supplementary corpora ~150 GB).
- 5080: `data/raw/ncse` (384 MB, excluded source, never ingested, referenced by no MANIFEST) removed
  (user OK). Kept by the user's choice: scratch that a rerun would reuse (paragraph-dedup year shards,
  held-out shingles, MinHash signature caches) and every raw download of the supplementary corpora.
- Disk cleanup on both machines (user-approved lists; Andrew ran the scripts): 2080 C: free 351 → 494 GB
  (redundant project data, transfer packs, caches, some game data, recycle bin, unused WSL distro);
  5080 C: free 372 → 592 GB (NVIDIA shader cache 67 GB, games, caches, recycle bin, hibernation off).
- 5080: WSLg's weston crashed every 103 s and wrote a 171 MB crash dump each time (~6 GB/h, ~140 GB/day
  of SSD writes; most of the 87 GB the SSD counter gained overnight). `.wslconfig` gets
  `guiApplications=false` (user OK), WSL restarted, keepalive task relaunched.
