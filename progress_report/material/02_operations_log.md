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
- WSLg fix verified: no crash dump in the 5 minutes after the restart (the loop had produced one every
  103 s); the remaining 1.5 GB of dumps removed; idle disk writes on the 5080 fell from 2.1 MB/s to
  0.27 MB/s.

## 2026-10-09 — twin mixture

- Twin draw implemented (`src/data/mixture.py`, `data/mixture/MANIFEST_twin.json`, tests). Preview on the
  word-based estimates: 1920s and pre-1920 German are fully replaced by English of the same period, and
  science too, but the 1930s fall 0.42B short. The DECISION assumed 2.07B of 1930-33 English for a second
  epoch, while 1930-33 non-legal English is 1.23B (estimate; ~1.4B in real tokens) against 1.65B of 1930s
  German. With no third epoch the twin would total 11.58B. Options put to the user.
- Tokenisation started on the 5080 at 01:59 (26,603 filtered files, 8 workers x 2 threads, CPU ~82 °C).
  At ~02:20 the watcher flagged Windows memory at 1.1 GB: the WSL page cache had grown to 36 GB from
  reading the filtered files, and Windows was paging (up to ~180k pages/s). The page cache was dropped
  (Windows back to 24 GB free, paging 0), and `scripts/drop_cache_loop.sh` now runs as root and drops the
  cache every 2 min until the tokeniser ends.
- Tokenisation finished at 03:11 (72 min; 26,603 files, 56 GB uint16). Round trip checked on 139 EN and
  171 DE documents, no mismatch. Tokens by split: EN train 18.69B, val 0.39B, holdout 0.46B, embargo
  0.03B, post 2.92B; DE train 6.34B, val 0.13B, holdout 0.15B, embargo 0.02B, post 0.29B. Train counts
  include OCR-gated rows, which the mixture skips.
- WSL memory cap lowered to 40 GB (user OK; `.wslconfig.bak-20261009b`), WSL restarted.
- Mixture redrawn on real token counts (+1 <|endoftext|> per document). Main: 1930s 7.37B (target
  reached; second epochs 2.28B, German 25.0 %, legal 0.35B = 6.3 % of English), 1920s 2.68B, pre-1920 0.80B,
  science EN 0.85B + DE 0.30B = 12.00B. Twin (English only): 1930s 7.20B (1930s German 1.84B replaced by
  1.44B of 1930-33 second epochs, 0.23B unused case law, 0.17B moved to the 1920s), 1920s 2.85B, pre-1920
  0.80B, science 1.15B = 12.00B. In the main draw 0.27B of 1934-39 non-legal English is seen only once
  (the 1930s target stops the second epochs); the user is asked whether the twin should complete those
  first.
- Twin redrawn with the user's fill order: 1930s German 1.84B replaced by the 1934-39 second pass 0.27B
  (recorded known difference), 1930-33 second epochs 1.44B and unused 1930s case law 0.14B (twin legal 6.6 %
  of 1930s English); moved to the 1920s ~0. Twin period totals equal the main run's (1930s 7.37B, 1920s
  2.68B, pre-1920 0.80B, science 1.15B). Audit section 8 shows main and twin side by side.
- Document-order question researched (industry practice: Megatron, OLMo, Pythia, Llama 3 document masking,
  best-fit packing, data-constrained repetition, memorisation-by-order, curriculum studies); answer to user.

## 2026-10-14 — proposal revision
- Context length 4096 recorded (config, HANDOFF). Proposal checked against the current design: seven
  statements out of date (C1 handling, twin definition, RQ2 success criterion, token budget, compute,
  sequence length, which 1930s years repeat); six edits made in `docs/proposal.tex`.
- Quercus requirements read (APS360 course page, "Project Proposal" rubric and "project guidelines"):
  LaTeX on the course template; **main text at most 2 pages**, references unlimited; a submission reaching
  3 pages loses 20 %, over 3 pages is not accepted. Rubric: introduction 4, illustration 4 (overall model or
  idea, "core idea and architecture"), background (>= 5 works) 4, data processing 4, architecture 2,
  baseline 2, ethics 2, GitHub link 1, references 1, writing 1. GitHub repo confirmed public.
- LaTeX build on the 2080: TinyTeX lacked environ, lineno, units (nicefrac), trimspaces. `tlmgr` could not
  update itself on Windows; the four packages were taken from the TeX Live archive on CTAN and placed in
  TinyTeX's `texmf-local` (mktexlsr). Builds work.
- With the six edits the main text ran 2 lines onto page 3. Architecture figure redrawn as one row
  (`scripts/make_architecture_figure.py`: embedding, 24 x [RMSNorm, causal attention, +, RMSNorm, SwiGLU, +],
  output; shapes and parameter counts). Preview in the scratchpad (not in `docs/`): figure stacked under the
  overview figure in Figure 1, text trimmed; main text exactly 2 pages, references page 3, also with the
  private author line. Waiting for the user's approval before writing it into `docs/`.
- User approved the preview. `docs/proposal.tex` updated, `docs/figures/architecture.{pdf,png}` written by
  `scripts/make_architecture_figure.py`; `docs/proposal.pdf` (public) and `docs/proposal_submission.pdf`
  (with the private author line, gitignored) rebuilt: 3 pages each, main text ends on page 2. Submission zip
  (PDF + LaTeX source) sent to the user, who uploads it to Quercus by 2026-10-16 23:59.
- Rubric re-check on the user's request: all ten items covered. One factual fix made: the science sources
  in Data Processing now name what is actually used (JSTOR Early Journal Content; Internet Archive scans of
  Annalen der Physik, Scientific American, Nature, PNAS, the 1911 Britannica; JFM via zbMATH Open) instead
  of PNAS via PubMed Central, Britannica via Wikisource and Goettingen GDZ, none of which the corpus uses.
  Rebuilt: 3 pages, main text ends on page 2; submission PDF sent to the user.

## 2026-10-19 — throughput and n-gram measurements for the baseline plan (5080, scratch scripts)
- Scratch scripts in `~/scratch` on the 5080 (not project code): random token ids, fwd + bwd + AdamW, bf16
  autocast, chunked cross-entropy with checkpointing (logits never stored whole), context 4096.
- Transformer 349.2M (RMSNorm, RoPE, SwiGLU, SDPA causal, tied embeddings, torch.compile): 27.5k tok/s at
  1 x 4096 (peak 8.9 GB) -> ~121 h for 12B; 30.3k tok/s at 2 x 4096 (peak 13.0 GB, too close to the card's
  ~14.3 GB usable under Windows).
- Recurrent baselines, 2 layers, tied embeddings, cuDNN: RNN h960 (50.9M), LSTM h800 (49.6M), GRU h832
  (49.2M). With an eager chunked loss at 16 x 4096: ~83k tok/s. At 64 x 4096 the eager version needed
  13.2 GB and fell to 9k tok/s (Windows sysmem fallback). With the chunk loss compiled: 64 x 4096 =
  2^18 tokens per step, 7-11 GB, ~190k tok/s for all three -> 17.6 h each for 12B, 3.7 h for 2.5B. The
  49k-way output layer, not the recurrence, dominates.
- Distinct n-grams on a 300M-token sample of the main mixture's unique documents (EN + DE): at 300M,
  2-gram 23.9M, 3-gram 112M, 4-gram 198M, 5-gram 249M; growth exponents 0.62 / 0.81 / 0.91 / 0.95.
  Extrapolated: 1B tokens -> ~1.7B n-grams (orders 2-5); all 9.7B unique tokens -> ~13.6B. A KenLM
  5-gram with default estimation on the full stream does not fit the 40 GB WSL limit; ~1B tokens does
  (trie ~18 GB, temporary ARPA ~80 GB).
- Host disk: Windows C: 552 GB free; WSL ext4 200 GB free. KenLM, cmake and g++ are not installed on the box.

## 2026-10-19 — baselines branch: code, shards, KenLM, resume tests
- Branch `model/baselines`: `src/data/stream.py` (order, packing, shards, Val slices), `src/model/{loss,recurrent,
  transformer,build,ngram}.py`, `src/train/train.py`, `src/eval/bpb_by_year.py`, configs `stream.yaml`,
  `eval.yaml`, scripts `baselines_run.sh`, `ngram_run.sh`, `resume_test.sh`; tests for stream, models, training
  (resume exactness on CPU) and per-year bpb (21 pass on the 5080).
- Stream dry runs on the box: the first version kept 25M article ids as Python strings and reached 33 GB in WSL
  (5 GB available) -> stopped, rewritten with 64-bit id hashes and numeric columns (peak ~18 GB). One duplicate id
  among tokenized train documents (`dingler_mi311is12`, two different Dingler articles) -> rows matched on id and
  token count.
- Shards written (46 GB): main 23,715,840 document copies, 11,999,913,510 tokens, 2,929,666 rows of 4,096, halves
  6.0007B / 5.9992B, two copies of a parent at least 3.1M tokens apart (1 % 851M, median 6.0B); twin 31,131,396
  copies, 2,929,664 rows (exactly 45,776 steps x 64 rows); Val slices: English (American Stories) 34,817 documents
  8.0M tokens, German 686 pages 4.0M tokens.
- Per-year bpb sets built: English (American Stories, holdout) 222,843 documents 51.9M tokens 1900-1955; German
  7,328 pages 41.0M tokens 1900-1945.
- KenLM built from source in WSL (apt: build-essential, cmake, boost, compression libs). Boost 1.90 has no
  `boost_system` CMake component -> removed from KenLM's CMakeLists; Python module installed from the same source.
- Resume test LSTM (real model and stream, checkpoint every 100, SIGKILL at step 150, resume to 350): losses of the
  resumed steps 110-150 identical to the killed run's; Val after 350 steps (92M tokens): English 2.890 bpb.
  The first attempt found two bugs: evaluation outside autocast multiplied fp16 hidden states by fp32 weights, and
  autocast runs cuDNN RNNs in float16 even when bfloat16 is asked for (no loss scaling here). Fixed: the recurrence
  runs in bf16 on bf16 copies of the fp32 weights; evaluation casts to the weight dtype. Speed unchanged (~189k tok/s).
- Resume test Transformer s335m (checkpoint every 30, SIGKILL at step 50, resume to 110): steps 40 and 50 agree to
  a relative 1.1e-5; steady 29.3k tok/s in the real loop (-> ~114 h for 12B); Val after 110 steps: English 2.796 bpb.
  Box git: `core.fileMode false` (chmod +x on scripts had marked the tree dirty in `git_sha.txt`).
- Teaching prompt for the lecture artifact (lectures 3-6: stream and shards, baselines, training loop, evaluation;
  updates to lectures 0-2) given to the user before training; branch pushed to GitHub at 7ede386.
- Training started (WMI launch): `scripts/baselines_run.sh` (LR probes, then LSTM, GRU, RNN on 12B tokens, each
  followed by per-year bpb) and `scripts/ngram_run.sh` (KenLM on the first 1B tokens, CPU). Hourly watcher
  (`~/scratch/watch_train.sh 3600`) wakes me on the hour or on trouble.
- Resume-test checkpoints on the 5080 deleted (9.1 GB; user OK). The KenLM job was restarted with the ARPA
  output gzipped on the fly (`src/model/ngram.py`): uncompressed ARPA for ~1.7B n-grams is ~70 GB and the box has
  ~150 GB free beside the shards. Smoke test on 2M tokens: binary loads and scores. Relaunched; watcher restarted.
- KenLM 5-gram finished in 30 min (1B tokens, 1,989,211 documents): 2- to 5-grams 47.8M / 285M / 571M / 765M
  (1.67B, matching the extrapolated 1.72B); trie binary 16.8 GB; gzipped ARPA 18.0 GB (deleted). Per-year English
  bpb: 1.99 (1900), 1.75 (1935, lowest), 1.79 (1939), 1.88 (1940), 2.03 (1945), 2.08 (1955) -> `reports/ngram_kn5/`.
- LSTM probes: lr 5e-4 Val English 2.648 bpb; lr 1e-3 stuck near loss 7.65 from step 100 (2.893); lr 2e-3 the same
  by step 160. Queue stopped at that unfinished probe, grid extension deployed, queue relaunched (finished probes
  skipped, the interrupted one restarted).
- 5080 cleanup (user: delete what is confirmed unneeded): uv and pip caches (8.4 GB), `conda clean --all`, old
  scratch files in /tmp from earlier sessions, checkpoints of the two finished LSTM probes (2.4 GB; the choice reads
  only the logs), the ingested data of the excluded HMD source (6.1 GB). WSL free 136 -> 146 GB.
  Disk picture: Windows C: (the 2 TB drive, 1.86 TB usable) 1,473 GB used / 389 GB free; the WSL disk file
  `ext4.vhdx` is 888 GB and does not shrink when files are deleted inside WSL (compaction needs `wsl --shutdown`).
  Kept, pending the user: pipeline intermediates only needed to rerun cleaning (selected 49 GB, dedup 34 GB,
  para_dedup 43 GB, English ingested 115 GB, German ingested 50 GB); raw downloads stay (user rule).
- Probes (Val English bpb after 200M tokens): LSTM 5e-4 2.648 / 1e-3 2.893 (stuck at the unigram plateau all
  through) / 2e-3 2.557 (stuck until ~step 500, then fell); GRU 5e-4 2.401 / 1e-3 2.291 / 2e-3 2.171; RNN 5e-4
  2.454 / 1e-3 2.859 / 2e-3 running. GRU's best is the top of the grid; not extended upwards: a 762-step probe
  favours large rates, and 2e-3 is already high for 45,776 steps. LSTM -> 2e-3, GRU -> 2e-3.
- RNN probes: 2e-3 2.878; the grid extended downwards as planned: 2.5e-4 2.449 (best), 1.25e-4 2.619 (stop).
  Choices: LSTM 2e-3, GRU 2e-3, RNN 2.5e-4. Full LSTM run started 12:29 (box time), ~188k tok/s, ~17.7 h.
- 2026-10-19 23:35: the user's 5-hour usage runs out until 02:30. Monitoring handed to a scheduled task in the Claude
  app (`monitor-5080-baselines`, hourly; runs before 02:30 exit at once; relaunches the queue if it dies; never
  deletes data, changes settings or starts the main run). This session's own watcher stopped. LSTM at step ~24,500
  (Val English 1.571 / German 2.053 at step 24,000).
- 2026-10-19 23:45: user prefers a cloud agent (own credit) for monitoring: the scheduled task was deleted (prompt file
  removed too) and a monitoring-only prompt was written for the cloud agent (watch, report, kill hung project
  processes; no relaunch, no deletion, no settings changes).
- 2026-10-20 07:38: LSTM baseline finished (45,776 steps, 12.0B tokens, lr 2e-3, 19 h 9 min incl. evals): final Val
  English 1.5446 bpb, German 2.0206. Per-year bpb written 07:46 (`reports/lstm_50m/`): English 1900 1.646, 1920 1.589,
  1935 1.459 (lowest), 1939 1.489, 1940 1.542, 1945 1.678, 1955 1.706 -- ~0.3 bpb below the 5-gram everywhere, same
  shape (lowest mid-1930s, rising after 1939). German 1935 1.735, 1939 1.829, 1940 1.606 (the German 1940s sample is
  small and from few titles; to check). GRU started 07:47 (lr 2e-3, ~196k tok/s). Monitoring every 2 h (user).
- 2026-10-20 18:10: corpus v1 frozen (`reports/corpus_freeze_2026-10-20.json`: sha256 of the filtered, tokenized,
  mixture, shard, eval-set and tokenizer manifests). Then, on the user's instruction, the WSL cleanup: manifests of the
  deleted stages packed into `reports/manifests_deleted_stages_2026-10-20.tar.xz` (142 files); deleted data/selected
  (49 GB), dedup (34), para_dedup (43), ingested (115), foreign/de/ingested (50), raw (251), foreign/de/raw (69), and
  the older LSTM checkpoint. WSL disk: 813 GB used -> 206 GB (751 GB free). The VHD on Windows stays at 888 GB until
  it is compacted (needs `wsl --shutdown`; after the baselines). Not touched, pending the user: ~/models
  (DeepSeek-Prover-V2-7B, 20 GB) and ~/venvs (cc, quant, vllm; 9 GB), both from 2026-10-15, not this project.
- Windows side, read-only survey for the user: C: 1,450 GB used. Hidden C:\3b41550f... (2 GB of .m3u8/.ts segments,
  written at night) is Tencent Video's P2P cache (`%APPDATA%\Tencent\QQLive\p2plive.ini: video_dir=...`;
  QQLiveService runs as LocalSystem, auto-start). NetEase CloudMusic wrote 92 GB in 2.5 days. Reported to the user.
- 2026-10-20 18:30 (user's Windows housekeeping on the 5080): Tencent Video and MuMu had already been uninstalled by
  the user when I came to stop the QQLive service; I removed the leftovers (Tencent Video program folder 1.6 MB, MuMu's
  stale autostart entry). The 2 GB P2P cache folder was gone. C: free 412 -> 452 GB. ~/models (DeepSeek-Prover) kept:
  it belongs to the user's "5080 连接方式" session (Desktop\Rate, a theorem-prover project, 2026-10-15).
- 2026-10-20 21:33: paused for the user's own GPU use: queue and GRU trainer stopped right after the step-33,000
  checkpoint (Val English 1.538 bpb), watcher stopped; GPU back to 2.6 GB / 0 %. To resume: relaunch
  scripts/baselines_run.sh through start_download.ps1 (LSTM skipped, GRU resumes from step_033000.pt).
- 2026-10-21 00:00: user: from 03:00 check every 30 min and resume training if nobody uses the GPU. Waiter on the
  box (`~/scratch/wait_gpu_idle.sh 03:00 4500`: 2-min sample, free = mean utilisation < 10 % and memory < 4.5 GB;
  each check logged to ~/scratch/wait_gpu.log); it wakes me when the GPU is free, then I relaunch the queue.
- 2026-10-21 00:59: user: the 5080 is free; waiter stopped, queue relaunched (WMI); GRU resumed from step_033000.pt
  at step 33,000 (pause 21:33 -> 00:59, 3 h 26 min). Two-hour watcher restarted.
- 2026-10-21 02:46: the 5080 went offline (Tailscale: offline, last seen 02:47; SSH timeouts); watcher and the
  GRU-sampling waiter failed with SSH 255. GRU was near step 37,500. Local reachability check started (every
  minute, up to 6 h); on return: check boot time, WSL and the queue, resume from the last checkpoint.
- 2026-10-21 04:32: 5080 back. Cause: unexpected power-off at 02:42:18 (EventLog 6008; Kernel-Power 41 with
  BugcheckCode 0, no power-button press); no bugcheck, dump, WHEA or nvlddmkm event; nothing logged 02:20-02:42 ->
  hard hang or power loss under GPU load (~290 W). Booted again 04:01. Same pattern 2026-10-17 07:59 and 22:48 and
  in August (6008 history). The GRU log ended in NUL bytes (unflushed tail): truncated to the last full line (step
  37,360; copy kept in ~/scratch/gru_log_before_fix.jsonl); step_037000.pt and step_036500.pt verified (load, finite,
  optimiser state). Queue relaunched 04:34, GRU resumed at step 37,000; watcher and GRU-sampling waiter restarted.
- 2026-10-21 ~08:00: the 5080 is unreachable again (SSH and `tailscale ping` time out; last healthy check 06:35,
  GRU at step 41,490). Second outage in ~5.5 h. 45-min reconnect task started; on return: same diagnosis, log
  repair, resume.
- 2026-10-21 08:07: the second "outage" was a network drop only (Tailscale relay "sfo"; tailscale ping timed out
  ~07:55-08:06): no reboot since 04:01, no new 6008/41, GRU kept training (step 44,910 at 08:07). Watcher and
  GRU-sampling waiter restarted.
- 2026-10-21 08:39: GRU baseline finished (lr 2e-3; final Val English 1.5257, German 2.0041; LSTM 1.5446 / 2.0206).
  Per-year English: 1900 1.630, 1920 1.572, 1935 1.438 (lowest), 1939 1.469, 1940 1.524, 1945 1.661, 1955 1.691.
  RNN started 08:39 (lr 2.5e-4). Samples for the user's four "Hitler" prompts (greedy + 3 at T=0.8, top-k 40, CPU)
  for LSTM and GRU saved in reports/samples/ (scripts/sample_lm.py); translated and analysed for the user.
- 2026-10-22 02:55: Windows records for the power-offs (user asked): Reliability Monitor lists "not properly shut
  down" at 10-17 07:59, 10-17 22:48, 10-21 02:42 (plus August); every Kernel-Power 41 has BugcheckCode 0, no dump,
  WHEABootErrorCount 0, no power-button press; no Modern Standby transitions (firmware offers S3 only), so
  ConnectedStandbyInProgress=true on 10-18/10-21 only marks a console/display-off state. 6008 time (02:42:18) is
  Windows' last heartbeat; the GRU log ran to 02:47 (WSL clock), GPU 56-61 C and 242-287 W in the last readings.
  10-17 03:58 was a requested restart (ToDesk). Board MSI X870 GAMING PLUS WIFI, Ryzen 7 9800X3D, DDR5-6000.
  Conclusion: abrupt loss of power or a hard freeze with no warning -- power delivery (PSU protection, wall power,
  connectors) or platform stability (memory EXPO), not software. The 5080 has been up since 10-21 04:01.
- 2026-10-22 03:05: the user asked whether a tripped household breaker could explain the power-offs (yes: an
  instant loss of power matches the records exactly), then confirmed the cause on their side and said to use the
  5080 normally. Decision: no GPU power cap, checkpoints stay at every 500 steps for the main run. RNN at step
  43,870/45,776 (188k tok/s, GPU 51 C, 253 W); its per-year evaluation follows automatically.
- 2026-10-22 03:50: RNN baseline finished (lr 2.5e-4; final Val English 1.7107, German 2.2129); per-year bpb done
  04:00, queue `scripts/baselines_run.sh` ended. All four baselines evaluated: English per-year minimum in 1932
  for every model; mean 1940-55 minus mean 1930-38 = +0.19 GRU, +0.19 LSTM, +0.18 RNN, +0.25 KN 5-gram; ranking
  GRU < LSTM < RNN < 5-gram in all 56 English years. `reports/baselines_v1.md` and
  `reports/figures/baselines_v1_bpb_by_year.{png,pdf}` (`scripts/make_baselines_figure.py`) written; RNN samples
  for the four prompts saved. Phase 3 gate passed (both resume tests passed earlier). KN 5-gram Val bpb not
  computed (user said not now).
- 2026-10-22 04:05-04:30: box repo synced to 036f3ce, then 45a589d (`scripts/main_run.sh`, run `main_s335m`).
  Uploads to the 5080 run at ~9 KB/s (Tailscale direct, 250 ms), so an 8 MB bundle stalled; the box first wrote
  its own identical copies of the large files into git (`git hash-object -w`) and received only a 0.46 MB pack of
  the missing objects. VHD compaction (fstrim, `wsl --shutdown`, diskpart compact, sparse, restart) not done yet:
  a PowerShell script started through WMI is denied on the 5080 ("Access is denied"; cmd and wsl.exe started the
  same way run). Nothing was changed (WSL still running); the user decides how to run it. Main run not started.
- 2026-10-22 06:30-06:43: VHD compaction run by the user in an admin PowerShell on the 5080 (commands from me):
  `fstrim -v /` trimmed 800.1 GiB; `wsl --shutdown`; diskpart `attach vdisk readonly` / `compact vdisk` /
  `detach vdisk`. ext4.vhdx 887.8 GiB -> 209.1 GiB (a first, interrupted attempt had taken it to 871.3 GiB);
  C: free 560 GB -> 1,184 GB. `--set-sparse true` did not take effect (file not sparse), so later deletions
  inside WSL need another fstrim + compact. WSL restarted 06:43 through the keepalive task (root fs 207 GB used
  of 1007 GB). Corpus verification against the freeze record and the content-addressed names started.
- 2026-10-22 06:44-06:46: corpus verified after the compaction: the 17 checksums of
  `reports/corpus_freeze_2026-10-20.json` match, and all 53,296 content-addressed files (shards *.bin, tokenized
  and filtered *.parquet) hash to their names. Main run `main_s335m` started 06:46 (`scripts/main_run.sh` via
  WMI, code 45a589d): s335m, lr 3e-4, 1 x 4,096 micro-batches, 45,776 steps. First steps: loss 10.09 (step 10)
  -> 9.07 (step 30), 29.5k tokens/s (~113 h of steps, end ~2026-10-27 with evaluations), GPU 59 C, 286 W,
  13.5 GB. 2-hour heartbeat watcher running.
- 2026-10-24 22:30: main run at step 25,000 of 45,776 (55 %), no crash or resume since 10-22 06:46; Val English
  1.228, German 1.625 (GRU final 1.526 / 2.004). Power-law refit of Val English on steps >= 1,000/3,000/6,000
  predicts 1.197/1.187/1.184 at the last step, before the cosine tail. CPU samples from step_025000 for the four
  "Hitler" prompts plus "Adolf Hitler, the German Chancellor," and "Hitler died in" (greedy + 2 at T 0.8, top-k
  40) in `reports/samples/main_s335m_step025000_prompts.txt`: whole grammatical sentences and topic kept over 60
  tokens (Nazi regime, Ribbentrop, "Chancellor Hitler in Munich", League of Nations); the Q/A column format is
  reproduced; facts are invented; lowercase "hitler" (" hit" + "ler") still goes to sports pages; "Hitler died
  in" yields no 1945 or bunker. Greedy decoding loops on a phrase.
- 2026-10-25 ~13:20-13:29: the 5080 unreachable (SSH and `tailscale ping` time out while Tailscale lists it
  online via relay "sfo"): a network drop, as on 10-21. Back at 13:29; no reboot (WSL up since 10-22 06:42), the
  main run never stopped (step 30,900, Val English 1.2095 / German 1.6013 at step 30,500). Watcher restarted.
- 2026-10-25 ~22:00: the local 2080 PC (this Claude session's host) crashed: bugcheck 0x133 DPC_WATCHDOG_VIOLATION
  (arg1 = 1, cumulative DPC time-out), preceded by a burst of ~140 NVIDIA driver errors (nvlddmkm 14/153) at
  22:07:49; minidump C:\Windows\Minidump\102526-40859-01.dmp; rebooted 22:10. The session and its 2-hour watcher
  ended with it, so the main run went unwatched from 10-25 21:32 to the end. The 2080 also carries an unsigned
  sample indirect-display driver ("IddSampleDriver Device", oem69.inf, a virtual monitor), a common partner in
  such GPU driver time-outs. Reported to the user.
- 2026-10-27 03:25: main run `main_s335m` finished, 45,776 steps, no crash or resume in 4.9 days (one network drop
  10-25). Final Val English 1.1811, German 1.5646 (GRU 1.5257 / 2.0041). Per-year bpb done 03:53. English: mean
  1930-38 1.093, 1940-55 1.305 (+0.212, ratio 1.194; baselines 1.11-1.14), lowest 1932 (1.075), 1939 1.125,
  1947 1.378. German: 1930-38 1.356, 1940-42 1.210.
- 2026-10-28 02:28: the 5080 restarted cleanly, requested from the Start menu by user AN (event 1074, "other,
  unplanned"), after the run had finished; WSL back at 02:29.
- 2026-10-28 19:55: run record and per-year report copied into the repo; `scripts/make_baselines_figure.py
  --with-main` draws the Transformer over the four baselines; final training curves redrawn.
- 2026-10-28 20:00-21:10: cross-lingual check built and run (`src/eval/crosslingual.py`, branch eval/crosslingual;
  CLAUDE.md rule 5 extended by the user for its German contexts). 75 topics with English and German patterns;
  items from the 1930-01 .. 1939-06 held-out split: first build (235 items) inspected and rejected before any
  scoring -- German contexts started at the top of whole newspaper pages and some were OCR debris -- so passages
  now start ~200 characters before the topic hit on long lines, need >= 80 % letters and spaces, and mismatched
  contexts are a one-to-one matching. Frozen at 289 items over 73 topics (probes/CHECKSUMS) before scoring.
  Result v1 (context and target as two documents, <|endoftext|> between): main model German -> English 0.00073
  bpb [-0.00024, 0.00177], English -> English 0.00097 [0.00024, 0.00172]; GRU 0.00088 [-0.00004, 0.00192] and
  0.00120. By the agreed rule: no twin. But the English reference is itself ~0.1 % of the bpb: the models
  barely use the previous document at all (training packs unrelated documents after <|endoftext|>), so v1 has
  almost no power to tell "German not linked" from "previous document ignored". Reported to the user.
- 2026-10-28 21:30: v2 of the cross-lingual check approved by the user. `--version v2` (config
  `crosslingual.joins`): <|endoftext|> context "\n\n" target; v1 outputs keep their names, v2 writes
  `reports/crosslingual/<run>_v2.{json,md}`. Commit ebb6778 sent to the box as an object pack (48 KB), 93 tests
  pass there; checksums of the frozen topics and items verified before scoring; main_s335m and gru_50m scored.
- 2026-10-28 22:30: cross-lingual v2 results (same frozen items, checksums OK; context "\n\n" target).
  main_s335m: German -> English -0.02144 bpb [-0.03028, -0.01275], English -> English 0.02064 [0.00898, 0.03269]
  (all tokens: -0.02126 / 0.02976). gru_50m: German 0.01485 [0.00731, 0.02217], English 0.04331, ratio 0.34.
  The English reference is now ~20x larger than in v1, so v2 has power. By the agreed rule (main model): no twin.
  The main model's German effect is significantly NEGATIVE (53 of 73 topics below 0): an on-topic German
  article makes the English passage less likely than an off-topic one.
- 2026-10-28 22:45: diagnostic `scripts/crosslingual_diag.py` (not a decision input) ->
  `reports/crosslingual/diag_v2_position.json`. Main model, by target position: tokens 0-3 German +0.029
  [-0.010, 0.067] (English +0.159); 4-15 German -0.071 [-0.098, -0.043]; 16-63 -0.014 [-0.021, -0.008].
  Probability mass on German-marked tokens at positions 4-15: 6.8 % after an on-topic German article vs 5.1 %
  after an off-topic one (16-63: 0.99 % vs 0.76 %). Reading: the model recognises that the German article is on
  the same topic, takes the English passage for its continuation and keeps expecting German; the cross-lingual
  link shows up as interference, not as English knowledge gained. The GRU's German mass is the same in both
  conditions (8.6 % vs 8.7 %) and its effect is positive. Decision on the twin put to the user.
- 2026-10-29: the user decided: no twin (the check already gives the answer), no email. CLAUDE.md rule 1,
  HANDOFF §12 and TASKS Phases 5, 6, 8 updated.
