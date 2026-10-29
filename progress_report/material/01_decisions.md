# Decisions

Who: "user" = Andrew (often after a reviewer's DECISION block), "instructor" = Prof. Justin Beland
(email). Full wording and rationale: `docs/HANDOFF.md` §12; hard rules: `CLAUDE.md`.

## Design (2026-10-01 … 10-06)

| date | decision | who | why |
|---|---|---|---|
| 10-01 | Data work is capped (~6 h; no marks for data collection/cleaning); baselines RNN → LSTM → GRU → Transformer; proposal ≤ 3 pages; contact by email only | instructor | course constraints |
| 10-01 | One decoder-only Transformer trained from scratch on text ending 1939-06-30; embargo 1939-07-01 … 08-31; tests from 1939-09-01; bits-per-byte with bootstrap CIs | user | the project measures sequence probabilities of unseen text; a causal LM gives them directly |
| 10-05 | The RTX 5080 (remote, WSL2) downloads, processes and trains; the local RTX 2080 SUPER PC is the fallback downloader | user | one training box; the local PC has an unmetered network |
| 10-05 | Collect as much pre-cutoff text as possible, always deduplicated and cleaned; extra corpora only with explicit approval and training-only | user | American Stories alone (~28B words pre-cutoff) already exceeds the compute budget; breadth over a single source |
| 10-05 | Near-dedup before the per-year 2 % holdout | user | a reprint must not sit on both sides of a split |
| 10-05 | Extra corpora approved: Congressional Record, LoC PD books, pre-1929 books, Federal Register 1936-39, Caselaw Access Project, Chronicling America pages; German DDB, Europeana, Völkischer Beobachter 1925/1930; DTA, ECCO/Evans-TCP, SCOWL as OCR lexicon anchors only; HMD, Trove, Papers Past, ANNO, NCSE out | user | licence, access and period fit |
| 10-05 | German is used in the original (no machine translation anywhere; French dropped); one bilingual 48k BPE trained on EN + DE pre-cutoff text | user | a translator would bring 2020s English into the corpus |
| 10-05 | Model 24 layers, d 1024, 16 heads, ~350M parameters; mixture rules (recent periods first, German ≤ 25 %, books ≤ 12 %, legal ≤ 10 %) | user | compute-bound on one 5080 before the 11-04 midterm |
| 10-05/06 | OCR gates: lexicon hit rate ≥ 0.75 and word-like share ≥ 0.70, both languages; token floor 50 for pages, 20 for article/speech/case-level sources; Völkischer Beobachter cleaned per segment; lexicon v2 (anchor + filtered pool) | user | thresholds chosen from the histograms; at 50 tokens American Stories lost 31 % of its articles |
| 10-06 | Science bucket: outside the period caps and recency weighting, ≤ 10 % of each language, ≤ 2 epochs; US journals 1931-39 in as public domain by non-renewal; German journals 1931-38 in for non-commercial research; period human translations count as period text | user | RQ3 needs scientific context (fission); licence basis recorded per source |
| 10-06 | Two Transformer runs, one architecture: main (EN + DE) and an English-only twin for RQ2 ("what reading the German press changes"); proposal rewritten; RQ3 compares against 1939 Gallup figures | user | turns German from a data-scale device into a measurable variable |
| 10-06 | Downloads move to the 2080 PC; the 5080's VPN is left alone | user | the 5080's VPN is metered (~50 GB left) and was down; the PC is unmetered and faster for archive.org |

## Cleaning (2026-10-07)

| date | decision | who | why |
|---|---|---|---|
| 10-07 | JFM (zbMATH) harvest stopped at 99.79 % (222,808 / 223,270 records) | user | the last 462 records are not worth more walks |
| 10-07 | Pre-1920 general text subsampled before dedup by recency weight (≈ 4x what the cap can use, ≥ 20M words per year and language); nothing before 1900 in the general pool | user | pre-1920 text is capped at a few % of training; deduplicating 44B words for it wastes days |
| 10-07 | German historical typography normalised at selection (long s → s, a/o/u + combining e → ä/ö/ü) | user | one spelling for the tokenizer |
| 10-07 | A near-duplicate cluster keeps its earliest document | user | the original, not the reprint; also keeps post-cutoff reprints of pre-cutoff text out of the tests |
| 10-07 | MinHash thresholds: OCR sources 0.60, keyed sources 0.80 (both rates reported) | user | 2080 experiment: OCR noise hides true duplicates at 0.80 |
| 10-07 | Paragraph-level dedup of page-level sources: (2a, required) remove any training paragraph that matches an American Stories held-out article; (2b) remove wire-service reprints within the pool, keep earliest | user | whole-page MinHash cannot see a reprinted article inside a page; 2a protects the scored sets |
| 10-07 | C1 coinages (radar, jeep, …) found before the cutoff are counted and kept, not dropped (rule 6 revised) | user | the hits are genuine in-window text (Popeye's Jeep, surnames, the balloonist Nadar) |
| 10-07 | German-language articles of US papers stay in the German pool; the twin drops them with the German slice | user | they are German text |

## Mixture and text (2026-10-08)

| date | decision | who | why |
|---|---|---|---|
| 10-08 | Budget 12B seen tokens (was 10B; 350M model ≈ 117 h on the 5080, ≈ 18 h on an H100) | user | the cleaned pool is larger than planned |
| 10-08 | Second epoch only for 1934-01-01 … 1939-06-30; within a period sampling order ∝ exp(-(1939 - year)/5) | user | the 1930s twice left almost nothing for the 1920s and earlier |
| 10-08 | Targets: 1930s 7.37B, 1920s 2.6B, pre-1920 0.8B, science 1.15B (EN 0.85B, DE 0.30B; PNAS, Nature, JSTOR STEM, Annalen der Physik, JFM first); legal ≤ 10 % of each period's English (1930s case law ≤ 0.35B) | user | case law was 17 % of 1930s English |
| 10-08 | The 0.89B left by the German and legal caps goes to the 1920s (2.6B → 3.45B) | user | keeps the 12B budget; the 1930s cannot take more under the caps |
| 10-08 | Text normalisation of training-only sources when shards are written: Europeana repeated words, DDB/RSC spaces before punctuation, German ⸗ line-end hyphens, Congressional Record commas; American Stories untouched | user | extraction artefacts found in the quality read |
| 10-08 | All token numbers are re-derived once the real tokenizer exists (1.35 / 1.6 tokens per word are placeholders) | user | estimates only |
| 10-08 | This folder: every decision and operation is recorded here | user | material for the progress report |

## Operations (machine policy)

| date | decision | who |
|---|---|---|
| 10-06 | Kill hung project processes on the 5080 on every visit (stray loops kept it hot) | user |
| 10-08 | The 5080 is dedicated to the project: Steam and other background apps closed, Wallpaper Engine and ToDesk kept; WSL memory 48 GB + 16 GB swap | user |
| 10-08 | Hardware monitoring allowed (PawnIO + LibreHardwareMonitor library); unused installers removed | user |
| 10-08 | Long unattended runs: hourly check-ins with progress, memory, temperatures and disk writes | user |

## Evaluation data (2026-10-08)

| date | decision | who | why |
|---|---|---|---|
| 10-08 | Missing evaluation data is downloaded on the 5080 through the VPN (back on); ChroniclingAmericaQA dev + test only | user | success criterion 5 needs it; the train split is not used (no fine-tuning) |
| 10-08 | ChroniclingAmericaQA train.json (1.38 GB) downloaded as fine-tuning material; a fine-tuning stage itself still conflicts with CLAUDE.md rule 1 / HANDOFF §6.6 (no tuning stage) and awaits the user's rule change | user | "fine-tuning will be needed in the end" |
| 10-08 | Tokenizer: compare 32k / 48k / 64k byte-level BPE before freezing | user | 48k was an assumption; measure compression and OCR-junk share |
| 10-08 | Fine-tuning is post-project exploration only: a copy of the finished main model, period-safe data, no reported number from it; rule 1 / §6.6 keep "no tuning stage" for the project | user | the research measures the base model's probabilities; tuning would shift them |
| 10-08 | Tokenizer: 48k (49,152) byte-level BPE, frozen in `config/tokenizer.yaml` (sha256 a5f5186e…) | user | 32k→48k gains 2.7 % (EN) / 4.6 % (DE) compression; 64k adds only 1.5 % / 2.9 % for 17M more parameters |
| 10-08 | Probe-token check limited to C1/C2 terms; C3 sense-shift words (blitz, occupation, resistance, collaboration) and C4 proper nouns may be single tokens | user | they are ordinary pre-1939 words in a tokenizer trained only on pre-1939 text |
| 10-09 | 5080 cleanup: delete only the excluded NCSE raw; keep rerun scratch (year shards, shingle arrays, signature caches) and all raw downloads | user | anything a later rerun may need stays; re-downloading DDB/Europeana through the metered VPN is impractical |
| 10-09 | English-only twin is compute-matched: same 12B tokens, steps, schedule, tokenizer; each period's German replaced by English of that period (1930s: second epoch of 1930-33 English; 1920s / pre-1920: extra English by the same recency weights; no third epoch); RQ2 compared on English | user (pasted DECISION) | isolates "reading German" from "seeing fewer tokens"; period shares identical |
| 10-09 | 5080: WSLg disabled (`guiApplications=false`), hibernation off | user | weston crash loop wrote ~6 GB/h of dumps; the remote box never hibernates |
| 10-09 | Twin 1930s shortfall: first the unused 1930s case law (once, legal ≤ 10 % of the twin's 1930s English), the rest from the 1920s | user | keeps the twin compute-matched and its 1930s share close to the main run's |
| 10-09 | Twin 1930s fill order: (1) second pass of the 0.266B 1934-39 English the main run saw once, (2) second epoch of 1930-33 English, (3) case law only if still short, legal ≤ 10 % of 1930s English; period shares identical; the extra 1934-39 second pass is a recorded known difference; main and twin tables side by side in the audit | user (DECISIONS) | the main draw stops its 1930s second epochs at the 7.37B target |
| 10-14 | Context length 4096 (was 2048) for the main run and the twin | user | 53 % of seen tokens are in documents longer than 2048 tokens; the coherent long texts (books, journals, case law, ~11 %) fit whole at 4096; ~+15 % compute |
| 10-14 | Proposal: architecture figure (one row, every layer with shape and parameter count) stacked under the overview figure as Figure 1; text trimmed to keep the main text within 2 pages; plan line set to the completion plan (corpus frozen 20 Oct, main 22-31 Oct, twin and baselines 1-12 Nov) | user | Quercus limit is 2 pages of main text; the rubric's illustration asks for idea and architecture |
| 10-19 | Training stream: unit = parent document (book / case-law chunks adjacent, in chunk order); hash(seed, parent, copy) decides position; two halves: a twice-seen parent has one copy in each half, a once-seen parent goes to a half by hash; within a half sorted by hash; `<|endoftext|>` after every document; cut into 4096-token rows, no padding, no intra-document mask; shards written once in final order and read sequentially (resume = step); the twin uses the same hash, so shared English documents keep their relative order; the recurrent baselines read the main stream | user (confirmed my proposal) | neutral, reproducible order; no recency or repetition bunching at the end; two copies far apart; paired RQ2 comparison with identical order |
| 10-19 | Baselines on the full 12B main stream (as the proposal says): RNN (tanh) h=960 50.9M, GRU h=832 49.2M, LSTM h=800 49.6M; 2 layers, tied embeddings, rows of 4096 with state reset; 2^18 tokens per step; LR from {5e-4, 1e-3, 2e-3} on 200M-token probes, by Val bpb | user | measured 17.6 h each on the 5080 (compiled chunked loss, batch 64 x 4096); width solved from the parameter formula for ~50M and rounded to a multiple of 32/64 |
| 10-19 | Kneser-Ney 5-gram: KenLM default estimation (no pruning) on the first 1B tokens of the main stream; binary trie for querying | user | measured n-gram growth: the full stream would need ~13.6B n-grams; 1B tokens ~1.7B n-grams, ~18 GB, within the 40 GB WSL limit |
| 10-19 | Order: baselines first, then the main Transformer (lr 3e-4, 1 x 4096 micro-batches, 64-step accumulation, 45,776 steps), then the twin. Gate before the main run: unit tests, resume test on the Transformer, four baselines evaluated (Val and per-year bpb with CIs), corpus and tokenizer checksummed | user | TASKS Phase 4 gate kept; measured 121 h for the main run |
| 10-19 | During training I wake myself every hour to check every job (progress, hung or looping processes, GPU temperature, memory, disk); before training starts I give the user a teaching prompt covering the code as written, with the changes for the teaching artifact | user | the user learns the project from its own code |
| 10-19 | LR probe grid for the recurrent baselines extends downwards: when the best of the probed rates is the lowest one, half of it is probed too (at most three times) | me, within the user's "LR from probes" decision | LSTM: 5e-4 -> 2.648 Val bpb after 200M tokens, 1e-3 stuck at the unigram plateau (2.893), 2e-3 the same; the optimum may lie below the grid |
| 10-19 | When the corpus is final (frozen at the launch of the main Transformer run): delete the cleaning intermediates (`data/selected`, `dedup`, `para_dedup`, `ingested`, `foreign/*/ingested`) and the raw downloads of the training sources; keep `filtered`, `tokenized`, `mixture`, `shards`, `tokenizer`, `eval`, `lexicon`, and every evaluation-only download (ChroniclingAmericaQA). Then return the space to Windows: `wsl --shutdown`, sparse VHD on, compact `ext4.vhdx`, restart WSL, launch the main run | user | the corpus is only rebuilt before the main run; ~290 GB of intermediates + ~320 GB of raw; the 888 GB VHD does not shrink by itself |
| 10-22 | 5080 power-offs: no GPU power cap and no hardware change; the main run keeps checkpoints every 500 steps | user | the user checked the cause on their side after the breaker question; Windows records show an instant loss of power, and every resume from a checkpoint so far has been exact |
| 10-22 | Main run keeps checkpoints every 500 steps (~75 min); if it crashes more than twice during training, the next resume switches to every 250 steps (`checkpoint_every_steps` in `runs/main_s335m/config.yaml`) | user | a save takes ~3 s and 4.2 GB (keep 2); 500 steps of s335m take ~74 min, so a crash loses ~37 min on average |
| 10-22 | Fine-tuning a copy of the finished main model may go in the report, in its own section, before/after on the same held-out items, period-safe data; RQ results stay on the untuned model. The twin is trained only if a cross-lingual check on the main model shows a real English-German link (criterion fixed before running; decide by 10-29); otherwise no twin and the time goes to fine-tuning | user | "the course asks for a genuinely useful model"; a twin is only worth ~5 GPU-days if German can affect English at all |
| 10-22 | Twin criterion: train it if German -> English priming (native German article, then matched vs mismatched native English sentence) has a bootstrap 95 % CI excluding 0 and is >= 20 % of English -> English priming; embedding alignment of a bilingual word list reported as a second signal | user | fixed before the check is run |
| 10-28 | Cross-lingual check v2: same frozen items, topics and criterion, but context and target joined by a paragraph break ("\n\n", token 312) in one document instead of <|endoftext|>; v1 kept and reported next to it (rule 9); v2 decides the twin | user (my proposal) | v1's English -> English reference was ~0.001 bpb: the models reset at <|endoftext|> (training packs unrelated documents after it), so v1 could not tell "German not linked" from "previous document ignored" |
