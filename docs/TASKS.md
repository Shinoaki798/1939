# TASKS — ordered, with gates

Work top to bottom. Each task has an exit gate; do not start the next task
until the gate is met and recorded in `reports/` or `progress_report/material/`.
Dates are hard (HANDOFF §4). Rewritten 2026-10-19 for the current plan (no translation stage since
2026-10-05; English-only twin since 2026-10-06; 12B tokens, context 4096; baselines on the full stream).

## Phase 0 — repo bootstrap — done
- [x] Repo, `.gitignore`, `requirements.txt`, `config/paths.yaml`; the 5080 (WSL) reachable over SSH;
      versions recorded per run in `runs/<name>/env.txt`.

## Phase 1 — corpus v1 — done (2026-10-07 … 10-09)
- [x] Ingest, select, MinHash dedup on the selected pool, date splits (2 % holdout per year and language,
      Val 2 % ≤ cutoff, embargo set aside), paragraph dedup, OCR gates, C1 screen (counted, kept).
- [x] `reports/audit_v1.md`; mixture main + twin, 12.00B seen tokens each (`config/mixture.yaml`).
- [x] Tokenizer: byte-level BPE 49,152 (`config/tokenizer.yaml`); every split tokenised and verified.

## Phase 2 — training stream — done (2026-10-19)
- [x] `src/data/stream.py`: parent documents ordered by seeded hash in two halves, `<|endoftext|>` after
      each, rows of 4,096, shards written once in final order; main, twin and Val slices.
- [x] Per-year bpb sets (`src/eval/bpb_by_year.py --build-set`).

## Phase 3 — baselines (2026-10-19 … 10-24)
- [x] `src/model/recurrent.py` (RNN h960, LSTM h800, GRU h832, ~50M, tied), `src/model/ngram.py` (KenLM,
      first 1B tokens), `src/train/train.py` (resumable, checkpoints every 500 steps, GPU temperature every
      100 steps), `src/eval/bpb_by_year.py`; tests.
- [x] Resume test, LSTM (`scripts/resume_test.sh`; `runs/resume_test_lstm_50m/resume_test.json`).
- [x] Resume test, Transformer s335m (`runs/resume_test_s335m/resume_test.json`).
- [x] LR probes {5e-4, 1e-3, 2e-3} x 200M tokens per recurrent model; best by Val English bpb
      (LSTM 2e-3, GRU 2e-3, RNN 2.5e-4 after extending the grid downwards; `runs/probe_*_choice.json`).
- [x] Full runs, 12B tokens each (~17.6 h): LSTM, GRU, RNN (`scripts/baselines_run.sh`); KN 5-gram on CPU
      in parallel (`scripts/ngram_run.sh`).
- [x] Per-year bpb (English scored, German reported) with bootstrap CIs for all four; `reports/baselines_v1.md`
      (done 2026-10-22; gate passed).
  Gate: four baselines evaluated; both resume tests passed. **Do not start Phase 4 before this gate.**

## Phase 4 — the main Transformer run (2026-10-21 … 10-26)
- [x] Corpus and tokenizer frozen 2026-10-20 (`reports/corpus_freeze_2026-10-20.json`).
- [x] Deleted 2026-10-20 (user):  the cleaning intermediates and the training sources' raw downloads
      (keep filtered, tokenized, mixture, shards, tokenizer, eval, lexicon, ChroniclingAmericaQA).
- [ ] `wsl --shutdown`, sparse VHD, compact `ext4.vhdx` (888 GB -> ~210 GB); restart WSL.
- [ ] `src/model/transformer.py` s335m (349,225,984 parameters), lr 3e-4, 1 x 4,096 micro-batches,
      64-step accumulation, 45,776 steps (~121 h measured). Unattended, resumable, hourly checks.
- [ ] Per-year bpb of the main model.
  Gate: run finished; loss and Val curves in `reports/`.

## Phase 5 — evaluation harness (during the main run)
- [ ] OCR-quality covariate regression in `bpb_by_year.py`.
- [ ] `probes/rq1_terms.csv` (~200 terms, controls, ≥5 contexts each) and `src/eval/probes.py` (Δ_c,
      bootstrap over terms, C4 forced choice). Freeze with `probes/CHECKSUMS` before scoring.
- [ ] `probes/rq3_propositions.csv` (≥120, three tiers, US and DE contexts June / late August 1939, three
      wordings, Gallup figures) and `src/eval/foresight.py` (forced choice, Brier per tier). Freeze first.
- [ ] `src/eval/sanity_caqa.py` (pre-1920 ChroniclingAmericaQA subset).
  Gate: the harness runs end to end on a baseline checkpoint in one command.

## Phase 6 — the English-only twin (2026-10-30 … 11-04)
- [ ] Twin run on `data/shards/twin` (5080 ~121 h, or a rented H100 ~1 day: decide by 10-29).

## Phase 7 — first results + progress report (… 2026-11-20)
- [ ] Per-year bpb curves (Transformer + baselines overlaid); RQ1 Δ profile; RQ3 on the main model.
- [ ] Progress report: data complete, baselines done, first Transformer results. Submit by Nov 20.

## Phase 8 — RQ2 + final report (2026-11-21 … 12-08)
- [ ] RQ2: the same English batteries on the twin; paired differences with bootstrap CIs.
- [ ] Qualitative completions; final report. Submit by Dec 8.

## Open
- The old RQ2 detector (`src/tools/corpus_audit`): dropped from the proposal; drop or keep as an extension
  (decide before November).
