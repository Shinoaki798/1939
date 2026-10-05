# TASKS — ordered, with gates

Work top to bottom. Each task has an exit gate; do not start the next task
until the gate is met and recorded in `reports/`. Check boxes as you go.
Dates are hard (see HANDOFF §4, §9).

## Phase 0 — repo bootstrap (today)
- [x] `git init`, add `.gitignore`, `requirements.txt`, `config/paths.yaml`
      with real paths filled in by the user.
- [ ] `tests/` runs green on an empty repo (pytest discovers nothing).
- [ ] Verify the 5080 is visible (`torch.cuda.is_available()`), log driver +
      torch versions to `runs/env.txt`.
- [ ] Verify SSH to the translation box; log its GPU to `runs/env_remote.txt`.
  Gate: both machines reachable; versions recorded.

## Phase 1 — corpus v1 (freeze 2026-10-10)
- [ ] `src/data/ingest.py`: stream American Stories from HuggingFace; keep
      `article_id, newspaper, date (ISO), lang, text`. `--dry-run` prints
      article and token counts by year. Target years 1900–1955.
- [ ] `reports/audit_v1.md`: table of articles/tokens/bytes by year; histogram
      of OCR quality by year. **This decides the model size** (HANDOFF §6.1).
- [ ] `src/data/ocr_quality.py`: period-lexicon hit rate per article; store as
      column. Propose a drop threshold from the histogram; user confirms.
- [ ] `src/data/dedup.py`: MinHash near-dedup (datasketch), run on the full
      pool **before** any split. Report removed fraction by year.
- [ ] `src/data/splits.py`: per-year 2 % holdout (1900–1955); Train ≤
      1939-06-30; Val disjoint 2 % ≤ cutoff; Embargo 1939-07-01…08-31 set
      aside; Test-A/B/C defined. Assert: no article in two splits; no article
      ≥ 1939-07-01 in Train or Val. Write `data/splits/MANIFEST.json`.
- [ ] `src/data/tokenizer.py`: train 32k BPE on Train (native English) only.
      Save vocab + merges + sha256. Assert the vocab does not contain any
      whole-word probe term from `probes/rq1_seed.csv` as a single token.
- [ ] Tokenise all splits; write `data/tokenized/MANIFEST.json`.
  Gate: v1 frozen; manifest checksummed; `reports/audit_v1.md` committed;
  model size chosen and written to `config/model_ladder.yaml` → `selected`.

## Phase 1b — RQ3 contexts (parallel, hours)
- [ ] Draft `probes/rq3_propositions.csv` (≈60 rows: id, proposition, true
      continuation, false continuation, resolution source).
- [ ] For each proposition, locate one US, one DE, one FR same-topic article
      from June 1939 and from 1939-08-18…08-31. Translate DE/FR with the
      pinned NMT model on the remote box. Store under `probes/rq3_contexts/`.
  Gate: ≈180 contexts present; translator revision recorded.

## Phase 2 — translation for corpus v2 (remote box, parallel; freeze 2026-10-24)
- [ ] `src/data/translate.py`: NLLB-200 (pinned revision) via ctranslate2 or
      transformers; sentence-split → translate → rejoin; keep source lang and
      original article_id; write translated shards with manifest.
- [ ] Sources in priority: Chronicling America foreign-language titles →
      Gallica → Deutsches Zeitungsportal → (stretch) Chinese. Record bulk
      access status in `reports/sources.md`.
- [ ] Probe filter: drop any translated **sentence** containing a probe term
      or inflection from the frozen RQ1 list (use the seed list until frozen,
      then re-run with the frozen list).
- [ ] Per-year 2 % holdout on translated articles too.
- [ ] `reports/translation.md`: tokens by language/year; residual probe hit
      rate; OOV rate vs 1930s lexicon.
  Gate: v2 frozen on 24 Oct **only if complete**; otherwise the Transformer
  trains on v1 and this phase continues as an extension.

## Phase 3 — evaluation harness (on dummy weights, before any real model)
- [ ] `src/eval/bpb_by_year.py`: bits-per-byte on each per-year held-out set;
      bootstrap CI over documents; OCR-quality covariate regression.
- [ ] `src/eval/probes.py`: Δ_c per class; bootstrap CI over terms.
- [ ] `src/eval/foresight.py`: forced-choice LLR scoring; Brier + log score
      per perspective × window; bootstrap over propositions.
- [ ] `src/eval/detector.py`: features → monotone calibration map; ε̂ with CI;
      passage-dating AUC.
- [ ] `src/eval/sanity_caqa.py`: pre-1920 ChroniclingAmericaQA subset.
- [ ] All of the above run end to end on a randomly initialised model.
  Gate: harness produces every table/figure the final report needs, from
  dummy weights, in one command.

## Phase 4 — baselines on v1 (2026-10-17 … 10-24)
- [ ] `src/model/ngram.py`: KenLM modified Kneser–Ney 5-gram on tokenised
      Train; bpb on Val.
- [ ] `src/model/rnn.py`, `lstm.py`, `gru.py`: 2-layer, ~50 M params, same
      tokenizer, same token budget as planned for the Transformer.
- [ ] `src/train/train.py` shared by all neural models; `--resume`;
      checkpoint every 500 steps; GPU temperature logged every 100 steps.
- [ ] Resume test: kill a run at step ~1000, resume, assert loss continuity.
- [ ] Run eval harness on each baseline.
  Gate: four baselines evaluated; `reports/baselines_v1.md`; resume test
  passed. **Do not start Phase 5 before this gate.**

## Phase 5 — the Transformer run (2026-10-25 … 11-02)
- [ ] `src/model/transformer.py`: decoder-only, causal MHA, RMSNorm pre-norm,
      RoPE, SwiGLU, tied embeddings, context 1024. Unit test: causal mask
      (no future leakage), shapes, parameter count matches ladder.
- [ ] Train on the frozen corpus (v2 if frozen, else v1) at the selected size,
      ≤ 2 epochs. Unattended, resumable.
- [ ] Monitor: loss curve, GPU temp, throughput; alert on NaN or stall.
  Gate: converged before Nov 4. If not converged by Nov 2 → start next rung
  down immediately.

## Phase 6 — first results + progress report (2026-11-05 … 11-20)
- [ ] Finalise and **freeze** `probes/rq1_terms.csv` (≈200 terms, controls,
      ≥5 contexts each); write `probes/CHECKSUMS`.
- [ ] Per-year bpb curve (Transformer + baselines overlaid); RQ1 Δ profile.
- [ ] Retrain baselines on the final corpus if it is v2.
- [ ] Progress report: data complete, baseline done, first Transformer
      result. Submit by Nov 20.

## Phase 7 — RQ2 (2026-11-21 … 12-01)
- [ ] Build calibration corpora ε ∈ {0.1, 0.5, 2, 10, 50} %.
- [ ] Fit calibration map; report min. detectable ε; passage AUC.
- [ ] `src/tools/corpus_audit`: `corpus_audit <dir> --model <ckpt>` → ε̂ ± CI.
  Gate: tool runs on a fresh directory; `reports/rq2.md`.

## Phase 8 — RQ3 + qualitative + final report (2026-12-02 … 12-08)
- [ ] Score propositions; per perspective × window tables.
- [ ] Translation-dialect number (if v2 was used).
- [ ] Qualitative completions (`Hitler died in`, etc.) — harness only.
- [ ] Final report. Submit by Dec 8.
