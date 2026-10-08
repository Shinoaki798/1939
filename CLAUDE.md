# CLAUDE.md — APS360 project: The Shape of a Knowledge Boundary

Read `docs/HANDOFF.md` fully before doing anything, then `docs/TASKS.md`.
HANDOFF holds the research design, every locked decision with its rationale,
the instructor's constraints and the schedule. This file holds only the rules
that must never be violated and the conventions of the repo.

State as of 2026-10-01; rules 2, 4-7, 10, 11 revised 2026-10-05, rule 1 revised 2026-10-06, rule 6 revised 2026-10-07, rule 7 revised 2026-10-08 (HANDOFF §12). Proposal due 2026-10-16. The person you are working
with is Junlei An (goes by Andrew), an individual student on this project.

## Hard constraints (never violate, never "improve")

1. **Two Transformer runs, one architecture** (user decision 2026-10-06,
   HANDOFF §12): the main model (English + German) and, for RQ2 only, an
   English-only twin identical in size, tokenizer, English tokens and schedule
   with the German slice removed. No other control Transformer, no ablation,
   no second scale, no seed sweep, no instruction-tuning stage. All other
   controls are constructed at evaluation time. Baselines (vanilla RNN, LSTM,
   GRU, n-gram) are not Transformers and may be retrained.
2. **The Transformer is decoder-only with causal self-attention, trained with
   next-token cross-entropy from random initialisation.** No encoder, no
   masked-LM objective, no pretrained weights anywhere in the model. There is
   no pretrained component anywhere in the project (no translator: German is
   used in the original). Target: GPT-2-class.
3. **Cutoff = 1939-06-30. Embargo = 1939-07-01 … 1939-08-31.** Embargo text
   is never in training and never in any scored evaluation set. One explicit
   exception: embargo-window articles may be used as *conditioning context
   only* in the RQ3 proposition scorer. Nothing dated ≥ 1939-09-01 is ever in
   training.
4. **Splits are by publication date. Never shuffle the pooled corpus.**
   MinHash near-dedup runs on the SELECTED pool (both languages), then the split:
   from every year 1900–1955, per language, a fixed 2 % of the deduplicated
   articles is held out *before* OCR filtering/tokenisation and never trained
   on. Val is a disjoint 2 % ≤ cutoff. The embargo applies to both languages.
5. **Every scored evaluation set is native English from American Stories**
   (per-year bpb, RQ1 probe contexts, RQ2 calibration corpora, RQ3 scored
   continuations, ChroniclingAmericaQA sanity). No translated text exists in
   this project. German text enters training, the German per-year bpb curve
   (reported, not a success criterion) and RQ3 conditioning contexts only.
   Other training-only sources (books, Congressional Record, legal text,
   page-level Chronicling America OCR) are never scored.
6. **C1 coinages are screened, counted and kept** (revised 2026-10-07, user):
   documents dated before the cutoff that contain a C1 term (EN list from the
   probe set; DE list in `probes/c1_screen.csv` — *Blitzkrieg* is attested in
   German before 1939 and is NOT a C1 term) are genuine in-window text
   (surnames, OCR noise, Popeye's Jeep) and stay in training. Every hit is
   counted per term x year x split and listed in the audit for a misdating
   check; the training frequency of each C1 term is reported with RQ1. Never
   delete just the word.
7. **≤ 2 epochs, and only 1934-01-01…1939-06-30 text is repeated** (revised
   2026-10-08, user; HANDOFF §12). Budget 12B seen tokens, model fixed at 350M.
   Targets: 1930s 7.37B (all once + 1934–39.06 again), 1920s 2.6B, pre-1920
   0.8B (10B fallback: 1920s 1.4B, pre-1920 0.35B), science 1.15B; within a
   period the sampling order is by weight ∝ exp(-(1939-year)/5). Caps per
   period: German ≤ 25 %, books ≤ 12 %, legal ≤ 10 % of the period's English
   (1930s case law ≤ 0.35B). If the cleaned pool is short the run is shorter —
   never a third pass, never more pre-1920 text. The **science bucket**
   (`bucket: science` in `config/sources.yaml`) sits outside the period caps and
   recency weighting, any year, ≤ 10 % of each language's seen tokens (EN
   0.85B, DE 0.30B; PNAS, Nature, JSTOR STEM, Annalen der Physik, JFM first);
   ≤ 2 epochs still applies.
8. **Report bits-per-byte, never per-token perplexity, for any cross-model
   number.** Every reported number carries a bootstrap 95 % CI over items.
9. **Probe set and proposition set are frozen (checksummed) before the model
   is evaluated on them.** If a change is genuinely required after that,
   version it and report both.
10. **No pretrained tokenizer** (GPT-2, Llama, etc.) anywhere. The BPE (48k) is
    trained on native pre-cutoff text in the training languages (EN + DE) only;
    never on translated text.
11. **N/A since 2026-10-05: there is no translation stage.** The English
    corpus is still frozen first (10-10) and is sufficient for every graded result.

## Reproducibility conventions

- Every run writes `runs/<name>/{config.yaml, data_manifest.json, seed.txt,
  git_sha.txt, env.txt}`. A number that cannot be traced to a manifest is not
  reported.
- Checkpoint every 500 steps with optimizer state; all long runs are resumable;
  a resume test must pass before any run longer than 8 h is launched.
- Data files are content-addressed: `data/<stage>/<sha256[:12]>.parquet`, with
  a `MANIFEST.json` per stage listing sources, date ranges, counts, filters.
- Translator model name + revision hash go in the manifest of every
  translated shard.
- OCR quality score is a column on every article and is reported by year and
  language in `reports/`.
- Dates are ISO (`1939-06-30`). Years in filenames are four digits.

## Repo layout

```
CLAUDE.md
README.md
requirements.txt
.gitignore
docs/
  HANDOFF.md            full context — read first
  TASKS.md              ordered task list with gates
  proposal.pdf / .tex / references.bib / figures/
config/
  paths.yaml            all paths; never hard-code paths
  model_ladder.yaml     size-by-data rule
  train_defaults.yaml   optimiser / schedule defaults
probes/
  README.md             probe-set construction rules
  rq1_terms.csv         (to build; frozen with CHECKSUMS before eval)
  rq3_propositions.csv  (to build; frozen with CHECKSUMS before eval)
  CHECKSUMS
data/                   gitignored: raw/ dedup/ filtered/ translated/ tokenized/
src/
  data/     ingest, dedup, ocr_quality, splits, translate, tokenizer
  model/    transformer.py (one architecture: main run + twin), rnn.py, lstm.py, gru.py, ngram.py
  train/    train.py, resume.py
  eval/     bpb_by_year.py, probes.py (RQ1), detector.py (RQ2), foresight.py (RQ3), sanity_caqa.py
  tools/    corpus_audit CLI (the RQ2 deliverable)
scripts/                remote-box helpers (remote_pull.sh)
tests/                  unit tests for every src/data stage and the model forward pass
runs/                   per-run manifests (checkpoints gitignored)
reports/                generated tables/figures
```

## Working style

- Record every decision and every operation (what was run, where, outcome,
  problems and fixes) in `progress_report/material/` as it happens (user,
  2026-10-08); see its README for the four files.
- One task per branch, named `data/<thing>`, `model/<thing>`, `eval/<thing>`.
- Before writing code for a stage, print the plan and the exit gate for that
  stage from `docs/TASKS.md`, then implement.
- Prefer small, testable modules. Every `src/data` stage has a `--dry-run`
  that reports counts without writing.
- Dependencies are limited to `requirements.txt`. Do not add a dependency
  without saying why.
- Never hard-code paths; read from `config/paths.yaml`.
- Reference implementations (nanoGPT, nanochat) may be read and cited. Do not
  fork or copy them. The attention, training loop and evaluation harness are
  written here. This is a course requirement (plagiarism is checked against
  public projects).
- When uncertain about a design point, check `docs/HANDOFF.md` §Decisions
  first; if it is not there, stop and ask — do not guess.
- The user communicates with the instructor by email only; never draft
  Quercus posts.

## Hardware

- Training machine: single RTX 5080 (16 GB) on the **remote** box (SSH alias
  `gpu`, WSL2 Ubuntu, code in `/home/an/1939`; see `config/paths.yaml` and
  HANDOFF §12). It also downloads and processes all data. This card has
  previously blacked out under sustained training load. Therefore: checkpoint
  often, keep runs resumable, log GPU temperature every 100 steps, and never
  schedule a run longer than 8 h without a resume test having passed.
- Fallback for the main run: one rented H100 (~15 h) via a cloud-run script
  (rsync tokenised shards + config, same train.py, resume from checkpoint).
- No translation machine is needed (German is used in the original).
