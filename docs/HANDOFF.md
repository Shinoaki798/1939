# HANDOFF — The Shape of a Knowledge Boundary

APS360 (UofT, Fall 2026) individual course project. Student: Junlei An
(Andrew), 1010864165. Repo: https://github.com/Shinoaki798/1939

This document is the single source of truth for *why* things are the way they
are. `CLAUDE.md` is the short list of inviolable rules; `TASKS.md` is the
ordered work plan. If the three ever disagree, this file wins and the other
two must be fixed.

Written 2026-10-01 after the instructor's feedback. Everything below
supersedes earlier drafts (an 8-page proposal draft with six Transformer runs,
a multi-cutoff "staircase" design, and a plan to audit externally released
vintage models were all abandoned; do not resurrect them).

**New session? Start with §13 (operating guide: read order, machines, how to control the 5080,
VPN budget, tools, pitfalls), then `reports/science_status_2026-10-06.md` (current state).**

---

## 1. The project in one paragraph

Train **one decoder-only Transformer language model** from scratch on
newspaper text that ends strictly before the outbreak of the Second World War
(cutoff 1939-06-30, two-month embargo, test from 1939-09-01), then use that
single model to answer three questions about the *structure* of its knowledge
boundary. The thesis: a cutoff is not a wall but a **membrane with structure**
— some post-cutoff concepts are unreachable, others are half-predictable from
parts the model already has. Deep learning is the instrument: none of the
quantities measured exists in corpus statistics; they exist only in a trained
model's next-token distribution.

## 2. Research questions

- **RQ1 — Permeability.** Which classes of post-1939 concepts can a pre-1939
  model reach? Four classes of probe terms (§6.3). Measure: class-wise
  anachronism gap Δ_c = mean surprisal of probe terms − mean surprisal of
  frequency-matched pre-cutoff controls, in fixed contexts. Hypothesis:
  Δ_C2 (compositional) is smallest; C1 (coinage) and C4 (existing proper
  noun, new association) are largest.
- **RQ2 — Invertibility.** Run the boundary backwards into a working tool:
  (a) a passage-level dater and (b) a corpus-level estimator ε̂ of the
  fraction of post-1939 text. Calibrated on held-out corpora with a known
  mixing fraction ε ∈ {0.1, 0.5, 2, 10, 50} %. Report minimum detectable ε
  (3σ separation from ε = 0) and passage-dating AUC. Deliverable: a CLI
  (`src/tools/corpus_audit`) that consumes a corpus and emits ε̂ with a CI.
- **RQ3 — Foresight by perspective.** How much of 1939–45 was already implied
  by the press of mid-1939, and did it depend on whose press? ~60 binary
  propositions about 1939–45, each with three same-topic conditioning
  contexts (US, DE, FR press) from two windows: June 1939 (in training
  period) and the last two weeks of August 1939 (embargo; conditioning only).
  Scored as forced choice between two contrasting English continuations via
  normalised log-likelihood ratio → Brier and log score per perspective and
  window, against chance and the n-gram floor.

Success criteria (stated in advance, in the proposal):
1. Δ_C2 significantly smallest among Δ_C1..C4.
2. Per-year bits-per-byte curve on held-out articles is a *step* at 1939, not
   a slope (OCR quality regressed out); baselines' curves overlaid.
3. ε̂ monotone in ε; minimum detectable ε reported.
4. At least one RQ3 perspective above the n-gram floor; perspectives
   significantly different.
5. Sanity: model and baselines score sensibly on a pre-1920 subset of
   ChroniclingAmericaQA.

All CIs are bootstrap 95 % over items (terms / documents / propositions).
There are no seeds to average over — one model.

## 3. Instructor constraints (Prof. Justin Beland, email, 2026-10-01)

These are binding.

1. **Communication by email only.** He does not monitor Quercus discussion.
2. **Data work capped.** "If more than 6 hours, focus on the models rather
   than data collection, as points are not awarded for data collection /
   cleaning." → Corpus v1 (American Stories, pre-extracted) must be loadable,
   date-split and deduplicated in well under 6 h. Translation is framed as
   *data scale, not cleaning*, runs on a separate machine, and can never
   block graded work (§5).
3. **Baseline progression RNN → LSTM → GRU → Transformer.** "RNN should at
   least be the baseline … what your primary model is would be determined by
   your studies." → Vanilla RNN, LSTM and GRU language models are all trained
   as baselines; the Transformer's choice as primary model is justified by
   those comparisons.
4. **Transformers are taught late (Nov 30), after the progress report.** He
   allows it, will send last semester's material, and expects a literature
   review first.
5. **"Why only decoder?"** Answer (already sent, keep consistent): everything
   measured is a sequence probability of unseen text; a masked encoder yields
   per-token pseudo-likelihoods that neither form a sequence probability nor
   score a continuation; an encoder–decoder has no source sequence here;
   decoder-only is the Transformer configured as a language model, same role
   as the RNN/LSTM/GRU baselines, and is the GPT-2 configuration.
6. Proposal is deliberately high-level: 2-page main text, unlimited
   references, >3 pages = rejected.

## 4. Course deadlines

| Date | Item | Needs |
|---|---|---|
| 2026-10-16 23:59 | Proposal (5 %) | done — `docs/proposal.pdf` |
| 2026-11-04 | Midterm | the Transformer run must be *finished* before this |
| 2026-11-20 23:59 | Progress report (5 %) | all data collected; baseline model done; **at least one result from training the Transformer** |
| 2026-12-08 23:59 | Final report (10 %) | goals, method, data prep, model design, training process, results, critical analysis, limitations, future work |

Late: −20 % within 24 h; nothing accepted after.

## 5. Data design (locked)

### 5.1 Two-stage corpus
- **Corpus v1** — freeze **2026-10-10**. Native English only: American
  Stories (Dell et al. 2023, HuggingFace), article-level structured text
  extracted from Library of Congress Chronicling America scans. Article-level,
  not page OCR, to avoid column bleed and headline fragments.
- **Corpus v2** — freeze **2026-10-24**. v1 + machine-translated non-English
  pre-cutoff press, in priority: (i) foreign-language titles inside Chronicling
  America (German, Italian, Spanish, Czech immigrant press — same date
  metadata and OCR pipeline); (ii) Gallica (FR) and Deutsches Zeitungsportal
  (DE) for domestic German and French press; (iii) Republican-era Chinese
  newspapers as a stretch, only if a bulk-accessible source exists.
- **The Transformer trains on v2 if v2 is frozen on time, otherwise on v1,
  and the translation pipeline is then reported as an extension.** No graded
  result depends on v2.
- **RQ3's ~180 context articles (60 propositions × 3 countries; two windows)
  are translated first, independently of v2.** Hours, not weeks. RQ3 does not
  depend on v2.
- Chinese stretch note for the report: for China the war began July 1937; a
  1939-09-01 cutoff is a European one. State this whether or not Chinese text
  is obtained.

### 5.2 Splits (by publication date; never shuffle the pool)
1. MinHash near-duplicate removal (wire-service reprints across papers) —
   **before** splitting.
2. From every year 1900–1955, hold out a fixed 2 % of articles → **per-year
   held-out set**, never trained on. Gives the per-year bits-per-byte curve on
   both sides of the boundary.
3. **Train** = remaining articles ≤ 1939-06-30.
4. **Val** = a further disjoint 2 % ≤ 1939-06-30 (early stopping / LR).
5. **Embargo** = 1939-07-01 … 1939-08-31: excluded from training and from all
   scored evaluation; RQ3 conditioning contexts only.
6. **Test-A** = held-out 1930–1938; **Test-B** = 1939-09 … 1945-12;
   **Test-C** = 1946–1955.
7. Translated text in v2 gets the same per-year 2 % holdout so a held-out
   translated slice exists for the dialect measurement.

### 5.3 Cleaning
- Per-article **OCR quality** = dictionary hit rate against a period lexicon.
  Stored as a column, reported by year and language, regressed out before the
  per-year curve is interpreted. Fixed drop threshold; yield reported.
- Reverse audit: string-search post-cutoff probe terms in the training split;
  report residual hit rate as a floor on contamination. Report it honestly.

### 5.4 Tokenizer
32,768-token BPE trained on **native-English pre-cutoff text only**. An
off-the-shelf tokenizer leaks (fitted on *radar*, *Hiroshima*); a tokenizer
trained on translated text would let the translator define 1939 vocabulary.
Translated text is tokenised with this vocabulary and compresses slightly
worse — accepted. Baselines use the same tokenizer. All cross-model numbers
are **bits-per-byte** (total NLL / raw UTF-8 bytes).

### 5.5 Translation pipeline (data scale, not cleaning)
- Fixed-version, sentence-level NMT (NLLB-200 preferred; Marian-OPUS
  fallback). **Not an LLM** — a generative assistant can add hindsight; a
  sentence-level NMT only maps sentences. Model name + revision in every
  shard's manifest.
- Runs as inference on the **separate remote GPU machine** (SSH alias in
  `config/paths.yaml`); never on the 5080 while training.
- Three defences against the translator's 2020s English:
  1. Any translated sentence containing an RQ1 probe term or an inflection is
     dropped **whole**.
  2. Translated text is **never scored**: training data and RQ3 conditioning
     context only.
  3. Measured: residual probe hit rate and OOV-against-1930s-lexicon rate
     reported before training; after training, a held-out translated slice is
     scored against native-English held-out text of the same years → the
     **translation-dialect effect**, reported as a number.
- No fixed cap on the translated share; throughput is the bound. Translated
  tokens count toward the size ladder.
- Risk: German Fraktur OCR is markedly worse than English; apply the same OCR
  threshold per language. If European archives have no bulk access, the
  foreign-language titles inside Chronicling America alone are enough.

## 6. Model and evaluation design (locked)

### 6.1 The one Transformer
Decoder-only Transformer as taught in the course: token embeddings, stacked
blocks of multi-head causal self-attention + position-wise FFN, residual
connections, layer norm, softmax over vocab. Next-token cross-entropy. From
random initialisation. Implementation choices (standard refinements, each to
be justified in the final report): pre-norm RMSNorm, rotary position
embeddings, SwiGLU FFN, no biases, tied embeddings, context 1024, AdamW with
warmup + cosine decay, bf16. Written from scratch in PyTorch.

**Size follows data** (`config/model_ladder.yaml`): ~20 tokens/parameter,
≤ 2 epochs (repetition raises memorisation, which biases exactly the surprisal
measurements the project depends on).

| usable pre-1939 tokens | params | layers | d_model | heads | approx. time on 5080 |
|---|---|---|---|---|---|
| < 1.5 B | 90 M | 12 | 640 | 10 | ~4 h |
| 1.5–3 B | 160 M | 16 | 768 | 12 | ~8 h |
| 3–6 B | 250 M | 20 | 896 | 14 | ~18 h |
| > 6 B | 335 M | 24 | 1024 | 16 | 30–40 h |

Compute estimate 6·N·D; assumed achieved ~1.2e14 FLOP/s bf16 (conservative).
Memory is not a constraint (≈5.5 GB params+grads+AdamW at 335 M vs 16 GB).

GPT-2 reference points for expectation-setting: GPT-2 small = 124 M on ~9 B
tokens; our 335 M at 7 B tokens is an under-trained GPT-2-medium. OCR noise,
not token count, is the main drag on fluency.

### 6.2 Baselines (instructor-mandated progression)
All on the identical corpus, tokenizer and token budget as the Transformer:
- **Vanilla RNN**, **LSTM**, **GRU** language models — each 2-layer, ~50 M
  parameters, same next-token objective. (Not parameter-matched to the
  Transformer: a 335 M 2-layer LSTM would need ~5k hidden units and poor GPU
  utilisation would make three such runs cost more than the main run.)
- **Modified Kneser–Ney 5-gram** (KenLM defaults) — non-neural bits-per-byte
  floor; cannot compose unseen compounds, so it should show no C1/C2 ordering.
- Baselines train first on v1 (2026-10-17 … 10-24) to validate the whole
  pipeline before the single Transformer run is committed; they are retrained
  on the final corpus for every reported comparison.
- The RQ2 detector is also evaluated on the recurrent baselines.

### 6.3 RQ1 probe set (`probes/rq1_terms.csv`)
~200 post-cutoff terms in four classes, each paired with a frequency-matched
pre-cutoff control from the same semantic field, embedded in ≥ 5 natural
sentence contexts drawn from post-cutoff articles. Frozen + checksummed before
evaluation. Class assignment documented per term with rationale.

| class | definition | examples |
|---|---|---|
| C1 coinage | word did not exist before the war | radar (1940), genocide (1944), napalm (1942), kamikaze (1945 in English), jeep (1941), Quisling (1940) |
| C2 compositional | new concept built from existing words | atomic bomb, jet engine, guided missile, blackout curtain |
| C3 existing word, new sense | dominant meaning shifted during the war | blitz, resistance, collaboration, liberation, occupation, D-Day |
| C4 existing proper noun, new association | word present in 1939 text; association is not | Pearl Harbor, Hiroshima, Dunkirk, Normandy, Auschwitz, Vichy, Yalta |

C4 is why the taxonomy exists: a naive vocabulary check finds nothing wrong,
yet this is exactly the contamination a "1939 model" would be accused of.

### 6.4 RQ2 detector
Features: per-token surprisal under the Transformer; fraction of
high-surprisal tokens; class-wise Δ statistics. Calibration layer: a fitted
**monotone map** from these statistics to ε (isotonic or logistic) — explicitly
*not* a second neural network. Calibration corpora: held-out pre-cutoff
articles with ε ∈ {0.1, 0.5, 2, 10, 50} % of 1939-09…1945-12 articles mixed in,
document count fixed. Outputs: ε̂ with bootstrap CI; passage-level dating score
with AUC on held-out articles of known date. Ships with its detection limit
and failure modes (ethics).

### 6.5 RQ3 scorer
Base LMs do not emit calibrated probabilities on request. Each proposition is
posed as a forced choice between two contrasting English continuations after a
conditioning context; the normalised log-likelihood ratio is the model's
probability. Contexts: US / DE / FR, June 1939 and late August 1939 (embargo).
References: chance (0.5), n-gram floor. Report per perspective × window.
Propositions written and frozen before any model is evaluated.

### 6.6 Qualitative demo (harness only; no SFT)
A base model continues text; it does not answer questions. Demo via completion
prompts, e.g. `Adolf Hitler, the German Chancellor,` (should be known — he had
been in the news since the 1920s) and `Hitler died in` (should *not* produce
1945 or a bunker). Optional few-shot Q/A template with pre-1939-only examples.
**No instruction-tuning stage**: modern-written Q/A pairs are a contamination
vector and a second training stage breaks the "one run" discipline.

## 7. Ethics (from the proposal; keep in the final report)
- Pre-1939 newspapers carry period racial/ethnic/gender language; the model
  reproduces it; document, do not sanitise.
- Translated 1930s German press includes state propaganda; NMT errors can
  misattribute words. Report corpus shares by source/period. RQ3 results are
  properties of a textual record, not verdicts on nations or culpability.
- The RQ2 tool can be misused as an accusation (forgery, contamination); ship
  it with its calibrated detection limit and failure modes.
- Research instrument, not for deployment. Public-domain data, attributed.

## 8. Related work (cited in the proposal; do not re-litigate)
Lazaridou et al. 2021 (temporal degradation); Dhingra et al. 2022 (temporal
conditioning); Drinkall et al. 2024 TimeMachineGPT; fixed-cutoff vintage
models incl. 1939 — TimeCapsuleLLM, Ranke-4B/History LLMs, Talkie-1930 — cited
neutrally in one sentence, **no comparison or audit against them** (user
decision); SemEval-2020 Task 1, HistBERT, Hosseini et al. 2021 (encoder-based
semantic change; contrasted with our causal-LM need); American Stories;
ChroniclingAmericaQA; NLLB.

## 9. Schedule

| window | work | gate |
|---|---|---|
| Oct 1–10 | corpus audit (tokens + OCR quality by year), v1 ingest, dedup, per-year holdouts, tokenizer; translation starts on remote box; RQ3 contexts translated | **v1 frozen 10 Oct**; provisional model size chosen |
| Oct 11–16 | eval harness on dummy weights; probe set + propositions drafted; submit proposal | proposal in by 16 Oct 23:59 |
| Oct 17–24 | KN 5-gram, RNN, LSTM, GRU on v1 | pipeline validated end to end; **v2 frozen 24 Oct** |
| Oct 25–Nov 2 | **the Transformer run** (checkpointed, resumable, unattended) | converged before Nov 4 midterm |
| Nov 5–13 | RQ1 profile; per-year curve; baselines retrained on final corpus; probe set finalised | first Transformer result |
| Nov 14–20 | progress report | submitted by Nov 20 |
| Nov 21–Dec 1 | RQ2 calibration corpora; detector; `corpus_audit` CLI | min. detectable ε reported |
| Dec 2–5 | RQ3 scoring; qualitative demo; translation-dialect number | — |
| Dec 6–8 | final report | submitted by Dec 8 |

Fallback: if the Transformer has not converged by Nov 2, start the next rung
down immediately; it still satisfies every success criterion.

## 10. Risks
- Single long run fails late → baselines first validate pipeline; resume test;
  rung-down fallback.
- Thin 1930s coverage (collection is denser pre-1922) → audit is a gate;
  translation is the first lever; design does not depend on token count.
- Residual contamination in "clean" corpus (bad date metadata,
  retrospectives) → embargo + reverse audit; report the floor honestly.
- Probe-set bias (author selects confirming terms) → freeze + checksum
  before eval; per-term rationale; release the set.
- Translation dominates corpus → report fraction; measure dialect effect;
  every scored set is native English; retrain one rung down with reduced
  share if the dialect effect exceeds the RQ1 gaps.
- 5080 blackouts under load → checkpoint/500 steps, temperature logging,
  ≤ 8 h runs without a passed resume test.
- Midterm Nov 4 → run finishes before; the surrounding days hold
  interruptible annotation/eval work.

## 11. Open questions (ask the user; do not decide alone)
- Which machine runs translation, and its GPU (affects NLLB size:
  600 M distilled vs 1.3 B). The 5080 is the training box (§12), so it can
  translate only while not training; the local PC has an RTX 2080 SUPER 8 GB.
- Whether the instructor approves keeping the ~180 translated RQ3 contexts
  (he was asked by email; default: keep).
- Exact OCR-quality drop threshold (set after the week-1 histogram).
- Bulk-access status of Gallica and Deutsches Zeitungsportal.
- (2026-10-06, twin) The old RQ2 — passage dater, corpus estimate ε̂,
  `src/tools/corpus_audit` — is not in the rewritten proposal: dropped, or kept
  as an extension?
- (2026-10-06, twin) Twin token budget: the same English token stream as the
  main run (same documents, order and repetition; the run is shorter by the
  German share, ≤ 25 %), or the same number of steps? The proposal ("identical
  … English tokens and schedule") reads as the former. Decide before the
  tokenised shards are built: the former needs the sampler to produce the
  English stream independently of the German one.
- (2026-10-06, twin) Gallup 1939 figures for RQ3: collected by hand per
  proposition from the Gallup volume (evaluation reference only, never
  training); when.

## 12. Decisions log

Dated entries; each supersedes anything above it that it contradicts.

- **2026-10-05 — machines.** The RTX 5080 is the *remote* box (SSH alias
  `gpu`, Windows 11 + WSL2, in China; `config/paths.yaml`). It downloads,
  processes all data and trains. The local PC (RTX 2080 SUPER 8 GB) is the
  fallback downloader. huggingface.co is reachable from the 5080 only through
  the Windows-side proxy; `src/data/download.py` handles this.
- **2026-10-05 — data volume and policy.** American Stories 1900–1955 is
  182 GB compressed; 1900–1922 is ~6 GB/year, 1923–1955 ~1 GB/year. A 10 MB
  sample of 1938 gave ~167 M words per compressed GB, so the pre-cutoff pool
  is roughly 28 B words before dedup/cleaning — far more than the ~6.7 B
  tokens the s335m rung consumes. Model size is therefore compute-bound, not
  data-bound. User policy: collect as much pre-cutoff text as possible, always
  deduplicated and cleaned; the year mix is not a design variable. Corpora
  beyond American Stories enter only with the user's explicit approval, are
  training-only, and obey every cutoff rule.
- **2026-10-05 — dedup before holdout.** §5.2 is authoritative: MinHash
  near-dedup runs on the full pool first; the per-year 2 % holdout is then
  drawn from the deduplicated pool, before OCR filtering and tokenisation.
  `CLAUDE.md` rule 4 was reworded to match.
- **2026-10-05 — repo.** Git repo at https://github.com/Shinoaki798/1939;
  the remote box pulls via `scripts/remote_pull.sh`. Data stays gitignored.
- **2026-10-05 — model size may exceed the ladder.** User: the parameter count
  may grow beyond s335m; enrich the training content first. The rung is still
  chosen after the audit (≤ 2 epochs, ≈ 20 tokens/param); the binding limit is
  compute — the run must converge before 2026-11-02 on one 5080.
- **2026-10-05 — extra corpora approved (training only).** British Library
  Heritage Made Digital newspapers (`biglam/hmd_newspapers`, UK, 1800–1896) and
  the US Congressional Record (Congresses 43–76, kept ≤ 1939-06-30 by day).
  Pinned in `config/sources.yaml`. Rejected by the user: Old Bailey (too
  narrow). Requested: more academic and everyday-life English text, and
  non-English pre-cutoff text (Chinese, German, Japanese, Italian, …) — each
  candidate goes to the user for approval first.
- **2026-10-05 — open: how non-English text is used.** §5 translates foreign
  text into English (v2) and rule 10 keeps the tokenizer English-only. Training
  on original-language text directly would change rule 10, the tokenizer and
  the proposal's corpus description. Collect first; ask before deciding.
- **2026-10-05 — second round approved.** English: JSTOR Early Journal
  Content, Royal Society Corpus, Evans/ECCO-TCP, NCSE, pre-1929 books, LoC PD
  books. Foreign: German only (DDB newspapers, Europeana German newspapers
  1900–1939, Deutsches Textarchiv), on condition that downloads are text, not
  scans (verified: OCR/hand-keyed text, no page images). Foreign text is stored
  apart from English under `data/foreign/<lang>/`; how it is used stays open.
  The user allows deleting the raw downloads of supplementary corpora after a
  verified ingest (`src/data/prune_raw.py`); American Stories raw is kept.
- **2026-10-05 — language filter for American Stories.** American Stories
  contains foreign-language titles (e.g. Puerto Rican Spanish, Cleveland Czech).
  Article rule + per-title-year rule in `src/data/filters.py`, calibrated on
  1923. Scans without an `lccn` block (~5 %) are kept; LCCN from the filename.
- **2026-10-05 — review decisions (supersede §2 RQ3 "US, DE, FR", §3.5 NMT
  sentence, §5.1, §5.5, §6.1 size table, §9 schedule, §11 translation question).**
  - *German in the original, no NMT anywhere.* French dropped. RQ3 perspectives:
    US (English contexts) vs DE (German contexts), June 1939 and 1939-08-18…08-31;
    scored continuations English. CLAUDE.md rules 2, 4–7, 10, 11 rewritten.
  - *Model:* one Transformer (no control Transformer — user, 2026-10-05), 24
    layers, d 1024, 16 heads, context 2048, bilingual 48k BPE, ≈350M params, up
    to 10B seen tokens; RTX 5080 (~85 h, bf16, compile, SDPA, checkpoint/500,
    resume test first), start 10-22, done by 11-01; fallback one rented H100.
  - *Mixture:* 1930-01-01…1939-06-30 every surviving document twice; 1920s
    fill the remainder (≤35 % of seen tokens); pre-1920 ≤8 %; German ≤25 % and
    books ≤12 % per period; legal/regulatory ≤10 %; half-life 5 y orders
    sampling within a period; recent periods are up-weighted by repetition
    before any older period is added; a short pool means a shorter run.
  - *Sources in:* American Stories; Congressional Record; LoC PD books (first)
    and pre-1929 books (second), years ≥1900, deduped against each other,
    raw kept until ingest verified; Europeana DE + DDB (keep; dedup by title +
    date, then MinHash); Völkischer Beobachter 1925 + 1930 (named source).
    Being verified (English 1930s): Chronicling America pages added after the
    American Stories snapshot, Federal Register 1936–39, Caselaw Access Project.
  - *Sources out:* Trove (no key without approval; NLA bars AI training),
    Papers Past (API text lower-cased, unpunctuated), ANNO (no bulk; bot check),
    HMD, JSTOR EJC, ECCO, Evans, Royal Society, NCSE (archaic / NC licences).
  - *Cleaning:* OCR score = period-lexicon hit rate per language; German
    lexicon from DTA (≤1938) + widespread pool words; threshold chosen by the
    user from the histogram (Europeana `mean_ocr ≥ 0.7` rejected: it removes
    nearly all 1925–39 German). C1 screen drops whole documents in both
    languages. German per-year 2 % holdout spans 1900–1955 (German curve only).
  - *Schedule:* 10-10 English audit + English v1 freeze; 10-17 German freeze;
    10-17…10-21 baselines (RNN/LSTM/GRU 50M bilingual + KN 5-gram); 10-20 corpus
    + tokenizer frozen; 10-22 main run; 11-01 done; 11-04 midterm; 11-20 progress.
  - *Audit (reports/audit_v1.md, 10-10):* tokens by year × language (unique and
    seen shares separately), 1925–1939-06 totals per language, OCR histograms +
    thresholds per language, dedup rates per source, C1 drop counts, mixture vs
    target and any shortfall.
- **2026-10-05 — German OCR threshold.** From `reports/ocr_quality_de.md`
  (lexicon = DTA ≤1938, 1.32M types, + 177k widespread pool types): hit rate
  ≥ **0.75** (user). Words lost before cutoff, German-language pages
  1900–1939 only: DDB 0–4.3 %, Europeana 4.3–8.7 %, VB 0 % (the first table
  also counted Czech/Polish/French pages and showed 16–21 % for Europeana;
  those pages go at the language filter anyway). German supply in every period exceeds its 25 % cap, so the
  stricter threshold costs no seen tokens. The hit rate ignores digits and
  one-letter fragments (number tables, shredded OCR still reach ≥ 0.8), so a
  second gate, `word_share`, is applied too; its threshold is chosen by the
  user from its own histogram before anything is dropped.
- **2026-10-05 — CAP file names.** static.case.law volume numbers repeat across
  reporters; the first run saved them under the bare number and overwrote 120
  volumes. Files now carry `<reporter>__<vol>.zip`, the downloader refuses any
  table whose keys share a local name, and `scripts/cap_fix_names.py`
  re-attributed the files on disk by sha256 (214 kept, 120 refetched). NCSE
  (excluded) had the same clash in 6 file names; its counts are a lower bound.
- **2026-10-06 — OCR gates (final; supersedes the 0.80 / digit-share draft).**
  Gate 1 lexicon hit rate ≥ 0.75; gate 2 word-like share ≥ 0.70 (share of
  non-punctuation tokens that are letter words of length ≥ 2), applied to
  pages passing gate 1; documents < 50 tokens dropped; no separate digit gate.
  Same gates and values for English after its own histogram check (if English
  needs other values, report both and justify). Völkischer Beobachter is
  included and cleaned per segment, not gated whole: its issue text breaks
  into 4-token lines and 8-token blank-line blocks, so blocks are merged into
  segments of ≥ 30 tokens and segments with word share < 0.70 are dropped
  (keeps ~96 % of VB words; the literal line split would keep 4 %). Gate
  parameters, scorer version and lexicon hash go into every filtered shard's
  MANIFEST (`ocr_quality.gate_params`). Drop rates per source × period per gate
  and boundary samples (0.65–0.75 word share) go into `reports/audit_v1.md`.
  Proposal Cleaning step (3) now reads "lexicon hit rate ≥0.75 and word-like
  share ≥0.70, per language".
- **2026-10-06 — OCR lexicon v2, both languages.** Lexicon = anchor ∪ pool
  types that survive a variant filter. Anchor (no OCR errors): de = DTA ≤ 1938,
  types ≥ 2; en = SCOWL 2020.12.07 english + american lists of size ≤ 60 (the
  hunspell en_US source; downloaded on the local PC with the user's approval,
  sha256 5587667c…, synced to the box) ∪ ECCO-TCP + Evans-TCP types ≥ 2. Pool:
  pre-cutoff (1900-01-01…1939-06-30) in-language newspapers, df ≥ 50 in ≥ 5 (de)
  / ≥ 10 (en, American Stories only) titles. Variant filter: a pool type within
  edit distance 1 of a type ≥ 50× more frequent is dropped unless it is in the
  anchor; dropped types are logged, top 50 in `reports/ocr_lexicon_<lang>.md`.
  Gates unchanged (0.75 / 0.70 / 50). German is re-reported on v2; if any
  source × period drop rate moves by more than 5 points, the boundary samples go
  to the user before proceeding. ECCO, Evans, DTA and SCOWL are "lexicon anchor,
  not training" in `config/sources.yaml`, kept, and refused by `prune_raw.py`.
  The lexicon is an OCR instrument only; it never touches the tokenizer.
- **2026-10-06 — English OCR check; token floor by document type.** English
  lexicon v2: anchor 370,774 types (SCOWL ≤ 60 + TCP), pool 24,572 types of
  which 23,557 already in the anchor, 275 variants dropped (lhe, thls, aas, iof,
  …). Hit rate ≥ 0.75 and word share ≥ 0.70 kept unchanged for English (AS
  median hit rate 0.98; the two gates drop < 2 % of words except Federal
  Register 1937–39, 14 %, tables). The 50-token floor stays for page- and
  issue-level documents; article-, speech- and case-level sources (American
  Stories, Congressional Record, CAP, JSTOR EJC, RSC) use 20 tokens, the point
  below which neither score is defined (user). Reason: at 50, AS lost 31 % of
  articles / 6–7 % of words and the Congressional Record 68 % of speeches /
  16–17 % of words; at 20 the losses are 0.6 % and ~5 %.
- **2026-10-06 — science bucket (user task + answers).** New bucket "science",
  exempt from recency weighting and outside the period caps (pre-1920 ≤ 8 %,
  books ≤ 12 %, etc. do not apply to it); capped at 10 % of seen tokens **per
  language**, enforced after dedup + gates; ≤ 2 epochs. Same OCR gates; keyed /
  born-digital sources (JFM, Gutenberg, DTA, Dingler TEI) skip the OCR gates
  but get the C1 screen. Licence of the model release is non-commercial, so
  CC BY-NC sources are in. Every source: `config/sources.yaml` entry with
  title, bucket, licence, access_method, ocr_or_keyed, subject_filter_rule;
  per-file checksums; ingested MANIFEST carries these as `source_meta`.
  Nothing ≥ 1939-07-01 in any source; year-only items ≤ 1938. **JFM** reviews
  are kept by the publication date of the JFM volume (≤ 1939-06-30), not by the
  reviewed paper's year (a review of a 1938 paper may be written 1939–42).
  US public domain now reaches works published in 1930 (entered PD
  2026-01-01), not 1928. Re-included from disk: JSTOR EJC (STEM titles only,
  list in `config/science_jstor_titles.txt`) and Royal Society Corpus 6.0.4
  (all, 17,520 papers, 78.6M words). ECCO/Evans stay lexicon anchors.
  RQ3: fission propositions must draw their contexts from Jan–Jun 1939
  newspapers and PNAS. Audit adds science tokens by language × decade ×
  source, a licence table and the keyed vs OCR share of the bucket.
- **2026-10-06 — science sources: user decisions after the feasibility check**
  (`reports/science_feasibility_2026-10-06.md`). (1) US non-federal journals
  1931–39 (PNAS, Bull. AMS, BSTJ, Physical Review) are IN as US public domain by
  non-renewal (no renewals found, Online Books Page); basis recorded in each
  source's licence field. (2) German journals 1931–38 are IN, including the
  archive.org copies of Naturwissenschaften and Physikalische Zeitschrift that
  carry no licence (still in copyright; used for non-commercial research
  training, never redistributed). (3) Period human translations (NACA Technical
  Memorandums, translated letters in the RSC) count as native period text: in
  training, flagged `period_translation` in meta, never scored. (4) JFM: reviews
  in volumes ≤ 61 only (~153k); volume 62+ only if its publication date is
  shown to be ≤ 1939-06-30. Defaults applied: no US patents (no account-free
  bulk OCR; the Official Gazette would swamp the mix), no Zentralblatt (scans
  only; keyed reviews include modern retro-reviews), Meyers 6th ed. from
  archive.org OCR (zeno.org forbids robots), EB11 vols 2–17 keyed from
  Gutenberg and vols 1, 18–28, 30–31 OCR from archive.org (index vol 29 out),
  Nature to 1930 with dates from the volume field, PMC not used (scans only,
  bulk download prohibited) — PNAS and Public Health Reports come from
  archive.org. EJC science titles: `config/science_jstor_titles.txt` (69
  titles, 166,290 articles, 306M words before cleaning).
- **2026-10-06 — English token floor, CA routing.** See §13 for the operating
  guide. Chronicling America must never be downloaded through the VPN (68 GB
  > budget); it was fetched and ingested on the local PC and shipped as parquet.
- **2026-10-06 — two Transformer runs: main + English-only twin; proposal
  rewritten (user decision).** Supersedes "one Transformer" in §1, §2 RQ2/RQ3,
  §6.4, §6.5, §9 and "no control Transformer" of 2026-10-05.
  - *Runs:* one architecture, two runs. Main = English + German, as fixed on
    2026-10-05 (24 layers, d 1024, 16 heads, bilingual 48k BPE, ≈350M params,
    ≤ 10B seen tokens). Twin = identical size, tokenizer, English tokens and
    schedule, German slice removed; used for RQ2 only. Still no other control
    Transformer, ablation, second scale, seed sweep or instruction tuning.
    Main run 10-22…11-01 as planned; twin in November (~85 h on the 5080 or
    ~15 h on a rented H100). CLAUDE.md rule 1 rewritten.
  - *RQ2 = perspective:* what reading the German press changes. The RQ1 and
    RQ3 batteries run on both models; the paired difference is the effect.
    Success: a significant paired difference on DE-context propositions.
  - *RQ3 = foresight against informed contemporaries:* ≥ 120 binary
    propositions resolved after 1939-09-01, in three tiers by what was knowable
    in June 1939 (near: war in Europe this year; mid: third Roosevelt term, US
    entry, self-sustaining chain reaction; far: Hitler defeated, a fission
    weapon used); each with same-topic US (English) and DE (German) contexts
    from June and 1939-08-18…31, in period language, three wordings each;
    English continuations. Brier per tier × perspective × window against
    chance, the n-gram floor and the 1939 Gallup figure where polled (Gallup,
    *The Gallup Poll: Public Opinion 1935–1971*). Success: mid-tier Brier
    better than chance and not worse than Gallup.
  - *RQ1* adds a C4 forced-choice test (e.g. Pearl Harbor: naval station vs.
    attack) that doubles as a contamination check.
  - *Proposal vs. HANDOFF:* the proposal stays high-level and is not edited for
    details it simplifies (user): PNAS comes from archive.org, not PubMed
    Central; the token floor is 20 for article-, speech- and case-level
    sources, 50 for pages; Math. Annalen and Crelle come from archive.org with
    GDZ as fallback. HANDOFF governs.
  - *E-mail:* kept out of the public repo again (`\authoremail` +
    gitignored `docs/author_private.tex`); `docs/proposal.pdf` is the public
    build, `docs/proposal_submission.pdf` (gitignored) the one to submit.

---

## 13. Operating guide for a new session (written 2026-10-06)

Read this section first if you are a new Claude session taking over. The user
(Andrew) chats in Chinese, wants short, concrete reports, decides policy (often
after consulting a separate reviewer agent whose DECISION blocks he pastes),
and expects you to do the work, not to ask about things the code or this file
already answers.

### 13.1 Read, in this order

| # | file | why |
|---|---|---|
| 1 | `CLAUDE.md` | hard rules (main + English-only twin Transformer, cutoff/embargo, splits, ≤ 2 epochs + caps + science bucket, bpb, no pretrained tokenizer) |
| 2 | `docs/HANDOFF.md` §12 (from 2026-10-05) and this §13 | every decision since the review: German used natively, sources in/out, OCR gates, lexicon v2, science bucket, copyright decisions |
| 3 | `reports/science_status_2026-10-06.md` | what is running, what is done, exact resume commands |
| 4 | `reports/science_feasibility_2026-10-06.md` | every science source: route, terms, size |
| 5 | `config/sources.yaml`, `config/paths.yaml` | every source with dest, licence, access method, filters; all paths |
| 6 | `C:\Users\27409\Desktop\APS360 Model\remote-gpu.md` (outside the repo) | the 5080 box: SSH, quoting, WSL, tmux, rules (verified commands) |
| 7 | `docs/TASKS.md` | ordered plan with gates (Phase 1: audit due 10-10, corpus freeze 10-20) |
| 8 | `reports/ocr_quality_*.md`, `ocr_gates_*.md`, `ocr_lexicon_*.md` | OCR evidence behind the gates |

### 13.2 Machines

| machine | role | notes |
|---|---|---|
| 5080 box (`ssh gpu`, Windows 11 + WSL2 Ubuntu, repo `/home/an/1939`, user `an`) | downloads, processing, training | default SSH shell is **cmd**; run Linux via `wsl -d Ubuntu --`. Python: `~/miniconda3/envs/torch-gpu/bin/python`. VPN = Windows proxy `127.0.0.1:7890`, **reachable only by Windows programs** (`/mnt/c/Windows/System32/curl.exe -x http://127.0.0.1:7890`), not by WSL. VPN traffic is **metered (~50 GB left on 2026-10-06)** |
| local PC (the "2080", Windows, repo `C:\Users\27409\Desktop\1939`) | editing, git push; downloads of anything slow or blocked on the 5080 | its network is **not** metered. Python 3.12 with pyarrow/pyyaml (user site). Large results go to the 5080 via Baidu Netdisk (Andrew uploads/downloads) |

### 13.3 Controlling the 5080 (all verified)

- Multi-line work: write a local script, run `ssh gpu 'wsl -d Ubuntu -- bash -s' < script.sh`.
  Filter the harmless UTF-16 WSL warning with `| tr -d '\000' | grep -av "localhost proxy"`.
- Inline: single quotes locally, double quotes for the remote part; never put `|` inside nested
  quotes (cmd breaks them) — use `bash -s` instead. PowerShell on the box: pipe a script into
  `ssh gpu 'powershell -NoProfile -Command -'`.
- Windows programs called from WSL (`curl.exe`, `git.exe`) need `< /dev/null`.
- **Long jobs that use the VPN** (curl.exe interop) must be started through WMI so they survive
  SSH: `(echo '$Script = "<script>.sh"'; echo '$DlArgs = "<args>"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'`.
  Pure-Linux jobs can use tmux, but tmux sessions have vanished once without explanation; WMI is safer.
- Code sync: commit + push locally, then `ssh gpu 'wsl -d Ubuntu -- bash -s -- data/corpus-v1' < scripts/remote_pull.sh`
  (always pass the branch; without it the box checks out `main`). The script falls back to a direct
  GitHub fetch if the proxy is down and moves aside reports/config tables that the box generated.
- Tables generated on the box (`config/*_files.tsv`, `*_items.tsv`, reports) must be copied back and
  committed locally: `ssh gpu 'wsl -d Ubuntu -- bash -c "cd /home/an/1939 && tar czf - <files> | base64 -w0"' | ... | base64 -d | tar xzf -`.
- Rules (from remote-gpu.md): ask Andrew before shutdown/reboot, sshd/firewall/Tailscale changes, Windows
  update or power settings, deleting remote data; no password login, no port forwarding; kill only
  processes/sessions you created; never write secrets anywhere.

### 13.4 Network and VPN budget

- Measured from the 5080: archive.org, Gutenberg mirror, GitHub and Wikimedia are unusable or very
  slow without the VPN; static.case.law and pubs.usgs.gov work directly; chroniclingamerica/tile.loc.gov
  answer 403 directly. From the 2080: LoC ~50 MB/s, Wikimedia ~3 MB/s. archive.org djvu.txt measured
  2026-10-06 12:30 (10 files each, 1 s gap): 2.8-5.4 s/file from the 2080 vs 7-20 s/file on the 5080
  through the VPN (Annalen d. Physik 2.8 vs 20.2); Gutenberg 4.7 vs 6.3 s/file (2 s gap); zbMATH OAI
  ~5 s/page vs ~10 s (VPN) / ~50 s (direct). Since then the 2080 downloads archive.org, Gutenberg,
  Dingler and JFM (`scripts/fetch_local_chain.sh`, key lists in `data/logs/local_keys/`, reverse file
  order so a resumed 5080 chain meets it in the middle; `jfm_harvest --direct`).
- Downloader routing: `src.data.download --via proxy|mirror|auto`. **Force `--via proxy` for archive.org,
  Gutenberg, GitHub** (the `auto` probe flaps and then hammers dead direct routes); `--via mirror`
  (= direct) for CAP and USGS.
- **Chronicling America never through the VPN** (124 batches, 83.8 GB). Use `scripts/fetch_local.py`
  on the 2080, ingest there, ship parquet with `scripts/transfer_ingested.py pack|merge`.
- Check usage: `python3 scripts/vpn_usage.py --since 2026-10-06T05:45:00+00:00` on the box (sums
  MANIFEST bytes fetched via proxy; a floor). Tell Andrew before a planned download would exceed
  the remaining budget.
- The VPN client on the box (Forest; see `reports/science_status_2026-10-06.md`) dropped three times on
  2026-10-06 (nothing listening on 7890; the third time ~12:00, `forest-core.exe` gone). Open item:
  find out whether it can be restarted over SSH (identify the client and its executable; options are
  a scheduled task run in Andrew's interactive session via `schtasks /run`, or the client's own
  auto-start/auto-reconnect setting). Creating a task or changing client settings needs Andrew's OK.

### 13.5 Pipeline and tools added on 2026-10-05/06

| tool | does |
|---|---|
| `src/data/download.py` | url/HF sources, resumable, checksums, 429/403 backoff, `|`-separated URL alternatives (404 -> next) |
| `src/data/ia_catalog.py` | archive.org source -> `config/<src>_files.tsv` + `_items.tsv` (dates, window) |
| `src/data/ingest_extra.py` | adapters for every extra source (CR, books, CA, FR, CAP, DDB, Europeana, VB, EJC, RSC, JFM, PSM Wikisource, Gutenberg, generic archive.org/PDF text) |
| `src/data/ocr_quality.py` | lexicon v2 (anchor + pool + variant filter), histograms, gates, VB segment cleanup, gate reports |
| `src/data/jfm_harvest.py`, `gutenberg_select.py`, `usgs_catalog.py`, `science_books.py` | science-bucket collection |
| `scripts/science_ia.sh`, `science_ia_now.sh`, `science_misc.sh`, `science_usgs.sh`, `jfm_run.sh`, `ingest_run.sh` | WMI-launchable chains on the box |
| `scripts/fetch_local.py`, `scripts/transfer_ingested.py` | 2080-side download and the Baidu Netdisk hand-over |

### 13.6 Pitfalls already hit

- The Bash tool halves backslashes in heredocs: write Python patch scripts with the Write tool, or use Edit.
- Under `set -euo pipefail`, `x=$(... | grep ...)` with no match aborts the script silently.
- Auto mode's safety check can start blocking every side-effecting command for the rest of a long
  conversation; then switch to the default permission mode or start a fresh session with §13.7.
- `ProcessPoolExecutor`, not `multiprocessing.Pool` (hangs on OOM-killed workers); flush parquet by bytes.
- Never `remote_pull.sh` without the branch argument.
- archive.org `/download/<id>/<file>` from the 2080 can redirect to a cache node (`dn*.ca.archive.org`)
  that answers HTTP 500 for hours while the item's two storage replicas (metadata `d1`/`d2` + `dir`)
  serve the file; `scripts/fetch_local.py` falls back to them. There is no other public mirror.
- On the 5080, check memory and kill hung or finished project processes on every visit (Andrew,
  2026-10-06: stray python loops keep the box hot); keep the two WSL `sleep infinity` keep-alives.

### 13.7 Starting prompt for a new session

See `reports/science_status_2026-10-06.md` for the current state; a ready-to-paste prompt is at the
end of that file.
