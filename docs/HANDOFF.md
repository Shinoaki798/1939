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
