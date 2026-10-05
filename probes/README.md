# Probe sets

## rq1_terms.csv (to build; freeze with CHECKSUMS before any model is evaluated)
Columns: term, class (C1|C2|C3|C4), first_attested_year, control_term,
control_freq_match_note, rationale, context_1 … context_5.

Classes (HANDOFF §6.3): C1 coinage; C2 compositional; C3 existing word, new
sense; C4 existing proper noun, new association. ~200 terms, roughly balanced.
Controls: same semantic field, pre-cutoff, frequency-matched in the training
split. Contexts: ≥5 natural sentences from post-cutoff articles with the term
in place; the control is substituted into the same sentence frames.

`rq1_seed.csv` is the starting list from the proposal; it is NOT the frozen set.

## rq3_propositions.csv (to build; freeze likewise)
Columns: id, proposition, true_continuation, false_continuation,
resolution_source, context_us_jun, context_de_jun, context_fr_jun,
context_us_aug, context_de_aug, context_fr_aug (paths into rq3_contexts/).
~60 binary propositions about 1939-09 … 1945-12 resolvable from the record.
Both continuations are English; contexts may be translated (DE/FR) but are
never themselves scored.

## CHECKSUMS
sha256 of each frozen file. Any later change is versioned (v2) and both are
reported.
