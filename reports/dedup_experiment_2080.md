# Selection + dedup experiment on the 2080 (2026-10-07)

Code: `src/data/select.py`, `src/data/dedup.py`, tests `tests/test_select_dedup.py` (7 pass).
Data: everything ingested on the 2080: Chronicling America 1930-1939.06 pages, PSM Wikisource and the
23 science sources plus JFM of the second hand-over. No American Stories, books, CR or CAP here, so
the pre-1920 sampling and the American Stories title rule were not exercised.

## Selection (`python -m src.data.select census`, then `run`)

| pool | documents | words |
|---|---|---|
| en | 463,398 | 1,338M |
| de | 302,407 | 348M |

Drops worth noting (full counts in `data/logs/select_local.log` and the per-source MANIFESTs):

- Chronicling America keeps 857M of 1,082M words in the English pool; the dropped pages are mostly
  Polish, Czech, Scandinavian, Spanish and Italian immigrant-press pages (the CA batches include them).
  1,992 German pages go to the German pool.
- JFM: 9,668 reviews (7 %) are `und` (too few stopwords to tell, mostly one-line reviews) and are
  dropped by the language rule; 1,227 short reviews are guessed Czech. Kept: 135,973 German reviews.
- Science journals lose a few hundred rows each to the language rule (tables, OCR debris guessed as
  cs/pl); Math. Annalen and Crelle send 587 + 1,066 French articles out, their English articles to the
  English pool.
- Speed: seconds per source (metadata + text rewrite); not a bottleneck.

## Dedup (`python -m src.data.dedup --lang <l> --dry-run --samples ...`)

| pool | documents | dropped | exact | near (J >= 0.8) |
|---|---|---|---|---|
| de | 302,407 | 11,420 | 11,394 | 26 |
| en | 463,359 | 257 | 55 | 202 |

- de: JFM 10,571 (the overlapping OAI walks, as expected), Meyers 843 (archive.org holds two copies of
  some volumes); everything else ~0.
- en: Chronicling America 193 of 285,405 pages (0.1 %), science journals ~0.
- MinHash estimate vs true 5-gram Jaccard on checked pairs: 0.836 vs 0.804, 0.805 vs 0.814 (within the
  expected +-0.04 at 128 permutations).
- Speed: signing ~0.8M tokens/s per worker (one 71M-word Dingler file took 90 s alone); 1.34B English
  tokens in 365 s on 16 workers. Matching: 2 s for 0.46M documents.

## Findings

1. **The 0.8 threshold is too strict for OCR text.** Samples in `reports/dedup_samples_en_2080.md`
   (0.80-0.85, 0.70-0.80) and `..._low.md` (0.60-0.70, 0.50-0.60): almost every pair down to ~0.55 is a
   real duplicate (the same article in two editions of one day, a county financial statement reprinted
   weekly, the same advertisement or comic-page header). One OCR error spoils 5 word 5-grams, so true
   duplicates of noisy pages land well below 0.8. The only non-duplicate seen above 0.5 was a pair of
   JFI index pages (0.62).
2. **Whole-page dedup does not catch reprinted articles.** A page holds dozens of articles; one
   reprinted wire story changes the page-to-page Jaccard very little. American Stories is article-level
   and will dedup properly; the page-level sources (Chronicling America 1.2B, DDB 12.9B, Europeana 4.7B
   words) need dedup below page level to remove reprints (split pages into paragraph blocks, match
   blocks across the pool including American Stories articles, drop repeated blocks from the page).
3. Scale on the 5080: the English pool there is ~16B words / ~75M documents, so 128-value signatures
   are ~38 GB. Matching must read them through a memory map (and build band keys in row chunks)
   instead of concatenating them in RAM; signing estimate ~1 h on 16 cores.

## Open decisions (Andrew)

- Threshold: keep 0.8 (proposal wording) or lower it for OCR text (e.g. 0.6 with 32 x 4 bands)?
- Sub-page dedup for page-level sources: yes / no.
