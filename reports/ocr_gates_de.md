# OCR gates, de

Gate parameters (also written to every filtered shard's MANIFEST):

```json
{
 "scorer_version": "2026-10-06.1",
 "lang": "de",
 "lexicon_file": "b102c2243712.txt",
 "lexicon_sha256": "b102c2243712d3006e47063a1f8d2a9372d125353c432760645ed7c93f3e9220",
 "hit_threshold": 0.75,
 "word_share_threshold": 0.7,
 "min_doc_tokens": 50,
 "segment_sources": [
  "voelkischer_beobachter_de"
 ],
 "segment_min_tokens": 30,
 "gate_order": [
  "short",
  "hit_rate",
  "word_share"
 ]
}
```

## Page / article gates

Same deterministic sample as reports/ocr_quality_*.md, including documents too short to score. Gates are applied in order; each drop is attributed to the first gate it fails.

| source | period | docs | docs: short | docs: hit rate | docs: word share | docs: total | words: short | words: hit rate | words: word share | words: total |
|---|---|---|---|---|---|---|---|---|---|---|
| ddb_newspapers_de | 1900-1919 | 7567 | 0.0% | 7.5% | 10.5% | 18.1% | 0.0% | 4.3% | 11.8% | 16.1% |
| ddb_newspapers_de | 1920-1929 | 3730 | 0.0% | 0.7% | 15.6% | 16.4% | 0.0% | 0.5% | 14.5% | 15.0% |
| ddb_newspapers_de | 1930-1933 | 1600 | 0.1% | 0.5% | 12.6% | 13.2% | 0.0% | 0.2% | 12.6% | 12.7% |
| ddb_newspapers_de | 1934-1936 | 1200 | 0.1% | 0.3% | 13.1% | 13.5% | 0.0% | 0.2% | 11.9% | 12.1% |
| ddb_newspapers_de | 1937-1939 | 1039 | 0.0% | 0.1% | 15.5% | 15.6% | 0.0% | 0.0% | 13.9% | 14.0% |
| ddb_newspapers_de | 1940-1955 | 727 | 0.0% | 0.7% | 22.3% | 23.0% | 0.0% | 0.3% | 16.5% | 16.7% |
| europeana_newspapers_de | 1900-1919 | 8000 | 0.0% | 7.1% | 10.3% | 17.4% | 0.0% | 7.3% | 10.4% | 17.7% |
| europeana_newspapers_de | 1920-1929 | 4000 | 0.2% | 10.1% | 7.4% | 17.8% | 0.0% | 8.4% | 9.2% | 17.6% |
| europeana_newspapers_de | 1930-1933 | 1326 | 0.1% | 8.4% | 8.2% | 16.7% | 0.0% | 8.7% | 10.3% | 19.0% |
| europeana_newspapers_de | 1934-1936 | 472 | 0.0% | 5.3% | 4.0% | 9.3% | 0.0% | 5.7% | 5.1% | 10.8% |
| europeana_newspapers_de | 1937-1939 | 415 | 0.0% | 5.8% | 2.9% | 8.7% | 0.0% | 4.3% | 3.6% | 7.9% |

## Segment cleanup (issue-level sources: voelkischer_beobachter_de)

Every document, not a sample. Blank-line blocks merged to >= 30 tokens; segments with word share < 0.7 dropped.

| source | period | segments | segments dropped | tokens | tokens dropped |
|---|---|---|---|---|---|
| voelkischer_beobachter_de | 1920-1929 | 59284 | 6.1% | 3,550,808 | 4.3% |
| voelkischer_beobachter_de | 1930-1933 | 150200 | 5.3% | 8,431,128 | 4.0% |

Boundary samples (0.65-0.75 word share, dated <= 1939): logs/ocr_gates_de_boundary.md.

