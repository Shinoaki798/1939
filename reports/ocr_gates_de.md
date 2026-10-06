# OCR gates, de

Gate parameters (also written to every filtered shard's MANIFEST):

```json
{
 "scorer_version": "2026-10-06.2",
 "lang": "de",
 "lexicon_file": "e04cdd822061.txt",
 "lexicon_sha256": "e04cdd8220615291cbff602d7c33a2b331f1d29607833a32773d0a59749ad5cd",
 "lexicon_version": "v2",
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
| ddb_newspapers_de | 1900-1919 | 7567 | 0.0% | 8.7% | 10.4% | 19.1% | 0.0% | 5.0% | 11.7% | 16.7% |
| ddb_newspapers_de | 1920-1929 | 3730 | 0.0% | 0.9% | 15.5% | 16.4% | 0.0% | 0.7% | 14.2% | 15.0% |
| ddb_newspapers_de | 1930-1933 | 1600 | 0.1% | 0.6% | 12.6% | 13.2% | 0.0% | 0.2% | 12.5% | 12.7% |
| ddb_newspapers_de | 1934-1936 | 1200 | 0.1% | 0.2% | 13.2% | 13.5% | 0.0% | 0.2% | 11.9% | 12.1% |
| ddb_newspapers_de | 1937-1939 | 1039 | 0.0% | 0.3% | 15.3% | 15.6% | 0.0% | 0.2% | 13.8% | 14.0% |
| ddb_newspapers_de | 1940-1955 | 727 | 0.0% | 1.0% | 21.9% | 22.8% | 0.0% | 0.6% | 16.1% | 16.7% |
| europeana_newspapers_de | 1900-1919 | 8000 | 0.0% | 8.2% | 10.1% | 18.2% | 0.0% | 8.5% | 10.0% | 18.5% |
| europeana_newspapers_de | 1920-1929 | 4000 | 0.2% | 11.8% | 6.9% | 19.0% | 0.0% | 10.1% | 8.5% | 18.6% |
| europeana_newspapers_de | 1930-1933 | 1326 | 0.1% | 9.9% | 7.3% | 17.3% | 0.0% | 10.4% | 9.2% | 19.6% |
| europeana_newspapers_de | 1934-1936 | 472 | 0.0% | 7.2% | 3.0% | 10.2% | 0.0% | 7.7% | 3.8% | 11.5% |
| europeana_newspapers_de | 1937-1939 | 415 | 0.0% | 8.0% | 1.7% | 9.6% | 0.0% | 7.3% | 2.0% | 9.2% |

## Segment cleanup (issue-level sources: voelkischer_beobachter_de)

Every document, not a sample. Blank-line blocks merged to >= 30 tokens; segments with word share < 0.7 dropped.

| source | period | segments | segments dropped | tokens | tokens dropped |
|---|---|---|---|---|---|
| voelkischer_beobachter_de | 1920-1929 | 59284 | 6.1% | 3,550,808 | 4.3% |
| voelkischer_beobachter_de | 1930-1933 | 150200 | 5.3% | 8,431,128 | 4.0% |

Boundary samples (0.65-0.75 word share, dated <= 1939): logs/ocr_gates_de_boundary.md.

