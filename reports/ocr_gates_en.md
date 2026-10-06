# OCR gates, en

Gate parameters (also written to every filtered shard's MANIFEST):

```json
{
 "scorer_version": "2026-10-06.2",
 "lang": "en",
 "lexicon_file": "52eef5d79ce7.txt",
 "lexicon_sha256": "52eef5d79ce770764e7d73b50dcbe50e8b9791e6688e85c5631d33627fd48802",
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
| american_stories | 1900-1919 | 8000 | 31.9% | 0.2% | 0.6% | 32.8% | 6.5% | 0.2% | 0.6% | 7.4% |
| american_stories | 1920-1929 | 3845 | 30.5% | 0.5% | 0.5% | 31.5% | 6.1% | 0.6% | 0.8% | 7.5% |
| american_stories | 1930-1933 | 1243 | 30.0% | 0.1% | 1.1% | 31.2% | 7.1% | 0.1% | 1.5% | 8.7% |
| american_stories | 1934-1936 | 1026 | 29.3% | 0.3% | 0.8% | 30.4% | 6.3% | 0.2% | 0.4% | 6.9% |
| american_stories | 1937-1939 | 853 | 32.7% | 0.4% | 0.5% | 33.5% | 7.7% | 0.6% | 0.3% | 8.6% |
| american_stories | 1940-1955 | 3864 | 26.3% | 0.4% | 0.7% | 27.4% | 5.6% | 0.5% | 0.8% | 6.9% |
| chronicling_america | 1930-1933 | 1532 | 0.5% | 0.1% | 0.7% | 1.3% | 0.0% | 0.1% | 0.2% | 0.3% |
| chronicling_america | 1934-1936 | 1200 | 0.2% | 0.0% | 0.2% | 0.4% | 0.0% | 0.0% | 0.0% | 0.0% |
| chronicling_america | 1937-1939 | 1017 | 0.4% | 0.1% | 0.2% | 0.7% | 0.0% | 0.1% | 0.1% | 0.2% |
| congressional_record | 1900-1919 | 7742 | 70.5% | 0.0% | 0.0% | 70.5% | 16.7% | 0.0% | 0.1% | 16.8% |
| congressional_record | 1920-1929 | 4000 | 68.6% | 0.0% | 0.1% | 68.7% | 17.2% | 0.0% | 0.1% | 17.4% |
| congressional_record | 1930-1933 | 1600 | 69.2% | 0.0% | 0.1% | 69.4% | 18.5% | 0.0% | 1.1% | 19.6% |
| congressional_record | 1934-1936 | 1200 | 64.3% | 0.0% | 0.2% | 64.5% | 13.4% | 0.0% | 2.1% | 15.5% |
| congressional_record | 1937-1939 | 1200 | 66.6% | 0.0% | 0.0% | 66.6% | 15.7% | 0.0% | 0.0% | 15.7% |
| federal_register | 1934-1936 | 206 | 0.0% | 0.0% | 2.9% | 2.9% | 0.0% | 0.0% | 3.0% | 3.0% |
| federal_register | 1937-1939 | 629 | 0.0% | 0.0% | 5.4% | 5.4% | 0.0% | 0.0% | 13.9% | 13.9% |
| loc_pd_books | 1900-1919 | 8000 | 0.0% | 0.2% | 2.8% | 3.0% | 0.0% | 0.1% | 3.6% | 3.8% |
| loc_pd_books | 1920-1929 | 2308 | 0.0% | 0.3% | 2.0% | 2.3% | 0.0% | 0.9% | 3.2% | 4.1% |
| loc_pd_books | 1930-1933 | 233 | 0.0% | 0.4% | 1.3% | 1.7% | 0.0% | 0.8% | 2.0% | 2.8% |
| loc_pd_books | 1934-1936 | 92 | 0.0% | 0.0% | 2.2% | 2.2% | 0.0% | 0.0% | 0.3% | 0.3% |
| loc_pd_books | 1937-1939 | 80 | 0.0% | 0.0% | 3.8% | 3.8% | 0.0% | 0.0% | 0.1% | 0.1% |
| pre_1929_books | 1900-1919 | 8000 | 0.0% | 0.2% | 2.8% | 3.0% | 0.0% | 0.2% | 4.3% | 4.5% |
| pre_1929_books | 1920-1929 | 3571 | 0.0% | 0.4% | 3.3% | 3.7% | 0.0% | 0.5% | 4.0% | 4.5% |

## Segment cleanup (issue-level sources: voelkischer_beobachter_de)

Every document, not a sample. Blank-line blocks merged to >= 30 tokens; segments with word share < 0.7 dropped.

| source | period | segments | segments dropped | tokens | tokens dropped |
|---|---|---|---|---|---|

Boundary samples (0.65-0.75 word share, dated <= 1939): logs/ocr_gates_en_boundary.md.

