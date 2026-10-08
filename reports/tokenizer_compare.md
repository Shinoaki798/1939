# Tokenizer comparison (2026-10-08)

Byte-level BPE trained once on the mixture-weighted sample (`data/tokenizer/sample`, MANIFEST there); the smaller vocabularies are prefixes of the largest run's merges. Held-out documents only (never in the tokenizer sample); text normalised as in training.

| eval set | vocab | docs | bytes / token | tokens / word |
|---|---|---|---|---|
| en: American Stories held-out (scored text) | 32,768 | 20,000 | 3.592 | 1.593 |
| en: American Stories held-out (scored text) | 49,152 | 20,000 | 3.688 | 1.551 |
| en: American Stories held-out (scored text) | 65,536 | 20,000 | 3.743 | 1.529 |
| de: DDB + Europeana held-out | 32,768 | 3,000 | 3.327 | 2.002 |
| de: DDB + Europeana held-out | 49,152 | 3,000 | 3.480 | 1.914 |
| de: DDB + Europeana held-out | 65,536 | 3,000 | 3.582 | 1.860 |

| vocab | embedding params (tied, d 1024) | share of a 350M model | output layer vs 24-layer body (FLOPs/token) |
|---|---|---|---|
| 32,768 | 33.6M | 9.6% | 11.1% |
| 49,152 | 50.3M | 14.4% | 16.7% |
| 65,536 | 67.1M | 19.2% | 22.2% |

Tokens each size adds over the previous one; word-initial = a space then >= 3 letters; junk = begins no word of the EN or DE OCR lexicon:

| vocab | tokens added | word-initial | junk | examples of junk |
|---|---|---|---|---|
| 32,768 | 32,768 | 18,436 | 7 (0.0%) | Mahnomen Wardman Brooksville Plentywood tiie Pocomoke mufs |
| 49,152 | 16,384 | 9,848 | 34 (0.3%) | socalled Wiggily Mishawaka CWA Askov Holtville Gallinger Blanton Guffey vealers Berryville WMAL Coulee kidnapers Scottsboro läfst WJSV Foxx anid Waubun Hendersonville BROOKSVILLE WRC Weslaco Onancock |
| 65,536 | 16,384 | 9,834 | 65 (0.7%) | hydrochloric McConnelsville Mattson Bloxom Manchukuo LaGuardia Follette Aggies Connally wiii Rcichs Wilby Mclnt Ahearn Philco kidnaper kénnen Ansonia Chincoteague Fosston Scobey Hartnett Einflufs Manush Feldstärke |

Probe terms that are a single token (must be none): 32,768: blitz, occupation, resistance; 49,152: blitz, occupation, resistance; 65,536: blitz, collaboration, occupation, resistance
