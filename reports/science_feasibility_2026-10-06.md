# Science bucket: source feasibility (2026-10-06)

Research only: documentation, terms pages, robots.txt and a few metadata requests per source; no bulk
downloads. Word counts are rough estimates from sampled text sizes. "IA" = archive.org; `sim_` items are
microfilm scans with issue-level dates and `_djvu.txt` OCR. US public domain now covers works published
in 1930 or earlier (1930 entered PD on 2026-01-01). Rights notes are not legal advice.

## Already on disk

| source | status | volume |
|---|---|---|
| Royal Society Corpus 6.0.4 open (1665-1920, CC BY-NC-SA 4.0) | ingested | 17,520 papers, 78.6M words |
| JSTOR Early Journal Content (<= 1922, non-commercial terms) | ingesting; STEM title list to follow | 1.12B words all journals; STEM share TBD |

## English

| # | source | text? | route | keyed/OCR | terms | est. words in window | recommendation |
|---|---|---|---|---|---|---|---|
| 4 | PNAS 1915-1939-06 | yes (IA); PMC no | IA `pub_proceedings-of-the-national-academy-of-sciences-usa` (`sim_`, 14 issues/yr, all years); BHL v.1-8; EJC 1915-22 | OCR | PMC: scans only, not OA, systematic download prohibited -> not used. IA: no rights field; no renewals found (Online Books Page) -> 1931-39 likely PD by non-renewal | 5-7M | IA, issue-level, dated by issue |
| 5 | Bulletin AMS 1891-1939-06 | yes (IA) | IA `pub_american-mathematical-society-bulletin` + Google volume scans pre-1894 | OCR | ams.org behind Cloudflare challenge (not bypassed); Project Euclid bans bulk. Renewals start Dec 1945 -> likely PD | 8-9M | IA |
| 6 | Encyclopaedia Britannica 11th (1910-11) | yes | Gutenberg slices (keyed, vols 2-17); Wikisource proofread vols 2-10, 12-14, 16, 17, 19 (dump 3.4 GB); IA OCR all 29 vols (`encyclopaediabri..chisrich`) | keyed + OCR | US PD | ~44M | Gutenberg keyed for vols 2-17, IA OCR for 1, 18-28; skip index vol 29 |
| 7 | Encyclopedia Americana (1918-20 ed.) | yes | IA LoC scans `encyclopediaamer01unse`..`30unse` (1922 printing; same prefix also holds 1967-2004 editions -> filter by date) | OCR | LoC: no known restrictions | ~25M | IA LoC set |
| 8 | NACA TR/TN/TM 1915-1939-06 | yes | IA `NASA_NTRS_Archive_<id>` (2,517 items in window); NTRS API for metadata (its own .txt OCR is poor) | OCR | US gov PD; NTRS allows harvesting | 10-15M | IA djvu.txt. **TMs (~1/3) are period human translations of German/French papers -> decision** |
| 9 | NBS J. Research 1928-1939-06 (+ Sci./Tech. Papers 1910-28) | yes | IA `pub_united-states-national-bureau-of-standards-journal`, `NISTJournalofResearch`, `NBSScientificPapers`; nvlpubs PDFs with OCR layer | OCR | US gov PD (NIST statement) | 8-11M | IA |
| 10 | Nature <= 1930 | partial | IA BHL `nature{vol}{years}lock` (v.1-111) + Google `nature*goog`; >= 71 of 126 volumes identified | OCR | US PD <= 1930; UK life+70 for some articles outside the US | >= 50M | BHL first, Google to fill |
| 11a | J. Franklin Institute <= 1930 | yes | IA UofT `journalfranklini{N}fran` (vols 1-196, 1826-1923) + `sim_` 1924-30 | OCR | US PD | ~50M | IA |
| 11b | Popular Science Monthly 1872-1930 | yes | Wikisource vols 1-87 (1872-1915, proofread); IA `pub_popular-science` 1916-30 | keyed + OCR | US PD | 22-30M keyed + more | Wikisource dump for 1-87, IA for 88-117 |
| 11c | Scientific American <= 1930 | yes | IA `pub_scientific-american` (4,148 issues); Gutenberg 87 keyed issues | OCR | US PD | 100M+ | IA, issue-level; heavy ads |
| 12 | Bell System Technical Journal 1922-1939-04 | yes | IA `bstj-archives` (645 article items to Apr 1939; July 1939 is embargo) | OCR | posted by Bell Labs; no renewals found -> likely PD | 3-4M | IA, "open, non-commercial" as instructed |
| 13 | Monthly Weather Review / Public Health Reports / USGS Prof. Papers 1920-1939-06 | yes | IA `pub_monthly-weather-review`, `pub_public-health-reports`; USGS Pubs API + PDF text layer (196 PP; also 608 Bulletins, 464 Water-Supply Papers) | OCR | US gov PD; AMS site and PMC not scriptable / prohibited -> IA | MWR ~8M (tabular), PHR 10-15M, PP 5-10M | IA + USGS API |
| 14 | US patents 1920-1939-06 | **no** | no account-free bulk OCR (USPTO full text from 1976; Google BigQuery needs an account). Alternative: Official Gazette scans on IA (~200M words of claims/lists) | - | - | - | **skip** (Gazette would swamp the mix) |
| 15 | Subject filter of existing books | partial | neither LoC-PD-Books nor pre_1929_books has subject/LCC fields; pre_1929_books ids are IA ids -> IA `subject`/`call_number` metadata; LoC catalog blocks scripts | - | - | TBD | pre_1929_books via IA metadata; LoC by title keywords only |
| 16 | Project Gutenberg science/maths | yes | PG catalog CSV has LoCC + subjects: Q* 2,501 English books (QA-QD 577); 1,024 with all persons dead by 1930; fetch via PG robot harvest / mirror rsync | keyed | US PD; automated use only via harvest endpoint or mirrors | 40-70M | Q* by LoCC, publication year <= 1930 by death-year rule + title-page check |
| - | Physical Review 1929-1939-06 | yes (unexpected) | IA `pub_physical-review` (305 issues 1929-39) | OCR | renewals start 1956 -> likely PD by non-renewal; cut at 1939-06-15 issue | ~15M | **decision** (non-renewal reasoning) |
| - | Reviews of Modern Physics | no | - | - | - | - | drop |

## German

| # | source | text? | route | keyed/OCR | terms | dates | est. size | recommendation |
|---|---|---|---|---|---|---|---|---|
| 17 | JFM (zbMATH Open) | yes | OAI-PMH `oai.zbmath.org/v1/?verb=ListRecords&metadataPrefix=oai_zb_preview&set=JFM` (223,270 records, ~2,233 pages) | keyed (ERAM) | CC BY-SA 4.0; zbmath.org blocks AI crawlers -> OAI/API only | paper year + JFM volume (from id); **no volume publication date in data**; Bd. 64 (1938) appeared 1940/42 (unverified), lags up to 7 years | vol <= 61: ~153k reviews, ~60-70M tokens | harvest, keep reviews of **vol <= 61** (strict); vol 62 only if its date is confirmed; strip modern cross-refs and placeholder strings |
| 17b | Zentralblatt 1931-39 | mostly no | same API | 35,759 scanned (no text); ~2k keyed incl. modern retro-reviews | CC BY-SA 4.0 | - | ~2k | **skip** (leakage) |
| 18 | Math. Annalen, Crelle | yes | GDZ METS + per-page OCR (`PPN235181684`, `PPN243919689`); IA fallback | OCR (formulas garbled) | GDZ: non-commercial research only, no redistribution of reproductions; 1930s under a Springer arrangement | volume year only -> Math. Ann. <= vol 115, Crelle <= vol 179 (<= 1938) | ~350-450M chars | GDZ; **decision on 1931-38 volumes** |
| 19 | Encyklopaedie d. math. Wiss. (1898-1935) | yes | GDZ METS/OCR | OCR | GDZ terms | part-volume year; 2nd ed. 1939-58 must be dropped | ~20k pages? | GDZ, 1st ed. only |
| 20 | Dingler 1820-1931 | yes, best | GitHub `deutschestextarchiv/dingler` (375 TEI volumes, 988 MB) | corrected OCR + TEI | CC BY-SA 4.0 | volume year | ~130-150M tokens | git clone; `<text>` only |
| 21 | Sitzungsberichte PAW | partial | BBAW images only (1882-1900); scattered IA items | OCR (IA) | CC BY-SA (BBAW) | year | small | low priority |
| 22 | Annalen der Physik | yes <= 1930 | IA `sim_annalen-der-physik_*` (1,856 items <= 1925, 398 more 1926-39); Gallica allows academic AI use but robots.txt disallows text endpoints -> would need an email to BnF | OCR | US PD <= 1930 | issue dates | ~0.4-0.6 GB <= 1925 | IA <= 1930 |
| 23 | Meyers 6th ed. (zeno.org) | not obtainable from zeno | zeno terms forbid robots and database extraction; IA scans of the 6th ed. (84 items, Fraktur OCR) | OCR (IA) | US PD | 1902-09 | ~155k entries | IA OCR, gates decide; or the user asks zeno.org |
| 24 | MDZ | yes | IIIF + hOCR per page; daily OCR cap; per-item rights (NoC-NC, PDM, some TDM reservations) | OCR | per item | per item | large | later, targeted |
| - | Naturwissenschaften, Phys. Zeitschrift (1930s) | on IA | `sim_naturwissenschaften_*` (1,364 issues), `per_physikalische-zeitschrift_*` (222 in 1930s) | OCR | <= 1930 US PD; 1931-39 not cleared | issue dates | ~135 + 140 MB | <= 1930 yes; **1931-39 decision** |
| - | Zeitschrift fuer Physik | no | Springer paywall, no IA copy | - | - | - | - | drop |

## Cross-cutting caveats

- Strip modern text before dedup and the C1 screen: Google-scan boilerplate, Gutenberg headers/licence,
  Wikisource templates, zbMATH placeholders and inserted cross-references, Dingler `teiHeader`, GDZ cover pages.
- Whole-volume/issue files must be split (articles, issues or page chunks) before the C1 whole-document drop.
- Date every document by issue or volume date; serial `year` fields on IA are often the start year.
- Several sources exist in multiple copies (EB11, Nature, JFI, PSM, SciAm): one canonical copy per volume, MinHash for the rest.
- IA publishes no numeric rate limit: serial requests, backoff on 429, `_djvu.txt` only (no PDFs).
- Rough potential: English ~500M+ words (more than the English 10 % cap of ~750M tokens needs only with
  SciAm/Nature/JFI); German JFM + Dingler + Math. Ann./Crelle + Annalen <= 1930 already exceed the German cap.
