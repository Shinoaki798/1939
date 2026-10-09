# Problems and fixes

Each entry: problem → cause → fix → effect. Material for the "challenges" part of the reports.

## Data access

1. **Remote box behind a metered VPN.** The 5080 (in China) reaches Hugging Face and GitHub only
   through a Windows-side proxy with ~50 GB left; WSL cannot use it. → Bulk sources that would exceed
   the budget (Chronicling America, 68 GB) and slow archive.org items were fetched on the local PC and
   shipped as ingested parquet through Baidu Netdisk; code reaches the box as a `git bundle` over SSH
   when GitHub is down. → No VPN-limited stage on the critical path.
2. **archive.org failures.** A storage node returned HTTP 500 for whole items; lending copies answered
   401/403. → Fall back to the item's replica nodes (d1/d2); give up on restricted items instead of
   pausing 30 min per attempt.
3. **JFM (zbMATH) OAI harvest lost its resumption token, and a new walk returns records in a different
   order.** → Pages are keyed per walk and saved only when they hold unseen record identifiers; the
   harvest stops when every identifier is saved (99.79 %).
4. **Caselaw volumes overwrote each other** (volume numbers repeat across reporters). → File names
   `<reporter>__<vol>`; the downloader refuses tables whose keys share a file name; 120 volumes refetched.
5. **Jobs on the local PC died** when the Claude app restarted, when a visible console window was
   closed, and after a Windows Update restart. → Long jobs start through WMI with a hidden window; all
   chains resume from verified files.

## Scale on one machine

6. **MinHash matching read 29 TB in 3.4 h** from a 33.6 GB memory-mapped signature file (random
   access). → Verification on 8-bit b-bit signatures held in RAM (bias-corrected), vectorised union-find
   band by band, documents under 25 words exact-only, band keys streamed instead of 50 GB of temporary
   files. → English dedup of 65.7M documents finished in one run.
7. **Out of memory in selection and signing** (one American Stories year exceeds a worker's RAM; 14
   signing workers). → Batched writes (50k rows), 8 workers, preallocated arrays.
8. **Out of memory in the reprint pass (2b)** — one worker reached 22.5 GB: 358 English-detected
   German pages made every year 1900-39 eligible, and each year was loaded whole as Python lists,
   including years with complete books. → 2b sources per language (English: Chronicling America and
   Federal Register, only 1930-39), the previous year streamed into a Bloom filter, year shards reused.
   → 22 min, 5 GB per worker.
9. **Windows paged ~160 GB/h to the SSD** (its wear counter moved from 1 % to 2 %): the filter held
   each input's whole output in memory (~5 GB per worker), the WSL VM filled its 48 GB and Windows was
   left with 3 GB. Found with Hyper-V virtual-disk counters (the VM wrote 3 GB in 3 min, the disk 11.7
   GB, 8.2 GB of it page-outs). → Output written in 20k-row groups (1.4-1.7 GB per worker); paging 0.
10. **CPU at 88.6 °C / 154 W** with 14 workers. → 8 workers (~80 °C); temperatures, memory and SSD
    writes checked every 10 minutes with alarms (CPU 90 °C, GPU 85 °C, SSD 75 °C, Windows memory
    < 1.5 GB).

## Correctness found during the run

11. **German paragraph dedup did nothing**: DDB and Europeana pages are one line of ~3,000 words, and
    blocks were cut only at line breaks, so a page was a single block. → Long lines are cut at the first
    sentence end past 60 words. → 2.2M German reprint blocks removed (155M words) instead of whole-page
    matches only.
12. **German-language American Stories is unusable**: Fraktur was OCR'd as Latin script ("Beiten ber,
    iino Die 8inter bodIt"); the OCR gate removes 99 %. → Documented; the German-language US slice is
    effectively Chronicling America's ~6M words.
13. **Audit mixture counted science twice** (shares summed to 111 %). → Science reserved inside the
    budget.
14. **Mixture**: PyYAML read `7.37e9` as a string (needs `7.37e+9`); chunks of one journal volume sat in
    both language pools under one id; the 10 % science rule was exceeded once the German cap bound;
    the caps left 0.89B of the 12B budget unused. → Signed exponents, per-language parent ids, science
    capped at 10 % of each language, the 1920s fill the remainder (user).
15. **Extraction artefacts in training-only sources** (Europeana repeated words at line breaks, spaced
    punctuation in DDB/RSC, ⸗ line-end hyphens, Congressional Record without commas). → Normalised at
    shard time; American Stories untouched so the scored text is the original.
16. **The 1920-22 spike** (English 1.1-1.2B words a year vs ~0.25B in 1925-29) is real: Chronicling
    America digitised public-domain years (≤ 1922) first. The recency weighting damps it in the mixture.

## Tooling

17. **Remote shell quoting**: the box's SSH shell is Windows cmd, which breaks nested quotes containing
    `|`; multi-line PowerShell piped to stdin fails silently. → Remote work goes through
    `wsl -d Ubuntu -- bash -s` with a heredoc; PowerShell one-liners or `-File` scripts.

18. **A silent SSD writer: WSLg's weston crash loop** on the 5080 (no GUI apps are used): a 171 MB dump
    every 103 s, ~6 GB/h. Found while cleaning the disk (`Temp\wsl-crashes` refilled after deletion).
    → WSLg disabled (`guiApplications=false`).
