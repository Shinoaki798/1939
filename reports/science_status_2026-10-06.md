# Science bucket / data collection: status and resume steps (2026-10-06, ~05:00 EDT)

Not yet committed (auto mode blocked further git commits in the session that wrote this):
`scripts/transfer_ingested.py` (pack on the 2080, verify + merge on the 5080) and this file.
Commit both on `data/corpus-v1` and push, then pull on the 5080 (`scripts/remote_pull.sh data/corpus-v1`).

## Running

| where | job | state |
|---|---|---|
| 5080 | CAP download (`--via mirror`, direct) | ~2,900 / 3,464 volumes |
| 5080 | USGS Professional Papers (`science_usgs.sh`, direct) | ~150 / 186 PDFs |
| 5080 | JFM OAI harvest (`jfm_run.sh`) | ~23 % of 223,270 records; falls back to direct (slow) while the VPN is down |
| 2080 | Chronicling America ingest (101 batches) -> `data/ingested/chronicling_america` | **done**: 362,170 pages, 1.08B words (pages: en 293,322, fr 18,812, cs 13,869, pl 13,137, es 6,514), 3.1 GB parquet; ready to pack |

## Done

- 2080: Chronicling America raw, 101 batches / 68.1 GB, all sha256-verified (`data/raw/chronicling_america`).
- 2080: Popular Science Monthly vols 1-87 (Wikisource, proofread) ingested: 15,173 documents, 31.7M words (`data/ingested/psm_wikisource`, 72 MB).
- 5080 ingested: CAP so far 429,421 cases / 512M words; Chronicling America 23 batches / 109M words; RSC 78.6M; JSTOR EJC (all journals) 1.10B, science titles 306M.
- 5080 downloaded: PNAS (229), EB11 OCR vols (14), Nature (59), Bull. AMS (164 / 516), Gutenberg (782 / 3,133).
- Science book selection: `config/science_books_selection.tsv` (5,742 books, ~531M words).

## Paused: needs the 5080 VPN (proxy 127.0.0.1:7890 was down at ~04:45)

When the VPN is back, relaunch (all resume; verified files are skipped):

    (echo '$Script = "science_ia.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
    (echo '$Script = "science_ia_now.sh"'; echo '$DlArgs = "annalen_physik_ia naturwiss_ia physz_ia math_annalen_ia crelle_ia encyklopaedie_ia meyers6_ia sitzungsberichte_ia"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
    (echo '$Script = "science_misc.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'

VPN traffic since it was restored (05:45 UTC): 10.0 GB (8.7 GB was the Chronicling America run that was
stopped; Chronicling America must never go through the VPN). Remaining proxy downloads ~8 GB.
Check with `python3 scripts/vpn_usage.py --since 2026-10-06T05:45:00+00:00` on the 5080.

## Transfer 2080 -> 5080 (Baidu Netdisk)

After the 2080 CA ingest finishes:

    python scripts/transfer_ingested.py pack --sources chronicling_america,psm_wikisource --out D:/1939_transfer

Upload `D:/1939_transfer` to Baidu Netdisk, download it on the 5080 (e.g. to `C:\Users\AN\Downloads\1939_transfer`), then on the 5080:

    python scripts/transfer_ingested.py merge --in /mnt/c/Users/AN/Downloads/1939_transfer

## Still to do

MDZ (after the German sources are in); ingest of every science source once downloaded; OCR gates
(keyed sources skip them); MinHash dedup; splits; C1 screen; science-bucket audit (tokens by language x
decade x source, licence table, keyed vs OCR share).
