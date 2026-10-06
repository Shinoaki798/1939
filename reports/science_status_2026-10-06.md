# Data collection: status and resume steps (2026-10-06, ~13:10 EDT)

Operating guide (machines, remote control, VPN budget, tools, pitfalls): `docs/HANDOFF.md` §13.

## Where things stand

- **All remaining downloads run on the 2080** (local PC, unmetered). Speed test 12:30 (HANDOFF §13.4):
  archive.org 2.8-5.4 s/file vs 7-20 s on the 5080 through the VPN; JFM ~5 s/page vs ~10-50 s.
- **The 5080's download chains are stopped** (Andrew, 13:00; its VPN dropped again ~12:00). Do not
  relaunch `science_ia.sh`, `science_ia_now.sh`/`science_ia_more.sh`, `science_misc.sh` or `jfm_run.sh`:
  they would fetch what the 2080 fetches, and the merge refuses keys ingested on both machines.
  What the 5080 had fetched is ingested there: BAMS 370 issues, Annalen d. Physik 66 vols (1799-1810),
  Gutenberg science 997 books, Sitzungsberichte 1 vol.
- 2080 hand-over (2026-10-06 morning) merged on the 5080: Chronicling America complete (124 batches),
  PSM Wikisource. Transfer files on the box deleted (Andrew's OK).

## Running on the 2080 (started through WMI so they survive the Claude app; logs in `data/logs/`)

Key lists `data/logs/local_keys/<source>.txt`: keys the 5080 has not verified, reverse file order.

| chain | sources in order | files | projected raw text | expected end (EDT) |
|---|---|---|---|---|
| `misc` (DELAY=2) | gutenberg_sci_en, gutenberg_sci_de, dingler | 2,137 | ~1.3 GB + Dingler tarball | 10-06 ~16:00 |
| JFM (`python -m src.data.jfm_harvest --delay 2 --direct`) | whole harvest from page 1 | ~2,233 pages | ~0.7 GB XML (~0.1 GB gz) | 10-06 ~19:00 |
| `de` | sitzungsberichte, meyers6, encyklopaedie, crelle, math_annalen, physz, naturwiss, annalen_physik | 5,100 | ~3.0 GB | 10-06 ~19:00 |
| `en` | sciam, psm, jfi, americana, mwr, phr, physrev, naca, bstj, nbs_papers, nbs_jres, bams | 10,162 | ~1.9 GB | 10-07 ~06:00-08:00 |

Projected sizes: archive.org metadata of 5 items per source (4.9 GB in all), measured averages for
Gutenberg. Total raw ~7 GB on the 2080; ingested parquet ~40 % of that, so the Netdisk hand-over is
~2.5-3 GB. Some archive.org items fail for good (no OCR text: 404; lending copies: 401/403); a
datanode answering HTTP 500 makes a file fail after 8 attempts; rerunning a chain retries failures.

Restart a chain (verified keys are skipped):

    # PowerShell. Start-Process children died with the Claude app restart at 17:45 on 10-06; WMI ones do not.
    # Hide the window (ShowWindow 0): a visible bash window was closed by hand on 10-06 and killed the job.
    $si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0 }
    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ ProcessStartupInformation = $si; CurrentDirectory = 'C:\Users\27409\Desktop\1939'; CommandLine = '"C:\Program Files\Git\bin\bash.exe" scripts/fetch_local_chain.sh <name> <source> ...' }
    # JFM: CommandLine = '"C:\Program Files\Git\bin\bash.exe" -c "python -u -m src.data.jfm_harvest --delay 2 --direct >> data/logs/jfm_harvest_local.log 2>&1"'

## Do next, in this order

1. As each 2080 source finishes: ingest it on the 2080 (`python -m src.data.ingest_extra --source <s>`).
2. When all are done: `python scripts/transfer_ingested.py pack --sources <all fetched sources incl. jfm,
   dingler, gutenberg_sci_de> --out <dir>`; Andrew moves `<dir>` by Baidu Netdisk to
   `C:\Users\AN\Downloads\` on the 5080; merge there with `transfer_ingested.py merge` (as on 10-06).
3. Then OCR gates (keyed sources skip them), MinHash dedup, splits, C1 screen, and the audit
   (`reports/audit_v1.md`, due 2026-10-10). `dedup.py`, `splits.py` and the tokenizer do not exist yet.
4. VPN on the 5080: left alone (Andrew). Client details below if needed.

## Data inventory (raw words, before dedup and OCR gates; M = million)

Training pool by period (the caps act on seen tokens after dedup + gates):

| lang | source | topic | < 1900 | 1900-19 | 1920-29 | 1930-1939-06 | state |
|---|---|---|---|---|---|---|---|
| EN | American Stories | US newspapers, article level; backbone and sole source of scored sets | - | 21,748 | 4,584 | 1,555 | ingested |
| EN | Chronicling America pages | US newspapers, page OCR, batches after the AS snapshot | - | - | - | 1,217 | ingested, complete |
| EN | Congressional Record | US Congress debates | 174 | 181 | 91 | 89 | ingested |
| EN | LoC PD books | books | - | 2,759 | 593 | 19 | ingested |
| EN | pre-1929 books | books | - | 4,171 | 958 | - | ingested |
| EN | Caselaw Access Project | US case law (legal + FR <= 10 %) | 2 | 2 | 257 | 468 | ingested, complete |
| EN | Federal Register 1936-39 | US regulation | - | - | - | 14.5 | ingested |
| | **EN general total** | | 176 | 28,861 | 6,483 | 3,362 | |
| DE | DDB (Deutsches Zeitungsportal) | German newspapers, page OCR | 10,117 | 1,380 | 612 | 639 | ingested |
| DE | Europeana | German/Austrian newspapers | - | 3,111 | 1,103 | 449 | ingested |
| DE | Voelkischer Beobachter 1925, 1930 | NSDAP daily (named source) | - | - | 3.6 | 8.4 | ingested |
| | **DE general total** | | 10,117 | 4,491 | 1,719 | 1,097 | |

Science bucket (any year, <= 10 % of each language's seen tokens):

| lang | source | topic | words | state |
|---|---|---|---|---|
| EN | Science books (title keywords, from the two book sets) | science/technology books | 531M | selected |
| EN | JSTOR EJC science titles (69) | journals, pre-1923 | 306M | ingested |
| EN | Royal Society Corpus | Phil. Trans. 1665-1920 | 78.6M | ingested |
| EN | Gutenberg science (997 of 2,983 books) | science/maths books, keyed | 58.5M | ingested (5080) |
| EN | Nature <= 1930 / PSM 1872-1915 / EB11 OCR vols | journal / popular science / encyclopaedia | 49.9M / 31.7M / 21.9M | ingested |
| EN | USGS Prof. Papers / BAMS (370 issues) / PNAS | geology / mathematics / general science | 11.4M / 6.0M / 5.1M | ingested |
| EN | Sci. Am., PSM 1916-30, J. Franklin Inst., Americana, MWR, PHR, Phys. Rev., NACA, BSTJ, NBS x2, rest of BAMS | journals, encyclopaedia | ~280-300M (1.9 GB) | downloading (2080) |
| EN | Gutenberg science, other 1,986 books | books | ~115M | downloading (2080) |
| | **EN science** | | ~1.10B on hand, ~1.5B with downloads | |
| DE | Annalen d. Physik (66 vols on the 5080) / Sitzungsberichte | physics / academy proceedings | 3.8M / 0.2M (+97 vols ~150 MB on the 2080) | ingested / downloaded |
| DE | Annalen d. Physik rest, Meyers 6th ed., Naturwissenschaften, Math. Annalen, Phys. Zeitschrift, Crelle, Encyklopaedie d. math. Wiss. | physics, encyclopaedia, mathematics | ~400M (3.0 GB) | downloading (2080) |
| DE | JFM (vol <= 61 at ingest) / Dingler / Gutenberg de (150) | maths reviews / engineering journal (TEI) / books | ~60-70M tok / ~130-150M tok / ~9M | downloading (2080) |
| | **DE science** | | ~0.6-0.65B with downloads | |

Not training: embargo 1939-07-01..08-31 (RQ3 conditioning only) AS 23.0M, DDB 6.9M, Europeana 4.7M;
from 1939-09 (Test-B/C) AS 2,078M, DDB 128M, Europeana 5.5M; excluded HMD 2,387M (UK 1800-96),
JSTOR EJC non-science ~0.8B, NCSE; lexicon instruments DTA, ECCO/Evans TCP, SCOWL.

Reading against the mixture: the 1930-1939-06 pool (EN 3.36B + DE 1.10B words) seen twice is already
about 9B words, roughly the 10B-token budget before dedup and gates, so dedup of wire reprints and
the caps (German <= 25 % per period, legal <= 10 %) decide how much of the 1920s is used. German is
~25 % of the 1930s pool; legal is ~14 % of the English 1930s and will be capped. English science is
already above its cap (~0.75B tokens at 10B seen), German science is not.

German 1939 coverage (RQ3 contexts): DDB / Europeana pages June 1939 1,799 / 680; 1939-08-18..31
506 / 305 (19 titles). American Stories June 1939 90,532 articles, 08-18..31 39,258.

## VPN (5080)

- Use since 2026-10-06T05:45Z: 10.12 GB at 11:52 EDT (`vpn_usage.py`, a floor). No VPN downloads planned.
- Client: **Forest** (`C:\Program Files\Forest\forest.exe`, GUI in Andrew's interactive session),
  which runs `resources\forest-core.exe` (Clash-style core; 7890 HTTP, 7891 SOCKS, 1053 DNS, 64999
  local control port). Not started at boot. Dropped three times on 2026-10-06 (core process gone).
- Restart over SSH, options (each needs Andrew's OK): (a) a scheduled task in his interactive session
  (`/IT`) that relaunches `forest.exe`, run with `schtasks /run`, if Forest connects on launch;
  (b) the core's control port (needs the secret in Forest's config; not read); (c) `forest-core.exe`
  from WMI in session 0 (conflicts with the GUI; not recommended).

## Prompt for a new session (paste as the first message)

    你在接手 APS360 课程项目「The Shape of a Knowledge Boundary」（仓库 C:\Users\27409\Desktop\1939，分支 data/corpus-v1）。
    先按顺序读：CLAUDE.md；docs/HANDOFF.md 的第 13 节（新对话操作指南：读什么、两台机器、怎么控制 5080、VPN 流量预算、工具、踩过的坑）
    和第 12 节 2026-10-05 以后的决定；reports/science_status_2026-10-06.md（当前状态和下一步命令）；
    C:\Users\27409\Desktop\APS360 Model\remote-gpu.md（5080 远程规则）。读完用中文给我一段简短的现状确认，然后按
    science_status 文件里「Do next」的顺序继续：检查 2080 上的下载线，下载完成的源在 2080 上 ingest，全部完成后打包交接。
    规则：所有决定以 HANDOFF 为准，不确定就问我；
    关机、改防火墙/SSH/Tailscale、删除远程数据之前先问我；
