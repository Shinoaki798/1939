# Data collection: status and resume steps (2026-10-06, ~12:30 EDT)

Operating guide (machines, remote control, VPN budget, tools, pitfalls): `docs/HANDOFF.md` §13.

## Done since ~12:00

- HANDOFF §13 and the previous version of this file committed (`fd4b23b`).
- 2080 hand-over merged on the 5080 (`1939_transfer.rar` unpacked with Windows `tar` next to itself,
  then `transfer_ingested.py merge`, every file sha256-verified): Chronicling America is complete
  (124 / 124 batches, 1.22B words), PSM Wikisource ingested (31.7M words).
- Catalog tables the box regenerated copied back and committed (`fb662b6`).
- JFM had stopped at 05:02 (cursor 52,100 / 223,270): zbMATH lost the resumption token and answers it
  with HTTP 500, which the harvester did not treat as expiry. Fixed (`82e3715`: a token that fails 3x
  restarts the harvest once from page 1, saved pages are skipped) and relaunched.
- Chain scripts logged `exit 0` for everything (`$?` read after `$(date)`); fixed (`52145cd`).
- Ingested (`ingest_run.sh`): rest of CAP (now 3,464 volumes, 729M words), USGS PP, PNAS, Nature, EB11 OCR.

## Do next, in this order

1. **Wait for the downloads** (table below); restart any chain that dies (commands below).
2. **Ingest the rest of the science sources** as they finish (incremental, skips ingested files):

       (echo '$Script = "ingest_run.sh"'; echo '$DlArgs = "<sources>"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'

3. Then OCR gates (keyed sources skip them), MinHash dedup, splits, C1 screen, and the audit
   (`reports/audit_v1.md`, due 2026-10-10; science bucket by language x decade x source, licence
   table, keyed vs OCR share). `dedup.py`, `splits.py` and the tokenizer do not exist yet.
4. **VPN restart over SSH**: client identified (below); waiting for Andrew's choice.
5. On Andrew's OK only: delete `C:\Users\AN\Downloads\1939_transfer.rar` and `1939_transfer\` on the box
   (3.3 GB each; everything is merged and verified).

## Running on the 5080 (checked 12:30 EDT)

| job | route | progress | rate | expected end |
|---|---|---|---|---|
| `science_ia.sh` (English archive.org: BAMS, then NBS x2, BSTJ, NACA, Phys. Rev., PHR, MWR, Americana, JFI, PSM 1916-30, Sci. Am.) | proxy | BAMS 347 / 516; ~10,200 files left | ~7 s / file | 2026-10-07 evening |
| `science_ia_more.sh` via `science_ia_now.sh` (German archive.org: Annalen d. Physik, Naturwiss., Phys. Z., Math. Ann., Crelle, Encyklopaedie, Meyers 6, Sitzungsber.) | proxy | Annalen 66 / 2,252; ~5,100 files left | ~20 s / file (many HTTP 500 retries) | 2026-10-07 night / 10-08 morning |
| `science_misc.sh` (Gutenberg en 2,983, de 150, then Dingler tarball) | proxy | Gutenberg en 961 / 2,983 | ~6 s / file | today ~17:00 |
| `jfm_run.sh` (zbMATH OAI) | proxy, falls back to direct | restarted from page 1 at 12:10 (521 of ~2,233 pages already saved) | ~12 s / page | today ~20:00 |
| CAP, USGS PP, PNAS, Nature, EB11 OCR | - | downloaded and ingested | - | done |

When `science_ia.sh` reaches the German sources, the per-source download lock makes it skip any source
`science_ia_more.sh` is still fetching.

Relaunch commands (verified files are skipped):

    (echo '$Script = "science_ia.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
    (echo '$Script = "science_ia_now.sh"'; echo '$DlArgs = "annalen_physik_ia naturwiss_ia physz_ia math_annalen_ia crelle_ia encyklopaedie_ia meyers6_ia sitzungsberichte_ia"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
    (echo '$Script = "science_misc.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
    (echo '$Script = "jfm_run.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'

## VPN

- Use since 2026-10-06T05:45Z: 10.12 GB at 11:52 EDT (`vpn_usage.py`, a floor; JFM pages are stored
  gzipped, so its real transfer is ~10x its listed bytes, ~0.7 GB for the whole harvest).
- Still expected: ~10-12 GB (English IA ~5 GB, German IA ~4 GB, Gutenberg ~1.2 GB, JFM ~0.6 GB, Dingler).
- Client on the box: **Forest** (`C:\Program Files\Forest\forest.exe`, GUI in Andrew's interactive
  session since 2026-10-01), which runs `resources\forest-core.exe` (Clash-style core, `-d`/`-f` flags;
  listens on 7890 HTTP, 7891 SOCKS, 1053 DNS, 64999 local control port). The core was restarted at
  11:28:41 (the VPN coming back). Forest is not in Run keys, Startup folders or services, so it does
  not start at boot.
- Restart over SSH, options (each needs Andrew's OK): (a) a scheduled task that runs in his interactive
  session (`/IT`) and relaunches `forest.exe`, triggered with `schtasks /run`; works only if Forest
  connects on launch (a client setting); (b) the core's local control port (needs the secret from
  Forest's config; not read); (c) starting `forest-core.exe` from WMI in session 0 (conflicts with
  the GUI; not recommended).

## Data on hand (raw words before dedup / OCR gates)

| bucket | source | state | volume |
|---|---|---|---|
| EN general | American Stories (backbone, all scored sets) | ingested | 27.96B <= 1939 (1900s 10.41B, 1910s 11.34B, 1920s 4.58B, 1930-39 1.62B); 2.03B 1940-55 for holdouts |
| EN general | Chronicling America 1930-1939.06 pages | ingested, complete | 1.22B (124 batches) |
| EN general | Congressional Record | ingested | 0.53B |
| EN general | LoC PD books / pre-1929 books | ingested | 3.37B / 5.13B (science selection moves out, below) |
| EN general | Caselaw Access Project (<= 1939-06-30) | ingested, complete | 0.73B |
| EN general | Federal Register 1936-39 | ingested | 14.5M |
| DE general | DDB / Europeana / Voelkischer Beobachter | ingested | 12.88B / 4.67B / 12M |
| EN science | JSTOR EJC science titles / RSC | ingested | 306M / 78.6M |
| EN science | Science books (title keywords) | selected | 531M (5,742 books) |
| EN science | Nature <= 1930 / PSM 1872-1915 / EB11 OCR vols / USGS PP / PNAS | ingested | 49.9M / 31.7M / 21.9M / 11.4M / 5.1M |
| EN science | archive.org journals (above), Gutenberg Q*/T* + EB11 keyed | downloading | est. 250-400M; 40-70M |
| DE science | JFM (vol <= 61), Dingler, archive.org journals/encyclopaedias, Gutenberg de | downloading | est. JFM 60-70M tokens, Dingler 130-150M tokens, journals several 100M |
| out | HMD (2.39B), JSTOR EJC non-science titles | ingested, excluded by decision | - |
| instruments | DTA, ECCO/Evans TCP, SCOWL | kept, never training | - |

English science is already ~1.04B words ingested or selected, above its 10 % cap at 10B seen tokens
(~0.75B EN tokens); the remaining English downloads add coverage, not seen tokens.

German 1939 coverage (RQ3 contexts): DDB / Europeana pages June 1939 1,799 / 680; 1939-08-18..31
506 / 305 (19 titles). American Stories June 1939 90,532 articles, 08-18..31 39,258.

## Prompt for a new session (paste as the first message)

    你在接手 APS360 课程项目「The Shape of a Knowledge Boundary」（仓库 C:\Users\27409\Desktop\1939，分支 data/corpus-v1）。
    先按顺序读：CLAUDE.md；docs/HANDOFF.md 的第 13 节（新对话操作指南：读什么、两台机器、怎么控制 5080、VPN 流量预算、工具、踩过的坑）
    和第 12 节 2026-10-05 以后的决定；reports/science_status_2026-10-06.md（当前状态和下一步命令）；
    C:\Users\27409\Desktop\APS360 Model\remote-gpu.md（5080 远程规则）。读完用中文给我一段简短的现状确认，然后按
    science_status 文件里「Do next」的顺序继续：检查各下载线和 VPN 用量，下载完成的源先 ingest。
    规则：所有决定以 HANDOFF 为准，不确定就问我；
    关机、改防火墙/SSH/Tailscale、删除远程数据之前先问我；
