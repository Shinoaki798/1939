# Data collection: status and resume steps (2026-10-06, ~12:00 EDT)

Operating guide (machines, remote control, VPN budget, tools, pitfalls): `docs/HANDOFF.md` §13.

## Not yet committed

`docs/HANDOFF.md` (§13 and the top pointer) and this file. Commit both on `data/corpus-v1` and push,
then pull on the 5080: `ssh gpu 'wsl -d Ubuntu -- bash -s -- data/corpus-v1' < scripts/remote_pull.sh`.

## Do next, in this order

1. **Merge the 2080 hand-over** (Andrew already put it on the 5080 at `C:\Users\AN\Downloads\1939_transfer`):

       ssh gpu 'wsl -d Ubuntu -- bash -c "cd /home/an/1939 && ~/miniconda3/envs/torch-gpu/bin/python scripts/transfer_ingested.py merge --in /mnt/c/Users/AN/Downloads/1939_transfer"'

   It verifies every file against TRANSFER.json, then merges 101 Chronicling America batches
   (362,170 pages, 1.08B words) and Popular Science Monthly vols 1-87 (15,173 documents, 31.7M words).
2. **Copy back and commit the catalog tables the box regenerated** when `science_ia.sh` restarted
   (`config/*_ia_files.tsv`, `config/*_ia_items.tsv`; e.g. PHR 996 -> 1,001 items) and
   `config/gutenberg_sci_*` if changed (`git status` on the box shows them as modified).
3. **Check progress and VPN use** (`python3 scripts/vpn_usage.py --since 2026-10-06T05:45:00+00:00`;
   10.02 GB used at 11:36 EDT; ~8 GB more expected). Restart any chain that died (commands below).
4. **VPN restart over SSH** (open item, HANDOFF §13.4): identify the VPN client on the box (process
   name, executable, auto-start/auto-reconnect settings) read-only; propose a way; ask Andrew before
   creating a scheduled task or changing settings.
5. When downloads finish: ingest every science source (`scripts/ingest_run.sh <sources>` via WMI),
   then OCR gates (keyed sources skip them), MinHash dedup, splits, C1 screen, and the audit
   (`reports/audit_v1.md`, due 2026-10-10; science bucket by language x decade x source, licence
   table, keyed vs OCR share).

## Running on the 5080 (relaunched ~11:30 EDT after the VPN came back)

| job | route | last seen |
|---|---|---|
| `science_ia.sh` (all archive.org sources, English first; rebuilds catalogs, skips done files) | proxy | catalog rebuilt; continuing at Bull. AMS (164 / 516) |
| `science_ia_now.sh` (German archive.org sources) | proxy | Annalen der Physik started |
| `science_misc.sh` (Gutenberg en 2,983 + de 150, then Dingler tarball) | proxy | Gutenberg en 782 done |
| `jfm_run.sh` (zbMATH OAI, resumes from token) | proxy, falls back to direct | ~23 % at 04:38 |
| CAP download | direct (`--via mirror`) | ~2,900 / 3,464 at 04:38 |
| USGS Professional Papers | direct | ~150 / 186 at 04:38 |

Relaunch commands (verified files are skipped):

    (echo '$Script = "science_ia.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
    (echo '$Script = "science_ia_now.sh"'; echo '$DlArgs = "annalen_physik_ia naturwiss_ia physz_ia math_annalen_ia crelle_ia encyklopaedie_ia meyers6_ia sitzungsberichte_ia"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
    (echo '$Script = "science_misc.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
    (echo '$Script = "jfm_run.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
    (echo '$DlArgs = "--source caselaw_access_project --delay 5 --via mirror"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
    (echo '$Script = "science_usgs.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'

## Data on hand (raw words before dedup / OCR gates)

| bucket | source | state | volume |
|---|---|---|---|
| EN general | American Stories (backbone, all scored sets) | ingested | 27.9B all years; 1930-1939.06 1.48B; 1920s 4.34B |
| EN general | Congressional Record | ingested | 0.53B |
| EN general | LoC PD books / pre-1929 books | ingested | 3.37B / 5.13B (science selection moves out, below) |
| EN general | Federal Register 1936-39 | ingested | 14.5M |
| EN general | Caselaw Access Project (<= 1939-06-30) | partly ingested | 512M so far |
| EN general | Chronicling America 1930-1939.06 pages | 23 batches ingested on the box; 101 on the 2080, merge pending | 109M + 1.08B |
| DE general | DDB / Europeana / Voelkischer Beobachter | ingested | 12.88B / 4.66B / 12M |
| EN science | JSTOR EJC science titles / RSC | ingested | 306M / 78.6M |
| EN science | Science books (title keywords) | selected | 531M (5,742 books) |
| EN science | PSM 1872-1915 (Wikisource) | ingested on the 2080, merge pending | 31.7M |
| EN science | archive.org: PNAS, EB11 OCR vols, Nature done; Bull. AMS, NBS, BSTJ, NACA, Phys. Rev., PHR, MWR, Americana, JFI, PSM 1916-30, Sci. Am. in progress | downloading | est. 300-500M |
| EN science | Gutenberg Q*/T* + EB11 keyed; USGS PP | downloading | est. 100-200M; 5-10M |
| DE science | JFM (vol <= 61), Dingler, archive.org journals/encyclopaedias, Gutenberg de | downloading | est. JFM 60-70M tokens, Dingler 130-150M tokens, journals several 100M |
| instruments | DTA, ECCO/Evans TCP, SCOWL | kept, never training | - |

## Prompt for a new session (paste as the first message)

    你在接手 APS360 课程项目「The Shape of a Knowledge Boundary」（仓库 C:\Users\27409\Desktop\1939，分支 data/corpus-v1）。
    先按顺序读：CLAUDE.md；docs/HANDOFF.md 的第 13 节（新对话操作指南：读什么、两台机器、怎么控制 5080、VPN 流量预算、工具、踩过的坑）
    和第 12 节 2026-10-05 以后的决定；reports/science_status_2026-10-06.md（当前状态和下一步命令）；
    C:\Users\27409\Desktop\APS360 Model\remote-gpu.md（5080 远程规则）。读完用中文给我一段简短的现状确认，然后按
    science_status 文件里「Do next」的顺序继续：先把 C:\Users\AN\Downloads\1939_transfer 合并进 5080 的数据集，
    再把 5080 重新生成的目录表拷回提交，检查各下载线和 VPN 用量。
    规则：所有决定以 HANDOFF 为准，不确定就问我；
    关机、改防火墙/SSH/Tailscale、删除远程数据之前先问我；