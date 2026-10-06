#!/usr/bin/env bash
# Resume Chronicling America after LoC's rate limit (HTTP 429 + Cloudflare challenge, 2026-10-05).
# 1. wait START_DELAY seconds (default 2 h) so the block can expire;
# 2. survey the remaining post-snapshot batches politely (src.data.ca_survey: 15 s apart, 30 min pause on 429/403);
# 3. fetch the batches already listed in config/chronicling_america_files.tsv, 30 s apart, same backoff.
# Needs WSL interop (curl.exe through the Windows proxy), so start it via WMI:
#   (echo '$Script = "ca_resume.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
set -uo pipefail
cd "$HOME/1939"
mkdir -p logs
PY="$HOME/miniconda3/envs/torch-gpu/bin/python"
log=logs/ca_resume.log
echo "$(date -Is) ca_resume start (start delay ${START_DELAY:-7200}s)" >> "$log"
"$PY" -u -m src.data.ca_survey --inventory data/raw/chronicling_america/inventory/ocr.json \
      --start-delay "${START_DELAY:-7200}" >> "$log" 2>&1
rc=$?; echo "$(date -Is) survey finished (exit $rc); downloading listed batches" >> "$log"
"$PY" -u -m src.data.download --source chronicling_america --delay 30 --backoff 1800 >> "$log" 2>&1
rc=$?; echo "$(date -Is) ca_resume done (exit $rc)" >> "$log"
