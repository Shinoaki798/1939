#!/usr/bin/env bash
# JFM OAI harvest (resumes from data/foreign/de/raw/jfm/harvest_state.json), via the Windows proxy.
# Start via WMI:  (echo '$Script = "jfm_run.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
set -uo pipefail
cd "$HOME/1939"
mkdir -p logs
"$HOME/miniconda3/envs/torch-gpu/bin/python" -u -m src.data.jfm_harvest --delay 2 >> logs/jfm_harvest.log 2>&1
rc=$?; echo "$(date -Is) jfm_harvest exit $rc" >> logs/jfm_harvest.log
