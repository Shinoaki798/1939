#!/usr/bin/env bash
# Tokenise the filtered corpus on the 5080 (resumable). Start via WMI so it outlives the SSH session:
#   (echo '$Script = "tokenize_run.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
set -uo pipefail
cd "$HOME/1939"
mkdir -p logs
PY="$HOME/miniconda3/envs/torch-gpu/bin/python"
log=logs/tokenize.log
echo "$(date -Is) tokenize start" >> "$log"
"$PY" -u -m src.data.tokenize_corpus --workers 8 --threads 2 "$@" >> "$log" 2>&1
rc=$?; echo "$(date -Is) tokenize exit $rc" >> "$log"
