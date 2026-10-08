#!/usr/bin/env bash
# Tokenizer comparison on the 5080 (user, 2026-10-08): sample -> train (one 64k run, 32k/48k as prefixes)
# -> compare. Start via WMI so it outlives the SSH session:
#   (echo '$Script = "tokenizer_run.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
# Optional argument: steps (default "sample train compare").
set -uo pipefail
cd "$HOME/1939"
mkdir -p logs
PY="$HOME/miniconda3/envs/torch-gpu/bin/python"
log=logs/tokenizer.log
steps="${*:-sample train compare}"
echo "$(date -Is) tokenizer_run start: $steps" >> "$log"
for step in $steps; do
  echo "$(date -Is) $step" >> "$log"
  "$PY" -u -m src.data.tokenizer "$step" >> "$log" 2>&1
  rc=$?; echo "$(date -Is) $step exit $rc" >> "$log"
  [ "$rc" = 0 ] || break
done
echo "$(date -Is) tokenizer_run done" >> "$log"
