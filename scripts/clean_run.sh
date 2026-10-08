#!/usr/bin/env bash
# Cleaning chain on the 5080: selection census and run, then MinHash signing for both language pools
# (signatures are cached per selected file, so a later run only signs files added since).
# Start via WMI so it outlives the SSH session:
#   (echo '$Script = "clean_run.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
# Optional argument: steps to run (default "census select sign_en sign_de").
set -uo pipefail
cd "$HOME/1939"
mkdir -p logs
PY="$HOME/miniconda3/envs/torch-gpu/bin/python"
log=logs/clean_run.log
steps="${*:-census select sign_en sign_de}"
echo "$(date -Is) clean_run start: $steps" >> "$log"
for step in $steps; do
  echo "$(date -Is) $step" >> "$log"
  case "$step" in
    census)  "$PY" -u -m src.data.select census --workers 12 >> logs/select.log 2>&1 ;;
    select)  "$PY" -u -m src.data.select run --workers 8 >> logs/select.log 2>&1 ;;
    sign_en) "$PY" -u -m src.data.dedup --lang en --workers 14 --sign-only >> logs/dedup_en.log 2>&1 ;;
    sign_de) "$PY" -u -m src.data.dedup --lang de --workers 14 --sign-only >> logs/dedup_de.log 2>&1 ;;
    *) echo "unknown step $step" >> "$log"; false ;;
  esac
  rc=$?; echo "$(date -Is) $step exit $rc" >> "$log"
  [ "$rc" = 0 ] || break
done
echo "$(date -Is) clean_run done" >> "$log"
