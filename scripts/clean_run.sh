#!/usr/bin/env bash
# Cleaning chain on the 5080: selection census and run, then MinHash signing for both language pools
# (signatures are cached per selected file, so a later run only signs files added since).
# Start via WMI so it outlives the SSH session:
#   (echo '$Script = "clean_run.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
# Optional argument: steps to run (default "census select sign_en sign_de"); later steps: dedup_en dedup_de
# heldout_en heldout_de reprint_en reprint_de filter_en filter_de audit.
# Worker counts keep the 9800X3D under ~80 C: 14 filter workers ran it at 88 C / 154 W (2026-10-08).
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
    sign_en) "$PY" -u -m src.data.dedup --lang en --workers 8 --sign-only >> logs/dedup_en.log 2>&1 ;;
    sign_de) "$PY" -u -m src.data.dedup --lang de --workers 8 --sign-only >> logs/dedup_de.log 2>&1 ;;
    dedup_en) "$PY" -u -m src.data.dedup --lang en --workers 8 --samples reports/dedup_samples_en.md >> logs/dedup_en.log 2>&1 ;;
    dedup_de) "$PY" -u -m src.data.dedup --lang de --workers 8 --samples reports/dedup_samples_de.md >> logs/dedup_de.log 2>&1 ;;
    heldout|heldout_en) "$PY" -u -m src.data.para_dedup heldout --lang en --workers 8 >> logs/para_dedup.log 2>&1 ;;
    heldout_de) "$PY" -u -m src.data.para_dedup heldout --lang de --workers 8 >> logs/para_dedup.log 2>&1 ;;
    reprint_en) "$PY" -u -m src.data.para_dedup reprint --lang en --workers 6 >> logs/para_dedup.log 2>&1 ;;
    reprint_de) "$PY" -u -m src.data.para_dedup reprint --lang de --workers 6 >> logs/para_dedup.log 2>&1 ;;
    filter_en) "$PY" -u -m src.data.filter --lang en --workers 8 >> logs/filter.log 2>&1 ;;
    filter_de) "$PY" -u -m src.data.filter --lang de --workers 8 >> logs/filter.log 2>&1 ;;
    audit)    "$PY" -u -m src.data.audit --out reports/audit_v1.md >> logs/audit.log 2>&1 ;;
    *) echo "unknown step $step" >> "$log"; false ;;
  esac
  rc=$?; echo "$(date -Is) $step exit $rc" >> "$log"
  [ "$rc" = 0 ] || break
done
echo "$(date -Is) clean_run done" >> "$log"
