#!/usr/bin/env bash
# Ingest the given sources one after another (incremental: only raw files not yet in the ingested MANIFEST).
# Start via WMI so it outlives the SSH session:
#   (echo '$Script = "ingest_run.sh"'; echo '$DlArgs = "chronicling_america caselaw_access_project"'; cat scripts/start_download.ps1) | ssh gpu ...
set -uo pipefail
cd "$HOME/1939"
PY="$HOME/miniconda3/envs/torch-gpu/bin/python"
for s in "$@"; do
  echo "$(date -Is) ingest $s" >> logs/ingest_run.log
  "$PY" -u -m src.data.ingest_extra --source "$s" --workers "${WORKERS:-4}" >> "logs/ingest_$s.log" 2>&1
  rc=$?; echo "$(date -Is) $s exit $rc" >> logs/ingest_run.log
done
