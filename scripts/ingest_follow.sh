#!/usr/bin/env bash
# Ingest American Stories years as their downloads get verified. Pure Linux (no WSL
# interop), so a detached tmux on the remote box is enough:
#   tmux new -d -s ingest "bash ~/1939/scripts/ingest_follow.sh"
# src.data.ingest skips years already in its MANIFEST, so each pass only does new years.
# Exits once every year in YEARS is ingested.
set -uo pipefail
cd "$HOME/1939"
YEARS="${YEARS:-1900-1955}"
PY="$HOME/miniconda3/envs/torch-gpu/bin/python"
LOG=logs/ingest_american_stories.log
mkdir -p logs
want=$("$PY" -c "from src.data.download import parse_years; print(len(parse_years('$YEARS')))")
while true; do
  "$PY" -u -m src.data.ingest --years "$YEARS" --workers 6 >> "$LOG" 2>&1
  have=$("$PY" -c "import json; print(len(json.load(open('data/ingested/american_stories/MANIFEST.json'))['years']))" 2>/dev/null || echo 0)
  echo "$(date -Is) ingested $have/$want years" >> "$LOG"
  [ "$have" -ge "$want" ] && break
  sleep 600
done
