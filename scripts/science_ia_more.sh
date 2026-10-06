#!/usr/bin/env bash
# Second archive.org batch (sources added after science_ia.sh started): waits for that chain to finish
# so that archive.org still sees one request at a time, then catalogs and fetches the given sources.
#   (echo '$Script = "science_ia_more.sh"'; echo '$DlArgs = "eb11_ia nature_ia"'; cat scripts/start_download.ps1) | ssh gpu ...
set -uo pipefail
cd "$HOME/1939"
PY="$HOME/miniconda3/envs/torch-gpu/bin/python"
log=logs/science_ia.log
until grep -q "science_ia done" "$log" 2>/dev/null; do sleep 300; done
for s in "$@"; do
  "$PY" -u -m src.data.ia_catalog --source "$s" >> "$log" 2>&1
  echo "$(date -Is) download $s" >> "$log"
  "$PY" -u -m src.data.download --source "$s" --delay 1 --backoff 1800 >> "logs/download_$s.log" 2>&1
  echo "$(date -Is) $s exit $?" >> "$log"
done
echo "$(date -Is) science_ia_more done ($*)" >> "$log"
