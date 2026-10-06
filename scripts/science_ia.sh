#!/usr/bin/env bash
# Science bucket, archive.org sources: build the catalogs, then fetch every source's djvu.txt files one
# source after another (one request at a time to archive.org, 1 s apart, 30 min pause on 429/403).
# Needs WSL interop (curl.exe through the Windows proxy), so start it via WMI:
#   (echo '$Script = "science_ia.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
set -uo pipefail
cd "$HOME/1939"
mkdir -p logs
PY="$HOME/miniconda3/envs/torch-gpu/bin/python"
log=logs/science_ia.log
echo "$(date -Is) science_ia start" >> "$log"
"$PY" -u -m src.data.ia_catalog --all >> "$log" 2>&1
for s in $("$PY" -c "
from src.data.download import load_config, repo_path
c = load_config(repo_path('config/paths.yaml')); s = load_config(repo_path(c['sources']))
print(' '.join(n for n, v in s.items() if v.get('ia_query')))"); do
  echo "$(date -Is) download $s" >> "$log"
  "$PY" -u -m src.data.download --source "$s" --delay 1 --backoff 1800 >> "logs/download_$s.log" 2>&1
  echo "$(date -Is) $s exit $?" >> "$log"
done
echo "$(date -Is) science_ia done" >> "$log"
