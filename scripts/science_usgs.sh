#!/usr/bin/env bash
# USGS Professional Papers: catalog (API) then PDFs, both direct (pubs.usgs.gov needs no VPN).
set -uo pipefail
cd "$HOME/1939"
PY="$HOME/miniconda3/envs/torch-gpu/bin/python"
"$PY" -u -m src.data.usgs_catalog --source usgs_pp >> logs/science_usgs.log 2>&1
"$PY" -u -m src.data.download --source usgs_pp --delay 2 --via mirror >> logs/download_usgs_pp.log 2>&1
rc=$?; echo "$(date -Is) usgs done $rc" >> logs/science_usgs.log
