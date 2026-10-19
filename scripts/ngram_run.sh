#!/usr/bin/env bash
# Kneser-Ney 5-gram on the 5080 (CPU; runs beside the GPU baselines), then its per-year bpb.
#   (echo '$Script = "ngram_run.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
set -u
cd /home/an/1939
PY=/home/an/miniconda3/envs/torch-gpu/bin/python
mkdir -p logs
exec >> "logs/ngram_$(date +%Y%m%d_%H%M%S).log" 2>&1
echo "== start $(date -Is) on $(git rev-parse --short HEAD)"
[ -f runs/ngram_kn5/model.binary ] || $PY -m src.model.ngram --name ngram_kn5 --memory 40%
[ -f reports/ngram_kn5/bpb_by_year.json ] || $PY -m src.eval.bpb_by_year --kenlm ngram_kn5
echo "== done $(date -Is)"
