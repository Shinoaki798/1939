#!/usr/bin/env bash
# Main Transformer run on the 5080 (decisions 2026-10-19, TASKS Phase 4). Start through WMI so it outlives SSH:
#   (echo '$Script = "main_run.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
# s335m, lr 3e-4, 1 x 4,096-token micro-batches (64-step accumulation), 45,776 steps of 2^18 tokens on
# data/shards/main, then the per-year bpb. Safe to restart (after a power loss): a run with checkpoints is
# resumed, a finished one is skipped.
set -u
cd /home/an/1939
PY=/home/an/miniconda3/envs/torch-gpu/bin/python
NAME=main_s335m
mkdir -p logs
LOG=logs/main_$(date +%Y%m%d_%H%M%S).log
exec >> "$LOG" 2>&1
echo "== start $(date -Is) on $(git rev-parse --short HEAD)"

finished() {  # 0 if the run's log reached the last step with a checkpoint
  $PY - "$NAME" <<'PYEOF'
import json, sys
from pathlib import Path
import yaml
p = Path(f"runs/{sys.argv[1]}/log.jsonl")
if not p.exists():
    sys.exit(1)
run = yaml.safe_load(open(f"runs/{sys.argv[1]}/config.yaml"))
last = int(run["tokens"] // run["batch_tokens"])
sys.exit(0 if any(json.loads(l).get("step") == last and "checkpoint" in l for l in p.open()) else 1)
PYEOF
}

if finished; then
  echo "-- $NAME finished"
elif [ -f "runs/$NAME/checkpoints/latest.txt" ]; then
  echo "-- $NAME resume $(date -Is)"; $PY -m src.train.train --name "$NAME" --resume || { echo "!! $NAME failed $(date -Is)"; exit 1; }
else
  echo "-- $NAME new $(date -Is)"
  $PY -m src.train.train --name "$NAME" --model s335m --lr 3e-4 --micro-batch-rows 1 --stream main \
    || { echo "!! $NAME failed $(date -Is)"; exit 1; }
fi
[ -f "reports/$NAME/bpb_by_year.json" ] || $PY -m src.eval.bpb_by_year --run "$NAME"
echo "== done $(date -Is)"
