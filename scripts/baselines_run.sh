#!/usr/bin/env bash
# Recurrent baselines on the 5080 (decisions 2026-10-19, TASKS Phase 4). Start through WMI so it outlives SSH:
#   (echo '$Script = "baselines_run.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
# 1. LR probes: RNN, LSTM, GRU x lr {5e-4, 1e-3, 2e-3}, 200M tokens each (warmup 100 steps).
# 2. Per model the probe lr with the lowest final Val (English) bpb; the full 12B-token run, then the per-year
#    bpb evaluation. Order LSTM, GRU, RNN. Safe to restart: finished probes are skipped, a full run with
#    checkpoints is resumed, a finished one is skipped.
set -u
cd /home/an/1939
PY=/home/an/miniconda3/envs/torch-gpu/bin/python
mkdir -p logs
LOG=logs/baselines_$(date +%Y%m%d_%H%M%S).log
exec >> "$LOG" 2>&1
echo "== start $(date -Is) on $(git rev-parse --short HEAD)"

finished() {  # run name -> 0 if its log reached the last step
  $PY - "$1" <<'PYEOF'
import json, sys
from pathlib import Path
p = Path(f"runs/{sys.argv[1]}/log.jsonl")
if not p.exists():
    sys.exit(1)
import yaml
run = yaml.safe_load(open(f"runs/{sys.argv[1]}/config.yaml"))
last = int(run["tokens"] // run["batch_tokens"])
sys.exit(0 if any(json.loads(l).get("step") == last and "checkpoint" in l for l in p.open()) else 1)
PYEOF
}

train() {  # name, then new-run arguments
  local name=$1; shift
  if finished "$name"; then echo "-- $name finished"; return 0; fi
  if [ -f "runs/$name/checkpoints/latest.txt" ]; then
    echo "-- $name resume $(date -Is)"; $PY -m src.train.train --name "$name" --resume
  else
    echo "-- $name new $(date -Is)"; $PY -m src.train.train --name "$name" "$@"
  fi
}

for m in lstm_50m gru_50m rnn_50m; do
  for lr in 5e-4 1e-3 2e-3; do
    train "probe_${m}_lr${lr}" --model "$m" --lr "$lr" --tokens 200e6 --warmup-steps 100 || echo "!! probe ${m} ${lr} failed"
  done
done

for m in lstm_50m gru_50m rnn_50m; do
  best=$($PY - "$m" <<'PYEOF'
import json, sys
from pathlib import Path
m, res = sys.argv[1], {}
for p in Path("runs").glob(f"probe_{m}_lr*/log.jsonl"):
    vals = [json.loads(l) for l in p.open() if "val_en_bpb" in l]
    if vals:
        res[p.parent.name.split("_lr")[1]] = vals[-1]["val_en_bpb"]
print(min(res, key=res.get) if res else "")
json.dump(res, open(f"runs/probe_{m}_choice.json", "w"))
PYEOF
)
  if [ -z "$best" ]; then echo "!! no finished probe for $m"; continue; fi
  echo "-- $m: lr $best ($(cat runs/probe_${m}_choice.json))"
  train "$m" --model "$m" --lr "$best" || { echo "!! $m failed $(date -Is)"; continue; }
  [ -f "reports/$m/bpb_by_year.json" ] || $PY -m src.eval.bpb_by_year --run "$m"
done
echo "== done $(date -Is)"
