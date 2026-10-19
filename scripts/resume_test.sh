#!/usr/bin/env bash
# Resume test on the 5080 (CLAUDE.md: required before any run longer than 8 h; TASKS Phase 4).
# A short run of the real model on the real stream is killed with SIGKILL after it has passed a checkpoint,
# then resumed; the losses both processes logged for the same steps must agree (bf16 / cuDNN allow ~1e-3).
#   bash scripts/resume_test.sh <model> <micro_batch_rows> <checkpoint_every> <kill_at_step> <lr>
#   bash scripts/resume_test.sh lstm_50m 64 100 150 1e-3
#   bash scripts/resume_test.sh s335m 1 20 30 3e-4
set -u
cd /home/an/1939
PY=/home/an/miniconda3/envs/torch-gpu/bin/python
MODEL=$1; MICRO=$2; EVERY=$3; KILL_AT=$4; LR=$5
NAME=resume_test_$MODEL
rm -rf "runs/$NAME"
mkdir -p logs
TOKENS=$(( (KILL_AT + 2 * EVERY) * 262144 ))
$PY -m src.train.train --name "$NAME" --model "$MODEL" --lr "$LR" --micro-batch-rows "$MICRO" --tokens "$TOKENS" \
  --checkpoint-every "$EVERY" --eval-every 1000000 > "logs/$NAME.first.log" 2>&1 &
PID=$!
step() { tail -1 "runs/$NAME/log.jsonl" 2>/dev/null | $PY -c 'import sys, json; t = sys.stdin.read().strip(); print(json.loads(t).get("step", 0) if t else 0)'; }
while kill -0 "$PID" 2>/dev/null && [ "$(step)" -lt "$KILL_AT" ]; do sleep 5; done
kill -9 "$PID" 2>/dev/null
sleep 3
cp "runs/$NAME/log.jsonl" "runs/$NAME/log_before_kill.jsonl"
$PY -m src.train.train --name "$NAME" --resume > "logs/$NAME.resume.log" 2>&1
$PY - "$NAME" <<'PYEOF'
import json, sys
name = sys.argv[1]
before = [json.loads(l) for l in open(f"runs/{name}/log_before_kill.jsonl")]
after = [json.loads(l) for l in open(f"runs/{name}/log.jsonl")][len(before):]
b = {r["step"]: r["loss"] for r in before if "loss" in r}
a = {r["step"]: r["loss"] for r in after if "loss" in r}
both = sorted(set(a) & set(b))
diff = max((abs(a[s] - b[s]) / b[s] for s in both), default=None)
ok = bool(both) and diff < 1e-3 and after and after[-1].get("checkpoint")
res = {"run": name, "killed_after_step": max(b), "resumed_steps_compared": both, "max_relative_loss_difference": diff,
       "finished_step": after[-1]["step"] if after else None, "passed": bool(ok)}
json.dump(res, open(f"runs/{name}/resume_test.json", "w"), indent=1)
print(("PASS " if ok else "FAIL ") + json.dumps(res))
PYEOF
