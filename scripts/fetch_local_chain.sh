#!/usr/bin/env bash
# Run scripts/fetch_local.py on the LOCAL PC (the 2080) for several sources, one after another, each with
# its own key list data/logs/local_keys/<source>.txt (keys the 5080 has not verified, reverse file order).
# Start detached from Git Bash so it outlives the Claude session, e.g.
#   powershell -Command "Start-Process -WindowStyle Hidden 'C:\Program Files\Git\bin\bash.exe' -ArgumentList 'scripts/fetch_local_chain.sh','en','sciam_ia','psm_ia'"
# The first argument names the chain (its log: data/logs/fetch_local_chain_<name>.log); DELAY=<s> overrides the 1 s gap.
set -uo pipefail
cd "$(dirname "$0")/.."
name="$1"; shift
log="data/logs/fetch_local_chain_${name}.log"
echo "$(date -Is) chain $name start: $*" >> "$log"
for s in "$@"; do
  echo "$(date -Is) fetch $s" >> "$log"
  python -u scripts/fetch_local.py --source "$s" --keys-file "data/logs/local_keys/$s.txt" --delay "${DELAY:-1}" --attempts 8 \
    >> "data/logs/fetch_local_$s.log" 2>&1
  rc=$?; echo "$(date -Is) $s exit $rc" >> "$log"
done
echo "$(date -Is) chain $name done" >> "$log"
