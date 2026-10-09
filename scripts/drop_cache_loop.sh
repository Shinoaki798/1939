#!/usr/bin/env bash
# Drop the WSL page cache every N seconds while a job runs, so the VM hands memory back to Windows instead
# of making it page to the SSD (2026-10-08/09: a full cache left Windows ~1 GB). Runs as root, via WMI:
#   wsl.exe -d Ubuntu -u root -- bash /home/an/1939/scripts/drop_cache_loop.sh 'src[.]data[.]tokenize_corpus' 120
# Write the pattern with [.] so it does not match this loop's own command line.
pattern="${1:?process pattern}"
every="${2:-120}"
while pgrep -f "$pattern" > /dev/null; do
  sync
  echo 1 > /proc/sys/vm/drop_caches
  sleep "$every"
done
