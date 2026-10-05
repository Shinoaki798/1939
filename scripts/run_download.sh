#!/usr/bin/env bash
# Foreground runner for src.data.download on the remote box. Do not start it from an SSH
# session directly: Windows kills every process of an SSH session (including the
# curl.exe the downloader launches through WSL interop) when the session closes, and a
# detached tmux loses WSL interop once its launching wsl.exe exits. Start it with
#   ssh gpu 'powershell -NoProfile -Command -' < scripts/start_download.ps1
# which creates wsl.exe via WMI, outside the SSH session, and keeps it alive for the run.
# Arguments are passed to src.data.download (e.g. --source hmd_newspapers). A per-source
# lock inside the downloader prevents two runs of the same source.
set -uo pipefail
cd "$HOME/1939"
mkdir -p logs
src=american_stories
prev=""
for a in "$@"; do
  [ "$prev" = "--source" ] && src="$a"
  prev="$a"
done
log="logs/download_${src}.log"
echo "$(date -Is) start (pid $$) args: $*" >> "$log"
exec "$HOME/miniconda3/envs/torch-gpu/bin/python" -u -m src.data.download "$@" >> "$log" 2>&1
