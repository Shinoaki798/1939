#!/usr/bin/env bash
# Foreground runner for src.data.download on the remote box. Do not start it from an SSH
# session directly: Windows kills every process of an SSH session (including the
# curl.exe the downloader launches through WSL interop) when the session closes, and a
# detached tmux loses WSL interop once its launching wsl.exe exits. Start it with
#   ssh gpu 'powershell -NoProfile -Command -' < scripts/start_download.ps1
# which creates wsl.exe via WMI, outside the SSH session, and keeps it alive for the run.
set -uo pipefail
cd "$HOME/1939"
mkdir -p logs
if pgrep -f "python -u -m src.data.download" >/dev/null; then
  echo "$(date -Is) downloader already running; not starting another" >> logs/download_american_stories.log
  exit 0
fi
echo "$(date -Is) start (pid $$) args: $*" >> logs/download_american_stories.log
exec "$HOME/miniconda3/envs/torch-gpu/bin/python" -u -m src.data.download "$@" >> logs/download_american_stories.log 2>&1
