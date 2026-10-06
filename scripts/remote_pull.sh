#!/usr/bin/env bash
# Update the checkout on the remote 5080 box from GitHub. Run from the local machine:
#   ssh gpu 'wsl -d Ubuntu -- bash -s' < scripts/remote_pull.sh            # branch main
#   ssh gpu 'wsl -d Ubuntu -- bash -s -- data/corpus-v1' < scripts/remote_pull.sh
#
# WSL (NAT mode) cannot reach GitHub or the Windows-side proxy. So Windows git.exe
# fetches GitHub through the proxy into a bare mirror on C:, and WSL git pulls from
# that mirror. Every Windows executable gets </dev/null, otherwise it swallows the
# rest of this script from stdin.
set -euo pipefail

BRANCH="${1:-main}"
GITEXE="/mnt/c/Program Files/Git/cmd/git.exe"
PROXY="http://127.0.0.1:7890"
URL="https://github.com/Shinoaki798/1939"
MIRROR_WSL="/mnt/c/Users/AN/git-mirror/1939.git"
MIRROR_WIN='C:\Users\AN\git-mirror\1939.git'
WORK="$HOME/1939"
SAFE=(-c "safe.directory=$MIRROR_WSL")

if [ ! -d "$MIRROR_WSL" ]; then
  mkdir -p "$(dirname "$MIRROR_WSL")"
  "$GITEXE" -c http.proxy="$PROXY" clone --mirror "$URL" "$MIRROR_WIN" </dev/null
else
  "$GITEXE" -c http.proxy="$PROXY" -C "$MIRROR_WIN" remote update --prune </dev/null
fi

if [ ! -d "$WORK/.git" ]; then
  git "${SAFE[@]}" clone "$MIRROR_WSL" "$WORK"
fi
cd "$WORK"
git "${SAFE[@]}" fetch origin --prune
# Reports regenerated on this box (and later committed from the local machine) would block the merge:
# move them to logs/reports_prev/<time>/ first. Never touches anything outside reports/.
changed=$(git status --porcelain --untracked-files=all -- reports | cut -c4-)
if [ -n "$changed" ]; then
  keep="logs/reports_prev/$(date +%Y%m%dT%H%M%S)"
  mkdir -p "$keep"
  for f in $changed; do cp "$f" "$keep/"; done
  git checkout -q -- reports 2>/dev/null || true
  git status --porcelain --untracked-files=all -- reports | grep '^??' | cut -c4- | xargs -r rm -f
  echo "moved generated reports to $keep"
fi
git checkout -q "$BRANCH" 2>/dev/null || git checkout -q -b "$BRANCH" "origin/$BRANCH"
git "${SAFE[@]}" merge --ff-only "origin/$BRANCH"
git log --oneline -3
