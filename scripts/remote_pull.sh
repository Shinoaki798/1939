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
  # proxy first; if the Windows-side proxy is down, GitHub is usually still reachable directly (slowly)
  "$GITEXE" -c http.proxy="$PROXY" -C "$MIRROR_WIN" remote update --prune </dev/null \
    || { echo "proxy fetch failed; trying GitHub directly"; "$GITEXE" -c http.proxy= -c https.proxy= -C "$MIRROR_WIN" remote update --prune </dev/null; }
fi

if [ ! -d "$WORK/.git" ]; then
  git "${SAFE[@]}" clone "$MIRROR_WSL" "$WORK"
fi
cd "$WORK"
git "${SAFE[@]}" fetch origin --prune
# Reports and catalog tables (config/*_files.tsv, *_items.tsv) generated on this box and later committed
# from the local machine would block the merge: move them to logs/reports_prev/<time>/ first.
# Never touches anything else.
candidates=$( { git status --porcelain --untracked-files=all -- reports | cut -c4-;
               git status --porcelain --untracked-files=all -- config | cut -c4- | grep -E '[.]tsv$'; } || true )
changed=""
for f in $candidates; do   # only files the incoming commit adds or changes would block the merge
  if git cat-file -e "origin/$BRANCH:$f" 2>/dev/null &&      { ! git ls-files --error-unmatch "$f" >/dev/null 2>&1 || ! git diff --quiet HEAD "origin/$BRANCH" -- "$f"; }; then
    changed="$changed $f"
  fi
done
if [ -n "$changed" ]; then
  keep="logs/reports_prev/$(date +%Y%m%dT%H%M%S)"
  mkdir -p "$keep"
  for f in $changed; do
    cp "$f" "$keep/"
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then git checkout -q -- "$f"; else rm -f "$f"; fi
  done
  echo "moved generated files to $keep:$changed"
fi
git checkout -q "$BRANCH" 2>/dev/null || git checkout -q -b "$BRANCH" "origin/$BRANCH"
git "${SAFE[@]}" merge --ff-only "origin/$BRANCH"
git log --oneline -3
