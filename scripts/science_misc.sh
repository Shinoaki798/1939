#!/usr/bin/env bash
# Science bucket, non-archive.org sources: Gutenberg catalog -> selection -> plain texts (PG mirror),
# then the pinned Dingler TEI tarball (GitHub). Via the Windows proxy; start via WMI like science_ia.sh.
set -uo pipefail
cd "$HOME/1939"
mkdir -p logs data/raw/gutenberg
PY="$HOME/miniconda3/envs/torch-gpu/bin/python"
log=logs/science_misc.log
echo "$(date -Is) science_misc start" >> "$log"
cat_file=data/raw/gutenberg/pg_catalog.csv.gz
if ! gzip -t "$cat_file" 2>/dev/null; then
  rm -f "$cat_file"
  /mnt/c/Windows/System32/curl.exe -sS --fail -x http://127.0.0.1:7890 -o "$(wslpath -w "$cat_file")" \
    https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz < /dev/null >> "$log" 2>&1
fi
echo "catalog $(sha256sum "$cat_file")" >> "$log"
"$PY" -u -m src.data.gutenberg_select >> "$log" 2>&1
for s in gutenberg_sci_en gutenberg_sci_de dingler; do
  echo "$(date -Is) download $s" >> "$log"
  "$PY" -u -m src.data.download --source "$s" --delay 2 --backoff 1800 --via proxy >> "logs/download_$s.log" 2>&1
  echo "$(date -Is) $s exit $?" >> "$log"
done
echo "$(date -Is) science_misc done" >> "$log"
