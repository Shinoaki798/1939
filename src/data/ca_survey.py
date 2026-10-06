"""Survey Chronicling America batches ingested after the American Stories snapshot.

For every batch in the LoC inventory (ocr.json) ingested on/after --since, fetch its issue list
(batch_1.xml) politely - one request at a time, --delay seconds apart, pausing --backoff seconds
on HTTP 429/403 - and record which batches contain issues dated 1930-01-01..1939-06-30. Writes
the candidate download table (key, url, bytes, sha256 checksum from the inventory) for
config/chronicling_america_files.tsv and a JSON summary. Issue lists are cached, never refetched.

Runs on the remote box through the Windows-side proxy (curl.exe via WSL interop), so start it
like a download (WMI-launched wsl.exe), not from an SSH session.

    python -m src.data.ca_survey --inventory data/raw/chronicling_america/inventory/ocr.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from src.data.download import CURL_EXE, load_config, repo_path, win_path

ISSUE = re.compile(r'<ndnp:issue lccn="([^"]+)" issueDate="([^"]+)"')
WINDOW = ("1930-01-01", "1939-06-30")


def fetch(url: str, out: Path, proxy: str) -> str:
    tmp = out.with_suffix(".tmp")
    r = subprocess.run([CURL_EXE, "-sSL", "--compressed", "--max-time", "120", "-x", proxy, "-A",
                        "Mozilla/5.0 (research; APS360 student project; one request per 15 s)",
                        "-o", win_path(tmp), "-w", "%{http_code}", url],
                       stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
    code = (r.stdout or "").strip()[-3:]
    if code == "200" and tmp.exists() and tmp.stat().st_size > 0:
        out.write_bytes(gzip.compress(tmp.read_bytes()))
    tmp.unlink(missing_ok=True)
    return code


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", required=True, help="LoC ocr.json (batch list with sha256)")
    ap.add_argument("--cache", default="data/raw/chronicling_america/batch_issue_lists")
    ap.add_argument("--since", default="2023-04-22", help="American Stories snapshot boundary")
    ap.add_argument("--delay", type=float, default=15.0)
    ap.add_argument("--backoff", type=float, default=1800.0)
    ap.add_argument("--start-delay", type=float, default=0.0)
    ap.add_argument("--out", default="logs/ca_survey")
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    proxy = cfg["download"]["proxy"]
    inv = json.load(open(repo_path(args.inventory), encoding="utf-8"))
    post = [b for b in inv if b["ingested"][:10] >= args.since]
    cache = repo_path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    todo = [b for b in post if not (cache / f"{b['batch']}.xml.gz").exists()]
    print(f"{len(post)} post-snapshot batches, {len(post) - len(todo)} cached, {len(todo)} to fetch", flush=True)
    if args.start_delay:
        time.sleep(args.start_delay)
    for i, b in enumerate(todo, 1):
        aw = b["batch"].split("_")[0]
        url = (f"https://tile.loc.gov/storage-services/service/ndnp/{aw}/batch_{b['batch']}/"
               f"{b['batch_file'].split('/', 1)[1]}")
        for attempt in range(8):
            code = fetch(url, cache / f"{b['batch']}.xml.gz", proxy)
            if code == "200":
                break
            if code in ("429", "403"):
                print(f"HTTP {code} on {b['batch']}: pausing {args.backoff:.0f}s", flush=True)
                time.sleep(args.backoff)
            elif code == "404":
                print(f"404 {b['batch']}", flush=True)
                break
            else:
                time.sleep(60)
        time.sleep(args.delay)
        if i % 25 == 0:
            print(f"{i}/{len(todo)}", flush=True)

    rows, summary = [], []
    for b in post:
        f = cache / f"{b['batch']}.xml.gz"
        if not f.exists():
            summary.append({"batch": b["batch"], "status": "not fetched"})
            continue
        iss = ISSUE.findall(gzip.decompress(f.read_bytes()).decode("utf-8", "replace"))
        win = [x for x in iss if WINDOW[0] <= x[1] <= WINDOW[1]]
        summary.append({"batch": b["batch"], "issues": len(iss), "window_issues": len(win),
                        "est_window_pages": round(b["page_count"] * len(win) / max(1, len(iss))),
                        "lccns": sorted({x[0] for x in win})})
        if win:
            rows.append((b["batch"], b["url"], b["size"], b["sha256"]))
    out = repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    with open(str(out) + "_files.tsv", "w", newline="\n", encoding="utf-8") as fh:
        fh.write("key\turl\tbytes\tchecksum\n")
        for name, url, size, sha in rows:
            fh.write(f"{name}\t{url}\t{size}\tsha256:{sha}\n")
    missing = sum(1 for s in summary if s.get("status") == "not fetched")
    print(f"done: {len(rows)} batches with window issues, "
          f"{sum(r[2] for r in rows) / 1e9:.1f} GB; {missing} batches still unfetched", flush=True)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
