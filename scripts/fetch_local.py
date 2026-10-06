"""Fetch a url source's files on the LOCAL PC (the 2080, Windows), for hosts the 5080 cannot reach cheaply.

Same table and MANIFEST format as src.data.download (stdlib only, no curl/fcntl): resumable .part files,
publisher checksum verified, own sha256 recorded, `via: local-2080`. Polite: one file at a time,
--delay seconds apart, a 30-minute pause on HTTP 429/403 (never a bypass). Files land in the source's
`dest` under this checkout; ingest locally (python -m src.data.ingest_extra) and ship only the
ingested parquet to the 5080.

    python scripts/fetch_local.py --source chronicling_america --keys-file ca_left.txt --delay 30
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.download import load_config, load_file_table, repo_path  # noqa: E402

UA = "APS360-1939-corpus/1.0 (university course project; one file at a time)"


def fetch(url: str, part: Path, backoff: float) -> int:
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": UA, **({"Range": f"bytes={have}-"} if have else {})})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have and r.status == 206 else "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
            return r.status
    except urllib.error.HTTPError as e:
        if e.code in (429, 403):   # rate limit; a 403 that repeats after the pause is treated as restricted
            print(f"HTTP {e.code}: pausing {backoff:.0f}s", flush=True)
            time.sleep(backoff)
        return e.code
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        print(f"network error: {e}", flush=True)
        time.sleep(60)
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--keys-file", help="only these keys (one per line)")
    ap.add_argument("--delay", type=float, default=30.0)
    ap.add_argument("--backoff", type=float, default=1800.0)
    ap.add_argument("--attempts", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    src = load_config(repo_path(cfg["sources"]))[args.source]
    table = load_file_table(repo_path(src["files"]))
    keys = [k.strip() for k in open(args.keys_file, encoding="utf-8") if k.strip()] if args.keys_file else sorted(table)
    dest = repo_path(src["dest"])
    dest.mkdir(parents=True, exist_ok=True)
    mpath = dest / "MANIFEST.json"
    m = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {"source": args.source, "revision": None, "files": {}}
    todo = [k for k in keys if k not in m["files"]]
    print(f"{len(keys)} requested, {len(keys) - len(todo)} already verified, {len(todo)} to fetch", flush=True)
    for i, key in enumerate(todo, 1):
        spec = table[key]
        name = spec["file"].rsplit("/", 1)[-1]
        final, part = dest / name, dest / (name + ".part")
        algo, _, want = (spec.get("checksum") or "-").partition(":")
        urls, alt = spec["url"].split("|"), 0   # "a|b|c": a 404 moves on to the next alternative
        forbidden = 0
        for attempt in range(1, args.attempts + 1):
            t0 = time.time()
            url = urls[alt]
            if spec["bytes"] is None and part.exists():
                part.unlink()   # size unpublished: no resume, fetch the whole file each attempt
            status = fetch(url, part, args.backoff)
            size = part.stat().st_size if part.exists() else 0
            forbidden = forbidden + 1 if status == 403 else 0
            if status == 401 or forbidden >= 2:
                # access-restricted item (e.g. archive.org lending copies), not a rate limit: never retried
                print(f"[{key}] HTTP {status}{' again after the pause' if forbidden >= 2 else ''}; "
                      f"access restricted, giving up", flush=True)
                break
            if status == 404:
                if alt + 1 < len(urls):
                    alt += 1
                    continue
                print(f"[{key}] 404 on every URL; giving up", flush=True)
                break
            if spec["bytes"] is not None and size < spec["bytes"]:
                print(f"[{key}] attempt {attempt}: HTTP {status}, {size}/{spec['bytes']} bytes", flush=True)
                continue
            if spec["bytes"] is None and (status not in (200, 206) or size == 0):
                print(f"[{key}] attempt {attempt}: HTTP {status}, {size} bytes", flush=True)
                time.sleep(min(60, 10 * attempt))
                continue
            h = {"sha256": hashlib.sha256()}
            if algo in ("md5", "sha1"):
                h[algo] = hashlib.new(algo)
            with open(part, "rb") as f:
                while chunk := f.read(8 << 20):
                    for x in h.values():
                        x.update(chunk)
            got = {a: x.hexdigest() for a, x in h.items()}
            if algo in got and got[algo] != want:
                print(f"[{key}] checksum mismatch; deleting .part", flush=True)
                part.unlink()
                continue
            part.replace(final)
            m["files"][key] = {"file": name, "bytes": size, "sha256": got["sha256"],
                               "verified_against": spec.get("checksum") if algo in got else "size only",
                               "url": url, "via": "local-2080",
                               "verified_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
            tmp = mpath.with_suffix(".tmp")
            tmp.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(mpath)
            secs = max(time.time() - t0, 1e-6)
            print(f"[{key}] OK {size / 1e9:.3f} GB ({size / secs / 1e6:.1f} MB/s) [{i}/{len(todo)}]", flush=True)
            break
        else:
            print(f"[{key}] FAILED after {args.attempts} attempts", flush=True)
        time.sleep(args.delay)
    print("done", flush=True)


if __name__ == "__main__":
    main()
