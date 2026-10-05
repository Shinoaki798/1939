"""Download the files of an approved corpus onto the remote box.

Runs inside WSL on the 5080 machine. Sources are listed in `config/sources.yaml`;
each is pinned to one HF revision, and every file is verified against the sha256
in that source's file table.

Transport (`--via`):
  proxy   Windows-side curl.exe through the Windows system proxy. WSL in NAT
          mode cannot reach that proxy, but a Windows process launched from
          WSL can; curl.exe writes straight into the WSL filesystem via its
          \\\\wsl.localhost path. Fastest (15-24 MB/s measured).
  mirror  Linux curl against hf-mirror.com (no proxy needed, 0.2-4 MB/s).
  auto    proxy if it answers, otherwise mirror (default).

Downloads are resumable (.part files) and serial within a source; different
sources may run at the same time (one lock per source). Verified files are
recorded in MANIFEST.json next to them and never fetched again.

Start long downloads with scripts/start_download.ps1, not from an SSH session.

    python -m src.data.download --dry-run
    python -m src.data.download --source american_stories --years 1923-1955,1922-1900
    python -m src.data.download --source hmd_newspapers
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CURL_EXE = "/mnt/c/Windows/System32/curl.exe"


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (REPO / q).resolve()


def parse_years(spec: str) -> list[int]:
    """'1923-1955,1900-1922,1899' -> ordered, de-duplicated list of years."""
    years: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            step = 1 if b >= a else -1
            rng = range(a, b + step, step)
        else:
            rng = [int(part)]
        for y in rng:
            if y not in years:
                years.append(y)
    return years


def load_file_table(path: Path) -> dict[str, dict]:
    """TSV with header `<key> file bytes sha256` (comment lines start with #) -> {key: spec}."""
    table: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip() and not line.startswith("#")]
    for key, name, size, sha in rows[1:]:
        table[key] = {"file": name, "bytes": int(size), "sha256": sha}
    return table


def load_manifest(path: Path, hf_repo: str, revision: str) -> dict:
    if path.exists():
        m = json.loads(path.read_text(encoding="utf-8"))
        if m.get("revision") != revision:
            sys.exit(f"MANIFEST revision {m.get('revision')} != configured {revision}; refusing to mix revisions")
        return m
    return {"source": hf_repo, "revision": revision, "files": {}}


def save_manifest(path: Path, m: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def win_path(p: Path) -> str:
    return subprocess.run(["wslpath", "-w", str(p)], check=True, capture_output=True, text=True).stdout.strip()


def proxy_alive(proxy: str, endpoint: str) -> bool:
    if not Path(CURL_EXE).exists():
        return False
    r = subprocess.run(
        [CURL_EXE, "-sS", "-o", "NUL", "-w", "%{http_code}", "--max-time", "20", "-x", proxy, "-I", endpoint],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
    )
    return r.stdout.strip() not in ("", "000")


def curl_cmd(via: str, url: str, part: Path, proxy: str) -> list[str]:
    # No curl-internal --retry: a retried transfer may not re-read the resume offset. The outer loop in
    # fetch_one re-measures the .part file before every attempt; stalls (<50 KB/s for 90 s) abort and retry.
    common = ["-L", "-C", "-", "--fail", "-sS", "--connect-timeout", "30",
              "--speed-limit", "51200", "--speed-time", "90"]
    if via == "proxy":
        return [CURL_EXE, *common, "-x", proxy, "-o", win_path(part), url]
    return ["curl", *common, "-o", str(part), url]


def fetch_one(key: str, spec: dict, src: dict, dl: dict, out_dir: Path, via: str, manifest: dict, mpath: Path,
              max_attempts: int) -> bool:
    name = spec["file"].rsplit("/", 1)[-1]
    final = out_dir / name
    part = out_dir / (name + ".part")
    endpoint = dl["hf_endpoint"] if via == "proxy" else dl["hf_mirror"]
    url = f"{endpoint}/datasets/{src['hf_repo']}/resolve/{src['revision']}/{spec['file']}"

    for attempt in range(1, max_attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > spec["bytes"]:
            print(f"[{key}] .part larger than expected ({have} > {spec['bytes']}); restarting", flush=True)
            part.unlink()
            have = 0
        if have < spec["bytes"]:
            t0 = time.time()
            print(f"[{key}] attempt {attempt}: {have/1e9:.2f}/{spec['bytes']/1e9:.2f} GB via {via}", flush=True)
            r = subprocess.run(curl_cmd(via, url, part, dl["proxy"]), stdin=subprocess.DEVNULL)
            got = (part.stat().st_size if part.exists() else 0) - have
            secs = max(time.time() - t0, 1e-6)
            print(f"[{key}] curl exit {r.returncode}; +{got/1e6:.0f} MB in {secs:.0f}s ({got/secs/1e6:.1f} MB/s)",
                  flush=True)
            if not part.exists() or part.stat().st_size < spec["bytes"]:
                time.sleep(min(60, 10 * attempt))
                continue
        digest = sha256_of(part)
        if digest != spec["sha256"]:
            print(f"[{key}] sha256 mismatch ({digest[:12]} != {spec['sha256'][:12]}); deleting .part", flush=True)
            part.unlink()
            continue
        os.replace(part, final)
        manifest["files"][key] = {
            "file": name, "bytes": spec["bytes"], "sha256": digest, "url": url, "via": via,
            "verified_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        save_manifest(mpath, manifest)
        print(f"[{key}] OK {spec['bytes']/1e9:.2f} GB sha256 verified", flush=True)
        return True
    print(f"[{key}] FAILED after {max_attempts} attempts", flush=True)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "config" / "paths.yaml"))
    ap.add_argument("--source", default="american_stories", help="a key of config/sources.yaml")
    ap.add_argument("--years", default=None, help="american_stories only, e.g. 1923-1955,1900-1922 "
                                                   "(default: paths.yaml download.years)")
    ap.add_argument("--via", choices=["auto", "proxy", "mirror"], default="auto")
    ap.add_argument("--max-attempts", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true", help="report what would be downloaded; write nothing")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    dl = cfg["download"]
    sources = load_config(repo_path(cfg["sources"]))
    if args.source not in sources:
        sys.exit(f"unknown source {args.source!r}; approved sources: {sorted(sources)}")
    src = sources[args.source]
    table = load_file_table(repo_path(src["files"]))
    out_dir = repo_path(src["dest"])
    mpath = out_dir / "MANIFEST.json"

    if args.source == "american_stories":
        keys = [str(y) for y in parse_years(args.years or dl["years"])]
        missing = [k for k in keys if k not in table]
        if missing:
            print(f"years not in the archive (skipped): {missing}")
        keys = [k for k in keys if k in table]
    else:
        keys = list(table)

    manifest = load_manifest(mpath, src["hf_repo"], src["revision"]) \
        if mpath.exists() or not args.dry_run else {"files": {}}
    todo = [k for k in keys if k not in manifest["files"]]
    todo_bytes = sum(table[k]["bytes"] for k in todo)
    print(f"[{args.source}] {len(keys)} files requested, {len(keys) - len(todo)} already verified, {len(todo)} to "
          f"fetch ({todo_bytes/1e9:.1f} GB; ~{todo_bytes/15e6/3600:.1f} h at 15 MB/s)", flush=True)
    if args.dry_run:
        for k in todo:
            part = out_dir / (table[k]["file"].rsplit("/", 1)[-1] + ".part")
            have = part.stat().st_size if part.exists() else 0
            print(f"  {k}  {table[k]['bytes']/1e9:6.2f} GB  partial={have/1e9:.2f} GB")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    lock = open(out_dir / ".download.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"[{args.source}] another downloader holds {out_dir}/.download.lock; not starting")

    via = args.via
    if via == "auto":
        via = "proxy" if proxy_alive(dl["proxy"], dl["hf_endpoint"]) else "mirror"
    print(f"[{args.source}] transport: {via}", flush=True)

    failed = []
    for k in todo:
        if args.via == "auto" and via == "proxy" and not proxy_alive(dl["proxy"], dl["hf_endpoint"]):
            via = "mirror"
            print("proxy stopped answering; falling back to mirror", flush=True)
        if not fetch_one(k, table[k], src, dl, out_dir, via, manifest, mpath, args.max_attempts):
            failed.append(k)
    print(f"[{args.source}] done. failed: {failed or 'none'}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
