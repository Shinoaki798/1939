"""Download the American Stories per-year archives onto the remote box.

Runs inside WSL on the 5080 machine. Files are pinned to one HF revision and
verified against the sha256 listed in `config/american_stories_files.tsv`.

Transport (`--via`):
  proxy   Windows-side curl.exe through the Windows system proxy. WSL in NAT
          mode cannot reach that proxy, but a Windows process launched from
          WSL can; curl.exe writes straight into the WSL filesystem via its
          \\\\wsl.localhost path. Fastest (~24 MB/s measured).
  mirror  Linux curl against hf-mirror.com (no proxy needed, 0.2-4 MB/s).
  auto    proxy if it answers, otherwise mirror (default).

Downloads are resumable (.part files) and serial: the mirror throttles
parallel connections. Verified files are recorded in MANIFEST.json next to
them; a verified file is never fetched again.

    python -m src.data.download --dry-run
    python -m src.data.download --years 1923-1955,1900-1922
"""

from __future__ import annotations

import argparse
import datetime as dt
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


def load_file_table(path: Path) -> dict[int, dict]:
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("year\t") or not line.strip():
                continue
            year, name, size, sha = line.rstrip("\n").split("\t")
            table[int(year)] = {"file": name, "bytes": int(size), "sha256": sha}
    return table


def load_manifest(path: Path, source: str, revision: str) -> dict:
    if path.exists():
        m = json.loads(path.read_text(encoding="utf-8"))
        if m.get("revision") != revision:
            sys.exit(f"MANIFEST revision {m.get('revision')} != configured {revision}; refusing to mix revisions")
        return m
    return {"source": source, "revision": revision, "files": {}}


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


def fetch_one(year: int, spec: dict, out_dir: Path, cfg: dict, via: str, manifest: dict, mpath: Path,
              max_attempts: int) -> bool:
    final = out_dir / spec["file"]
    part = out_dir / (spec["file"] + ".part")
    endpoint = cfg["download"]["hf_endpoint"] if via == "proxy" else cfg["download"]["hf_mirror"]
    url = f"{endpoint}/datasets/{cfg['american_stories_hf']}/resolve/{cfg['american_stories_revision']}/{spec['file']}"

    for attempt in range(1, max_attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > spec["bytes"]:
            print(f"[{year}] .part larger than expected ({have} > {spec['bytes']}); restarting", flush=True)
            part.unlink()
            have = 0
        if have < spec["bytes"]:
            t0 = time.time()
            print(f"[{year}] attempt {attempt}: {have/1e9:.2f}/{spec['bytes']/1e9:.2f} GB via {via}", flush=True)
            r = subprocess.run(curl_cmd(via, url, part, cfg["download"]["proxy"]), stdin=subprocess.DEVNULL)
            got = (part.stat().st_size if part.exists() else 0) - have
            secs = max(time.time() - t0, 1e-6)
            print(f"[{year}] curl exit {r.returncode}; +{got/1e6:.0f} MB in {secs:.0f}s ({got/secs/1e6:.1f} MB/s)",
                  flush=True)
            if not part.exists() or part.stat().st_size < spec["bytes"]:
                time.sleep(min(60, 10 * attempt))
                continue
        digest = sha256_of(part)
        if digest != spec["sha256"]:
            print(f"[{year}] sha256 mismatch ({digest[:12]} != {spec['sha256'][:12]}); deleting .part", flush=True)
            part.unlink()
            continue
        os.replace(part, final)
        manifest["files"][str(year)] = {
            "file": spec["file"], "bytes": spec["bytes"], "sha256": digest, "url": url, "via": via,
            "verified_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        save_manifest(mpath, manifest)
        print(f"[{year}] OK {spec['bytes']/1e9:.2f} GB sha256 verified", flush=True)
        return True
    print(f"[{year}] FAILED after {max_attempts} attempts", flush=True)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "config" / "paths.yaml"))
    ap.add_argument("--years", default=None, help="e.g. 1923-1955,1900-1922 (default: config download.years)")
    ap.add_argument("--via", choices=["auto", "proxy", "mirror"], default="auto")
    ap.add_argument("--max-attempts", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true", help="report what would be downloaded; write nothing")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    table = load_file_table(repo_path(cfg["american_stories_files"]))
    out_dir = repo_path(cfg["raw_american_stories"])
    mpath = out_dir / "MANIFEST.json"
    years = parse_years(args.years or cfg["download"]["years"])
    missing = [y for y in years if y not in table]
    if missing:
        print(f"years not in the archive (skipped): {missing}")
    years = [y for y in years if y in table]

    manifest = load_manifest(mpath, cfg["american_stories_hf"], cfg["american_stories_revision"]) \
        if mpath.exists() or not args.dry_run else {"files": {}}
    todo = [y for y in years if str(y) not in manifest["files"]]
    todo_bytes = sum(table[y]["bytes"] for y in todo)
    print(f"{len(years)} years requested, {len(years) - len(todo)} already verified, {len(todo)} to fetch "
          f"({todo_bytes/1e9:.1f} GB; ~{todo_bytes/20e6/3600:.1f} h at 20 MB/s)")
    if args.dry_run:
        for y in todo:
            part = out_dir / (table[y]["file"] + ".part")
            have = part.stat().st_size if part.exists() else 0
            print(f"  {y}  {table[y]['bytes']/1e9:6.2f} GB  partial={have/1e9:.2f} GB")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    via = args.via
    if via == "auto":
        via = "proxy" if proxy_alive(cfg["download"]["proxy"], cfg["download"]["hf_endpoint"]) else "mirror"
    print(f"transport: {via}", flush=True)

    failed = []
    for y in todo:
        if args.via == "auto" and via == "proxy" and not proxy_alive(cfg["download"]["proxy"], cfg["download"]["hf_endpoint"]):
            via = "mirror"
            print("proxy stopped answering; falling back to mirror", flush=True)
        if not fetch_one(y, table[y], out_dir, cfg, via, manifest, mpath, args.max_attempts):
            failed.append(y)
    print(f"done. failed years: {failed or 'none'}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
