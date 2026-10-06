"""Download the files of approved corpora onto the remote box.

Runs inside WSL on the 5080 machine. Sources are listed in `config/sources.yaml`.
HF sources are pinned to one revision and every file is verified against the sha256
in that source's file table; `kind: url` sources are verified against the
publisher's md5/sha1 where one is published (size only otherwise), and our own
sha256 is recorded either way.

Transport (`--via`):
  proxy   Windows-side curl.exe through the Windows system proxy. WSL in NAT
          mode cannot reach that proxy, but a Windows process launched from
          WSL can; curl.exe writes straight into the WSL filesystem via its
          \\\\wsl.localhost path. Fastest (15-24 MB/s measured).
  mirror  Linux curl without proxy: hf-mirror.com for HF sources (0.2-4 MB/s),
          the original URL for url sources.
  auto    resolved before every attempt: proxy if it answers, else mirror (default).

Downloads are resumable (.part files) and serial; `--source a,b,c` runs several
sources one after another. One lock per source prevents two runs of the same
source. Verified files are recorded in MANIFEST.json and never fetched again.

Start long downloads with scripts/start_download.ps1, not from an SSH session.

    python -m src.data.download --dry-run
    python -m src.data.download --source american_stories --years 1923-1955,1922-1900
    python -m src.data.download --source jstor_ejc,royal_society_corpus,dta
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
    """TSV (comment lines start with #) whose header is either
    `key file bytes sha256` (HF sources; the first column may be named `year`) or
    `key url bytes checksum` (url sources; checksum = md5:/sha1:/sha256:<hex> or '-').
    Returns {key: {"file", "bytes", "sha256" | "url" + "checksum"}}."""
    with open(path, encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip() and not line.startswith("#")]
    header, table = rows[0], {}
    for r in rows[1:]:
        rec = dict(zip(header[1:], r[1:]))
        rec["bytes"] = int(rec["bytes"])
        if "url" in rec:
            rec["file"] = rec["url"].rsplit("/", 1)[-1]
        table[r[0]] = rec
    return table


def load_manifest(path: Path, source: str, revision: str | None) -> dict:
    if path.exists():
        m = json.loads(path.read_text(encoding="utf-8"))
        if m.get("revision") != revision:
            sys.exit(f"MANIFEST revision {m.get('revision')} != configured {revision}; refusing to mix revisions")
        return m
    return {"source": source, "revision": revision, "files": {}}


def record_in_manifest(path: Path, manifest: dict, key: str, entry: dict) -> None:
    """Add one verified file under an exclusive lock, re-reading the MANIFEST first, so that several
    shard processes of the same source never overwrite each other's entries."""
    with open(path.with_suffix(".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else \
            {k: v for k, v in manifest.items() if k != "files"} | {"files": {}}
        current["files"][key] = entry
        save_manifest(path, current)


def save_manifest(path: Path, m: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def file_hashes(path: Path, algos: tuple[str, ...] = ("sha256",)) -> dict[str, str]:
    hs = {a: hashlib.new(a) for a in algos}
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            for h in hs.values():
                h.update(chunk)
    return {a: h.hexdigest() for a, h in hs.items()}


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


def source_url(spec: dict, src: dict, dl: dict, via: str) -> str:
    if "url" in spec:
        return spec["url"]
    endpoint = dl["hf_endpoint"] if via == "proxy" else dl["hf_mirror"]
    return f"{endpoint}/datasets/{src['hf_repo']}/resolve/{src['revision']}/{spec['file']}"


def verify(spec: dict, part: Path) -> tuple[bool, dict[str, str], str]:
    """-> (ok, hashes, expected). HF: sha256 must match. url: publisher checksum if given, else size only."""
    if "sha256" in spec:
        hs = file_hashes(part)
        return hs["sha256"] == spec["sha256"], hs, spec["sha256"]
    algo, _, want = spec.get("checksum", "-").partition(":")
    if algo in ("md5", "sha1", "sha256"):
        hs = file_hashes(part, tuple(dict.fromkeys(("sha256", algo))))
        return hs[algo] == want, hs, spec["checksum"]
    return True, file_hashes(part), "size only"


def fetch_one(key: str, spec: dict, src: dict, dl: dict, out_dir: Path, via: str, manifest: dict, mpath: Path,
              max_attempts: int) -> bool:
    name = spec["file"].rsplit("/", 1)[-1]
    final = out_dir / name
    part = out_dir / (name + ".part")

    for attempt in range(1, max_attempts + 1):
        # "auto" is resolved before every attempt: proxy whenever it answers, mirror/direct otherwise.
        cur = via if via != "auto" else ("proxy" if proxy_alive(dl["proxy"], dl["hf_endpoint"]) else "mirror")
        url = source_url(spec, src, dl, cur)
        have = part.stat().st_size if part.exists() else 0
        if have > spec["bytes"]:
            print(f"[{key}] .part larger than expected ({have} > {spec['bytes']}); restarting", flush=True)
            part.unlink()
            have = 0
        if have < spec["bytes"]:
            t0 = time.time()
            print(f"[{key}] attempt {attempt}: {have/1e9:.2f}/{spec['bytes']/1e9:.2f} GB via {cur}", flush=True)
            r = subprocess.run(curl_cmd(cur, url, part, dl["proxy"]), stdin=subprocess.DEVNULL)
            got = (part.stat().st_size if part.exists() else 0) - have
            secs = max(time.time() - t0, 1e-6)
            print(f"[{key}] curl exit {r.returncode}; +{got/1e6:.0f} MB in {secs:.0f}s ({got/secs/1e6:.1f} MB/s)",
                  flush=True)
            if not part.exists() or part.stat().st_size < spec["bytes"]:
                time.sleep(min(60, 10 * attempt))
                continue
        ok, hashes, expected = verify(spec, part)
        if not ok:
            print(f"[{key}] checksum mismatch (expected {expected}); deleting .part", flush=True)
            part.unlink()
            continue
        os.replace(part, final)
        entry = {
            "file": name, "bytes": spec["bytes"], "sha256": hashes["sha256"], "verified_against": expected,
            "url": url, "via": cur, "verified_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        manifest["files"][key] = entry
        record_in_manifest(mpath, manifest, key, entry)
        print(f"[{key}] OK {spec['bytes']/1e9:.2f} GB verified ({expected if expected == 'size only' else 'checksum'})",
              flush=True)
        return True
    print(f"[{key}] FAILED after {max_attempts} attempts", flush=True)
    return False


def run_source(name: str, src: dict, dl: dict, args) -> list[str]:
    table = load_file_table(repo_path(src["files"]))
    out_dir = repo_path(src["dest"])
    mpath = out_dir / "MANIFEST.json"
    ident = src.get("hf_repo", name)

    if name == "american_stories":
        keys = [str(y) for y in parse_years(args.years or dl["years"])]
        missing = [k for k in keys if k not in table]
        if missing:
            print(f"years not in the archive (skipped): {missing}")
        keys = [k for k in keys if k in table]
    else:
        keys = list(table)
    k_shard, n_shards = args.shard
    if n_shards > 1:   # stable split of the file list, so N processes can fetch one source in parallel
        keys = [k for k in keys if int(hashlib.sha1(k.encode()).hexdigest(), 16) % n_shards == k_shard]

    manifest = load_manifest(mpath, ident, src.get("revision")) \
        if mpath.exists() or not args.dry_run else {"files": {}}
    todo = [k for k in keys if k not in manifest["files"]]
    todo_bytes = sum(table[k]["bytes"] for k in todo)
    print(f"[{name}] {len(keys)} files requested, {len(keys) - len(todo)} already verified, {len(todo)} to "
          f"fetch ({todo_bytes/1e9:.1f} GB; ~{todo_bytes/15e6/3600:.1f} h at 15 MB/s) -> {out_dir}", flush=True)
    if args.dry_run:
        for k in todo[:20]:
            part = out_dir / (table[k]["file"].rsplit("/", 1)[-1] + ".part")
            have = part.stat().st_size if part.exists() else 0
            print(f"  {k}  {table[k]['bytes']/1e9:6.2f} GB  partial={have/1e9:.2f} GB")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    lock_name = ".download.lock" if n_shards == 1 else f".download.{k_shard}of{n_shards}.lock"
    lock = open(out_dir / lock_name, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[{name}] another downloader holds {out_dir}/{lock_name}; skipping", flush=True)
        return [f"{name}:locked"]
    failed = [k for k in todo if not fetch_one(k, table[k], src, dl, out_dir, args.via, manifest, mpath,
                                               args.max_attempts)]
    lock.close()
    print(f"[{name}] done. failed: {failed or 'none'}", flush=True)
    return [f"{name}:{k}" for k in failed]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "config" / "paths.yaml"))
    ap.add_argument("--source", default="american_stories", help="key(s) of config/sources.yaml, comma-separated")
    ap.add_argument("--years", default=None, help="american_stories only, e.g. 1923-1955,1900-1922 "
                                                   "(default: paths.yaml download.years)")
    ap.add_argument("--via", choices=["auto", "proxy", "mirror"], default="auto")
    ap.add_argument("--max-attempts", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true", help="report what would be downloaded; write nothing")
    ap.add_argument("--shard", default="0/1", help="k/N: fetch only the k-th of N stable slices of the file list "
                                                   "(run N processes for small-file sources)")
    args = ap.parse_args()
    k, n = (int(x) for x in args.shard.split("/"))
    if not 0 <= k < n:
        sys.exit(f"bad --shard {args.shard}")
    args.shard = (k, n)

    cfg = load_config(Path(args.config))
    sources = load_config(repo_path(cfg["sources"]))
    names = [s.strip() for s in args.source.split(",") if s.strip()]
    unknown = [n for n in names if n not in sources]
    if unknown:
        sys.exit(f"unknown source(s) {unknown}; approved sources: {sorted(sources)}")
    failed: list[str] = []
    for n in names:
        failed += run_source(n, sources[n], cfg["download"], args)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
