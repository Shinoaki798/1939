"""Delete the raw downloads of a training-only source once its ingest is verified.

Allowed by the user on 2026-10-05 for supplementary corpora only; American Stories raw
archives are never pruned. A raw file is deleted only if
  1. its key is in the ingested MANIFEST with an output file,
  2. that output file exists and its sha256 equals the MANIFEST's,
  3. pyarrow can open it and it has at least one row.
The raw MANIFEST keeps every entry (sha256, URL, revision) plus `pruned_at`, so the
exact file can be fetched again; download.py will not re-fetch it on its own.

    python -m src.data.prune_raw --source hmd_newspapers --dry-run
    python -m src.data.prune_raw --source hmd_newspapers
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from src.data.download import file_hashes, load_config, repo_path, save_manifest
from src.data.ingest_extra import ingested_dir

NEVER_PRUNE = {"american_stories"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.source in NEVER_PRUNE:
        sys.exit(f"{args.source}: raw archives are never pruned")

    cfg = load_config(Path(args.config) if args.config else repo_path("config/paths.yaml"))
    src = load_config(repo_path(cfg["sources"]))[args.source]
    if str(src.get("role", "")).startswith("lexicon anchor"):
        sys.exit(f"{args.source}: lexicon anchors are measuring instruments and are never pruned")
    raw_dir = repo_path(src["dest"])
    raw_mpath = raw_dir / "MANIFEST.json"
    raw_m = json.loads(raw_mpath.read_text(encoding="utf-8"))
    ing_dir = ingested_dir(cfg, args.source, src)
    ing_m = json.loads((ing_dir / "MANIFEST.json").read_text(encoding="utf-8"))

    to_delete, problems, freed = [], [], 0
    for key, entry in sorted(raw_m["files"].items()):
        raw_file = raw_dir / entry["file"]
        if entry.get("pruned_at") or not raw_file.exists():
            continue
        out = ing_m["files"].get(key)
        if not out or "file" not in out:
            problems.append(f"{key}: not ingested")
            continue
        out_file = ing_dir / out["file"]
        if not out_file.exists():
            problems.append(f"{key}: ingested file {out['file']} missing")
            continue
        if file_hashes(out_file)["sha256"] != out["sha256"]:
            problems.append(f"{key}: ingested file sha256 mismatch")
            continue
        if pq.ParquetFile(out_file).metadata.num_rows < 1:
            problems.append(f"{key}: ingested file has no rows")
            continue
        to_delete.append((key, raw_file))
        freed += raw_file.stat().st_size

    print(f"[{args.source}] {len(to_delete)} raw files verified for deletion ({freed/1e9:.2f} GB); "
          f"{len(problems)} kept: {problems[:5]}")
    if args.dry_run or not to_delete:
        return
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    for key, raw_file in to_delete:
        raw_file.unlink()
        raw_m["files"][key]["pruned_at"] = now
    save_manifest(raw_mpath, raw_m)
    print(f"[{args.source}] deleted {len(to_delete)} raw files, freed {freed/1e9:.2f} GB")


if __name__ == "__main__":
    main()
