"""Move ingested parquet from the local PC (2080) to the 5080 through an intermediary (Baidu Netdisk).

pack   (on the 2080): copy data/ingested/<source>/ (parquet + MANIFEST.json) and the raw-download
       MANIFEST of each source into <out>/<source>/, and write <out>/TRANSFER.json with every file's
       sha256. The raw archives stay on the 2080 (their MANIFEST records where and how they were fetched).
merge  (on the 5080): verify every file in <in> against TRANSFER.json, copy the parquet into
       data/ingested/<source>/ (or the foreign root for German sources), merge the MANIFEST entries
       (refusing to overwrite a key that already exists with a different sha256), and store the 2080 raw
       MANIFEST as data/raw/<source>/MANIFEST.local-2080.json.

    python scripts/transfer_ingested.py pack --sources chronicling_america,psm_wikisource --out D:/1939_transfer
    python scripts/transfer_ingested.py merge --in /mnt/c/Users/AN/Downloads/1939_transfer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.download import load_config, repo_path  # noqa: E402
from src.data.ingest_extra import ingested_dir  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def pack(cfg: dict, sources: dict, names: list[str], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    listing = {}
    for name in names:
        src = sources[name]
        d = out / name
        d.mkdir(exist_ok=True)
        ing = ingested_dir(cfg, name, src)
        files = sorted(ing.glob("*.parquet")) + [ing / "MANIFEST.json"]
        raw_m = repo_path(src["dest"]) / "MANIFEST.json"
        for f in files:
            shutil.copy2(f, d / f.name)
        shutil.copy2(raw_m, d / "RAW_MANIFEST.json")
        listing[name] = {f.name: sha256(d / f.name) for f in sorted(d.iterdir())}
        print(f"{name}: {len(files) - 1} parquet, {sum(f.stat().st_size for f in d.iterdir()) / 1e9:.2f} GB", flush=True)
    (out / "TRANSFER.json").write_text(json.dumps(listing, indent=2), encoding="utf-8")
    print(f"-> {out} (upload this whole folder)", flush=True)


def merge(cfg: dict, sources: dict, inp: Path) -> None:
    listing = json.loads((inp / "TRANSFER.json").read_text(encoding="utf-8"))
    for name, files in listing.items():
        d = inp / name
        bad = [f for f, h in files.items() if not (d / f).exists() or sha256(d / f) != h]
        if bad:
            sys.exit(f"{name}: {len(bad)} files missing or corrupt after transfer, e.g. {bad[0]}; nothing merged")
        src = sources[name]
        ing = ingested_dir(cfg, name, src)
        ing.mkdir(parents=True, exist_ok=True)
        mpath = ing / "MANIFEST.json"
        mine = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else None
        theirs = json.loads((d / "MANIFEST.json").read_text(encoding="utf-8"))
        if mine is None:
            mine = {k: v for k, v in theirs.items() if k != "files"} | {"files": {}}
        for key, e in theirs["files"].items():
            old = mine["files"].get(key)
            if old and old.get("sha256") != e.get("sha256"):
                sys.exit(f"{name}/{key}: already ingested here with a different sha256; refusing to overwrite")
            shutil.copy2(d / e["file"], ing / e["file"])
            mine["files"][key] = dict(e, ingested_on="local-2080")
        tmp = mpath.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(mine, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(mpath)
        raw_dir = repo_path(src["dest"])
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(d / "RAW_MANIFEST.json", raw_dir / "MANIFEST.local-2080.json")
        print(f"{name}: merged {len(theirs['files'])} ingested files -> {ing}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["pack", "merge"])
    ap.add_argument("--sources", default="")
    ap.add_argument("--out")
    ap.add_argument("--in", dest="inp")
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    sources = load_config(repo_path(cfg["sources"]))
    if args.command == "pack":
        pack(cfg, sources, [s for s in args.sources.split(",") if s], Path(args.out))
    else:
        merge(cfg, sources, Path(args.inp))


if __name__ == "__main__":
    main()
