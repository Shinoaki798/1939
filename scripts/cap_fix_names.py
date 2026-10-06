"""One-off repair (2026-10-05): the first Caselaw Access Project run saved every volume under the URL's
last path segment, so volumes with the same number in different reporters (ad/210.zip, cal/210.zip)
overwrote each other and MANIFEST entries point at files that no longer hold their bytes.

For every MANIFEST entry, the file still on disk is matched to the entry by sha256: the matching key
keeps it (renamed to its distinct name from config/caselaw_access_project_files.tsv); every other
entry for that old name is dropped so the downloader fetches it again. Files no entry matches are
moved to _unattributed/ (not deleted). Run only while no CAP download is running.

    python scripts/cap_fix_names.py --dry-run
    python scripts/cap_fix_names.py
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.download import file_hashes, load_config, load_file_table, repo_path  # noqa: E402

SOURCE = "caselaw_access_project"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    src = load_config(repo_path(cfg["sources"]))[SOURCE]
    table = load_file_table(repo_path(src["files"]))
    dest = repo_path(src["dest"])
    mpath = dest / "MANIFEST.json"
    with open(mpath.with_suffix(".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        m = json.loads(mpath.read_text(encoding="utf-8"))
        by_old = defaultdict(list)
        for key, e in m["files"].items():
            by_old[e["file"]].append(key)
        kept, dropped, moved, already = 0, 0, 0, 0
        for old, keys in sorted(by_old.items()):
            if all(old == table[k]["file"] for k in keys):
                already += len(keys)
                continue
            path = dest / old
            sha = file_hashes(path)["sha256"] if path.exists() else None
            owner = next((k for k in keys if m["files"][k]["sha256"] == sha), None)
            for k in keys:
                if k != owner:
                    del m["files"][k]
                    dropped += 1
            if owner:
                new = table[owner]["file"]
                if not args.dry_run:
                    os.replace(path, dest / new)
                m["files"][owner]["file"] = new
                kept += 1
            elif path.exists():
                if not args.dry_run:
                    (dest / "_unattributed").mkdir(exist_ok=True)
                    os.replace(path, dest / "_unattributed" / old)
                moved += 1
        print(f"kept+renamed {kept}, entries dropped for re-download {dropped}, "
              f"unattributed files moved {moved}, already distinct {already}; "
              f"{len(m['files'])} verified entries remain" + (" (dry run)" if args.dry_run else ""))
        if not args.dry_run:
            tmp = mpath.with_suffix(".tmp")
            tmp.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp, mpath)


if __name__ == "__main__":
    main()
