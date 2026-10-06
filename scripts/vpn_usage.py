"""Estimate VPN (Windows-side proxy) traffic from the download MANIFESTs.

Sums the bytes of every verified file whose MANIFEST entry says it came `via: proxy`, from --since on,
per source; JFM pages (no `via` field) count as proxy traffic. Files still in flight (.part) are added
separately. HTTP overhead, retries and failed transfers are not counted, so treat the total as a floor.

    python scripts/vpn_usage.py --since 2026-10-06T05:50:00+00:00
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="2026-10-06T05:50:00+00:00")
    args = ap.parse_args()
    by, parts = Counter(), Counter()
    for m in ROOT.rglob("MANIFEST.json"):
        if "/ingested/" in str(m) or "/lexicon/" in str(m):
            continue
        try:
            files = json.loads(m.read_text(encoding="utf-8")).get("files", {})
        except (json.JSONDecodeError, OSError):
            continue
        src = m.parent.name
        for e in files.values():
            when = e.get("verified_at") or e.get("harvested_at") or ""
            via = e.get("via", "proxy" if "harvested_at" in e else "")
            if via == "proxy" and when >= args.since:
                by[src] += e.get("bytes", 0)
        for p in m.parent.glob("*.part"):
            parts[src] += p.stat().st_size
    total = sum(by.values()) + sum(parts.values())
    for src, b in by.most_common():
        print(f"{src:32s} {b / 1e9:7.2f} GB")
    if parts:
        print("in flight:", {k: f"{v / 1e9:.2f} GB" for k, v in parts.items()})
    print(f"{'TOTAL since ' + args.since:32s} {total / 1e9:7.2f} GB")


if __name__ == "__main__":
    main()
