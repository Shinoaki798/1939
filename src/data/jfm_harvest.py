"""Harvest the Jahrbuch ueber die Fortschritte der Mathematik (JFM) from zbMATH Open over OAI-PMH.

    https://oai.zbmath.org/v1/?verb=ListRecords&metadataPrefix=oai_zb_preview&set=JFM

zbMATH Open content is CC BY-SA 4.0. zbmath.org itself blocks AI crawlers in robots.txt, so only the
OAI endpoint is used, one request at a time, --delay seconds apart. Every response page is kept raw
(gzip) under data/foreign/de/raw/jfm/pages/ and listed with its sha256 in the source MANIFEST.json;
the resumption token is saved after every page, so an interrupted run continues where it stopped
(tokens expire after about a day, or are lost server-side and answered with HTTP 500; then the run
starts over once and skips pages it already has).
Selection (JFM volume <= 61, reviews only) happens at ingest, not here.

    python -m src.data.jfm_harvest --delay 2
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from src.data.download import load_config, repo_path

BASE = "https://oai.zbmath.org/v1/"
UA = "APS360-1939-corpus/1.0 (university course project; polite OAI harvest, 1 request per few seconds)"
_TOKEN = re.compile(r"<resumptionToken([^>]*)>([^<]*)</resumptionToken>")


def fetch_via_proxy(url: str, proxy: str) -> bytes | None:
    """curl.exe through the Windows-side proxy (WSL cannot reach it directly); None if that fails."""
    import subprocess
    from src.data.download import CURL_EXE
    r = subprocess.run([CURL_EXE, "-sS", "--fail", "--max-time", "300", "-x", proxy, "-A", UA, url],
                       stdin=subprocess.DEVNULL, capture_output=True)
    return r.stdout if r.returncode == 0 and r.stdout else None


def fetch(params: dict, tries: int = 8, proxy: str | None = None) -> bytes:
    url = BASE + "?" + urllib.parse.urlencode(params)
    for attempt in range(tries):
        if proxy:
            body = fetch_via_proxy(url, proxy)
            if body is not None:
                return body
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            wait = 1800 if e.code in (429, 403, 503) else 60 * (attempt + 1)
            print(f"HTTP {e.code}; waiting {wait}s", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"network error {e}; waiting {60 * (attempt + 1)}s", flush=True)
            time.sleep(60 * (attempt + 1))
    raise SystemExit(f"giving up on {url}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--max-pages", type=int, default=0, help="stop after this many new pages (0 = all)")
    ap.add_argument("--direct", action="store_true", help="no Windows-side proxy (the local PC)")
    args = ap.parse_args()
    cfg = load_config(repo_path("config/paths.yaml"))
    proxy = None if args.direct else cfg["download"]["proxy"]
    src = load_config(repo_path(cfg["sources"]))["jfm"]
    dest = repo_path(src["dest"])
    pages = dest / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    mpath, state_path = dest / "MANIFEST.json", dest / "harvest_state.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {
        "source": "jfm", "revision": None, "endpoint": BASE, "set": "JFM", "metadataPrefix": "oai_zb_preview",
        "licence": "CC BY-SA 4.0 (zbMATH Open)", "files": {}}
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    params = {"verb": "ListRecords", "resumptionToken": state["token"]} if state.get("token") else \
             {"verb": "ListRecords", "metadataPrefix": "oai_zb_preview", "set": "JFM"}
    cursor, new, restarted = state.get("cursor", 0), 0, False
    while True:
        try:
            body = fetch(params, tries=3 if "resumptionToken" in params else 8, proxy=proxy)
        except SystemExit:
            # zbMATH answers a token it no longer knows with HTTP 500, not with badResumptionToken
            if "resumptionToken" not in params or restarted:
                raise
            print("resumption token keeps failing; restarting from the first page", flush=True)
            params, cursor, restarted = {"verb": "ListRecords", "metadataPrefix": "oai_zb_preview", "set": "JFM"}, 0, True
            continue
        text = body.decode("utf-8", "replace")
        if "<error" in text and "badResumptionToken" in text:
            print("resumption token expired; restarting from the first page", flush=True)
            params, cursor = {"verb": "ListRecords", "metadataPrefix": "oai_zb_preview", "set": "JFM"}, 0
            continue
        key = f"page_{cursor:07d}"
        out = pages / f"{key}.xml.gz"
        if key not in manifest["files"]:
            out.write_bytes(gzip.compress(body))
            manifest["files"][key] = {"file": f"pages/{out.name}", "bytes": out.stat().st_size,
                                      "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                                      "records": text.count("<record>"),
                                      "harvested_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
            new += 1
        m = _TOKEN.search(text)
        token = m.group(2).strip() if m else ""
        size = re.search(r'completeListSize="(\d+)"', m.group(1)) if m else None
        cursor += text.count("<record>")
        state = {"token": token, "cursor": cursor, "complete_list_size": int(size.group(1)) if size else None}
        tmp = mpath.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp.replace(mpath)
        state_path.write_text(json.dumps(state), encoding="utf-8")
        if len(manifest["files"]) % 50 == 0:
            print(f"{len(manifest['files'])} pages, cursor {cursor}/{state['complete_list_size']}", flush=True)
        if not token or (args.max_pages and new >= args.max_pages):
            break
        params = {"verb": "ListRecords", "resumptionToken": token}
        time.sleep(args.delay)
    print(f"done: {len(manifest['files'])} pages, {cursor} records", flush=True)


if __name__ == "__main__":
    main()
