"""Quick census of the not-yet-ingested raw sources: docs / words / bytes, total and pre-cutoff.
Words = len(text.split()). Pre-cutoff: day dates <= 1939-06-30, year-only <= 1938."""
import glob, gzip, json, re, tarfile, zipfile, ast, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyarrow.parquet as pq

D = "/home/an/1939/data"
CUT_DAY, CUT_YEAR = "1939-06-30", 1938
TAG = re.compile(r"<[^>]+>")


def add(c, text, pre, en=None):
    w = len(text.split()); b = len(text.encode("utf-8", "ignore"))
    c["docs"] += 1; c["words"] += w; c["bytes"] += b
    if pre:
        c["docs_pre"] += 1; c["words_pre"] += w
        if en is not None:
            c["words_pre_en"] += w if en else 0
            c["has_lang"] = 1


def job_parquet(a):
    src, f, rg = a
    c = Counter(); pf = pq.ParquetFile(f)
    want = {"loc_pd_books": ["text", "year"], "evans_tcp": ["text", "year", "language"],
            "ecco_tcp": ["text", "year", "language"], "ncse": ["text", "year", "is_duplicate_box"],
            "ddb_newspapers_de": ["text", "date"], "europeana_newspapers_de": ["text", "date"]}[src]
    have = set(pf.schema_arrow.names)
    if "text" not in have:
        c["NO_TEXT_COLUMN"] += 1
        return src, c
    rows = (r for b in pf.iter_batches(batch_size=200, row_groups=[rg], columns=[x for x in want if x in have])
            for r in b.to_pylist())
    for r in rows:
        text = r.get("text") or ""
        if src == "loc_pd_books":
            add(c, text, (r.get("year") or 9999) <= CUT_YEAR)
        elif src in ("evans_tcp", "ecco_tcp"):
            add(c, text, (r.get("year") or 9999) <= CUT_YEAR, en=(r.get("language") == "eng"))
        elif src == "ncse":
            if r.get("is_duplicate_box"):
                c["dup_boxes"] += 1; continue
            add(c, text, (r.get("year") or 9999) <= CUT_YEAR)
        elif src in ("ddb_newspapers_de", "europeana_newspapers_de"):
            d = str(r.get("date") or "9999")
            add(c, text, d <= CUT_DAY)
    return src, c


def job_pre1929(f):
    c = Counter()
    with gzip.open(f, "rt") as g:
        for line in g:
            r = json.loads(line); m = r.get("metadata") or {}
            if isinstance(m, str):
                m = ast.literal_eval(m)
            y = m.get("year") or 9999
            add(c, r.get("text") or "", float(y) <= CUT_YEAR, en=(m.get("language") == "eng"))
    return "pre_1929_books", c


def job_ejc(f):
    c = Counter()
    with tarfile.open(f, "r|bz2") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".xml"):
                continue
            x = tf.extractfile(m).read().decode("utf-8", "replace")
            y = re.search(r"<year>(\d{4})", x)
            langs = re.findall(r"<languages>(.*?)</languages>", x, re.S)
            pages = re.findall(r"<list-item>(.*?)</list-item>", x, re.S)
            text = TAG.sub(" ", " ".join(pages))
            en = (not langs) or ("eng" in langs[0] or "en" in langs[0].split())
            add(c, text, int(y.group(1)) <= CUT_YEAR if y else False, en=en)
    return "jstor_ejc", c


def job_rsc(f):
    c = Counter()
    z = zipfile.ZipFile(f)
    for n in z.namelist():
        if n.endswith(".txt"):
            add(c, z.read(n).decode("utf-8", "replace"), True)   # corpus is 1665-1920
    return "royal_society_corpus", c


def job_dta(f):
    c = Counter()
    z = zipfile.ZipFile(f)
    for n in z.namelist():
        if not n.endswith(".xml"):
            continue
        m = re.search(r"_(\d{4})\d*\.TEI", n)
        y = int(m.group(1)) if m else 9999
        x = z.read(n).decode("utf-8", "replace")
        body = x.split("<text", 1)[-1]
        add(c, TAG.sub(" ", body), y <= CUT_YEAR)
        if y > CUT_YEAR:
            c["docs_after_cutoff"] += 1
    return "dta", c


def main():
    jobs = []
    for src, pat in [("loc_pd_books", "raw/loc_pd_books/*.parquet"), ("evans_tcp", "raw/evans_tcp/*.parquet"),
                     ("ecco_tcp", "raw/ecco_tcp/*.parquet"), ("ncse", "raw/ncse/*.parquet"),
                     ("ddb_newspapers_de", "foreign/de/raw/ddb_newspapers_de/*.parquet"),
                     ("europeana_newspapers_de", "foreign/de/raw/europeana_newspapers_de/*.parquet")]:
        for f in sorted(glob.glob(f"{D}/{pat}")):
            for rg in range(pq.ParquetFile(f).metadata.num_row_groups):
                jobs.append((job_parquet, (src, f, rg)))
    jobs += [(job_pre1929, f) for f in sorted(glob.glob(f"{D}/raw/pre_1929_books/*.jsonl.gz"))]
    jobs += [(job_ejc, f"{D}/raw/jstor_ejc/ejc.tar.bz2"),
             (job_rsc, glob.glob(f"{D}/raw/royal_society_corpus/*texts_txt.zip")[0]),
             (job_dta, glob.glob(f"{D}/foreign/de/raw/dta/*.zip")[0])]
    # big single-stream jobs first so they overlap with the many small ones
    jobs.sort(key=lambda j: 0 if j[0] in (job_ejc, job_dta, job_rsc) else 1)
    print(f"{len(jobs)} jobs", flush=True)
    t0 = time.time(); tot = {}
    # ProcessPoolExecutor raises BrokenProcessPool if a worker is OOM-killed (Pool would hang forever).
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            src, c = fut.result()
            tot.setdefault(src, Counter()).update(c)
            if i % 100 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    json.dump({k: dict(v) for k, v in tot.items()}, open("/home/an/1939/logs/census_raw.json", "w"), indent=1)
    for k, v in sorted(tot.items()):
        print(k, dict(v), flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


def run(j):
    return j[0](j[1])


if __name__ == "__main__":
    main()
