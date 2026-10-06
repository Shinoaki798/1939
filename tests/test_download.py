from pathlib import Path

from src.data.download import load_file_table, parse_years

REPO = Path(__file__).resolve().parents[1]


def test_parse_years_ranges_order_and_dedup():
    assert parse_years("1923-1925,1922-1920") == [1923, 1924, 1925, 1922, 1921, 1920]
    assert parse_years("1939, 1939,1938") == [1939, 1938]


def test_file_table_covers_study_years_with_sha256():
    table = load_file_table(REPO / "config" / "american_stories_files.tsv")
    for year in range(1900, 1956):
        spec = table[str(year)]
        assert spec["file"] == f"faro_{year}.tar.gz"
        assert spec["bytes"] > 0
        assert len(spec["sha256"]) == 64


def test_url_table_and_publisher_checksum(tmp_path):
    import hashlib

    from src.data.download import verify

    payload = b"Early Journal Content"
    f = tmp_path / "ejc.tar.bz2.part"
    f.write_bytes(payload)
    tsv = tmp_path / "t.tsv"
    tsv.write_text("# c\nkey\turl\tbytes\tchecksum\n"
                   f"ejc\thttps://archive.org/download/x/ejc.tar.bz2\t{len(payload)}\tmd5:{hashlib.md5(payload).hexdigest()}\n"
                   f"meta\thttps://example.org/meta.zip\t3\t-\n", encoding="utf-8")
    table = load_file_table(tsv)
    assert table["ejc"]["file"] == "ejc.tar.bz2" and table["ejc"]["bytes"] == len(payload)
    ok, hashes, _ = verify(table["ejc"], f)
    assert ok and hashes["sha256"] == hashlib.sha256(payload).hexdigest()
    table["ejc"]["checksum"] = "md5:" + "0" * 32
    assert not verify(table["ejc"], f)[0]
    assert verify(table["meta"], f)[2] == "size only"


def test_record_in_manifest_merges_concurrent_writers(tmp_path):
    import json

    from src.data.download import record_in_manifest

    m = tmp_path / "MANIFEST.json"
    a = {"source": "x", "revision": None, "files": {}}
    b = {"source": "x", "revision": None, "files": {}}
    record_in_manifest(m, a, "k1", {"sha256": "1"})
    record_in_manifest(m, b, "k2", {"sha256": "2"})   # b never saw k1 in memory
    assert set(json.loads(m.read_text())["files"]) == {"k1", "k2"}


def test_unknown_size_zip_is_verified_by_integrity(tmp_path):
    import zipfile

    from src.data.download import verify

    good = tmp_path / "173.zip.part"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("CasesMetadata.json", "[]")
    spec = {"file": "173.zip", "url": "https://static.case.law/la/173.zip", "bytes": None, "checksum": "-"}
    ok, hashes, how = verify(spec, good)
    assert ok and how == "zip integrity" and len(hashes["sha256"]) == 64
    bad = tmp_path / "bad.zip.part"
    bad.write_bytes(b"<html>rate limited</html>")
    assert not verify(spec, bad)[0]
