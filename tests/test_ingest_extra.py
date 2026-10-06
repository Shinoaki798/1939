import datetime as dt
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.ingest_extra import rows_congressional_record, rows_hmd_newspapers

CUTOFF = dt.date(1939, 6, 30)


def test_congressional_record_drops_after_cutoff_and_bad_dates(tmp_path):
    t = pa.table({
        "speech_id": [1, 2, 3, 4], "date": ["19390630", "19390701", "1939xx01", "19380105"],
        "chamber": ["S", "H", "S", "H"], "speaker": ["A", "B", "C", "D"], "state": ["", "", "", ""],
        "congress": [76, 76, 76, 75], "party": ["", "", "", ""], "source": ["", "", "", ""],
        "speech_text": ["kept on the cutoff day", "after the cutoff", "bad date", "kept too"],
    })
    path = tmp_path / "cr.parquet"
    pq.write_table(t, path)
    stats = Counter()
    rows = list(rows_congressional_record(path, "congress_076", CUTOFF, stats))
    assert [r["article_id"] for r in rows] == ["cr_1", "cr_4"]
    assert rows[0]["date"] == "1939-06-30" and rows[0]["newspaper"] == "Congressional Record (Senate)"
    assert stats["dropped_after_cutoff"] == 1 and stats["dropped_bad_date"] == 1


def test_hmd_rows_have_stable_ids_and_dates(tmp_path):
    t = pa.table({
        "title": ["The Times", "The Times"], "location": ["London", "London"],
        "date": pa.array([dt.datetime(1855, 7, 17), None], pa.timestamp("ms")),
        "item_type": ["ARTICLE", "ADVERT"], "ocr_quality_mean": [0.9, 0.5], "ocr_quality_sd": [0.1, 0.2],
        "text": ["The Queen and the Prince were at the palace.", "no date"],
    })
    path = tmp_path / "hmd.parquet"
    pq.write_table(t, path)
    stats = Counter()
    rows = list(rows_hmd_newspapers(path, "train-00000-of-00029", CUTOFF, stats))
    assert [r["article_id"] for r in rows] == ["hmd_train-00000-of-00029_0"]
    assert rows[0]["date"] == "1855-07-17" and rows[0]["year"] == 1855
    assert stats["dropped_bad_date"] == 1


def test_german_adapters_keep_post_cutoff_rows_until_1955(tmp_path):
    from src.data.ingest_extra import rows_ddb_newspapers_de, rows_europeana_newspapers_de

    ddb = pa.table({
        "issue_id": ["A", "B", "C"], "date": pa.array([dt.date(1938, 5, 1), dt.date(1941, 1, 2), dt.date(1960, 1, 1)]),
        "paper": ["Hallische Nachrichten"] * 3, "page": [1, 2, 3], "language": ["ger"] * 3,
        "zdb_id": ["z"] * 3, "provider": ["p"] * 3, "license": ["pdm"] * 3,
        "text": ["Die Stadt und das Land", "Nach dem Kriege", "zu spaet"],
    })
    pq.write_table(ddb, tmp_path / "ddb.parquet")
    stats = Counter()
    rows = list(rows_ddb_newspapers_de(tmp_path / "ddb.parquet", "data_0", CUTOFF, stats))
    assert [r["date"] for r in rows] == ["1938-05-01", "1941-01-02"]   # 1941 kept for the German curve
    assert stats["dropped_after_1955"] == 1 and rows[0]["article_id"] == "ddb_A_1"

    eu = pa.table({
        "id": ["x1", "x2"], "date": ["1936-03-01", "not-a-date"], "title": ["Hamburger Anzeiger"] * 2,
        "mean_ocr": [0.55, 0.6], "std_ocr": [0.1, 0.1], "language": [["de"], ["de"]],
        "multi_language": [False, False], "issue_uri": ["u", "u"], "text": ["Der Senat der Stadt", "x"],
    })
    pq.write_table(eu, tmp_path / "eu.parquet")
    stats = Counter()
    rows = list(rows_europeana_newspapers_de(tmp_path / "eu.parquet", "de-1930", CUTOFF, stats))
    assert len(rows) == 1 and rows[0]["source"] == "europeana_newspapers_de" and stats["dropped_bad_date"] == 1


def test_books_keep_1900_to_1938_only(tmp_path):
    from src.data.ingest_extra import rows_loc_pd_books

    t = pa.table({"lccn": ["a", "b", "c"], "title": ["T"] * 3, "author": ["A"] * 3, "year": [1899, 1925, 1941],
                  "page_count": [10] * 3, "filename": ["a.txt", "b.txt", "c.txt"],
                  "text": ["old book", "a book of 1925", "too late"]})
    pq.write_table(t, tmp_path / "loc.parquet")
    stats = Counter()
    rows = list(rows_loc_pd_books(tmp_path / "loc.parquet", "train_00001", CUTOFF, stats))
    assert [r["date"] for r in rows] == ["1925-01-01"] and stats["dropped_year_outside_1900_1938"] == 2
    assert '"date_precision": "year"' in rows[0]["meta"]


def test_chronicling_america_window_and_ids(tmp_path):
    import io
    import tarfile

    from src.data.ingest_extra import rows_chronicling_america

    path = tmp_path / "x_batch_ver01.tar.bz2"
    with tarfile.open(path, "w:bz2") as tf:
        for name, txt in [("x_batch_ver01/data/sn86069021/1931/05/14/ed-1/seq-3/ocr.txt", "Falmouth news 1931"),
                          ("x_batch_ver01/data/sn86069021/1941/05/14/ed-1/seq-1/ocr.txt", "after the window"),
                          ("x_batch_ver01/data/sn86069021/1931/05/14/ed-1/seq-3/ocr.xml", "<alto/>")]:
            b = txt.encode()
            info = tarfile.TarInfo(name); info.size = len(b)
            tf.addfile(info, io.BytesIO(b))
    stats = Counter()
    rows = list(rows_chronicling_america(path, "x_batch_ver01", CUTOFF, stats))
    assert [r["article_id"] for r in rows] == ["ca_sn86069021_1931-05-14_ed1_seq3"]
    assert rows[0]["lccn"] == "sn86069021" and rows[0]["page"] == "p3" and stats["outside_window"] == 1


def test_federal_register_reads_left_column_first(tmp_path):
    import pytest
    fitz = pytest.importorskip("fitz")
    from src.data.ingest_extra import rows_federal_register

    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((40, 100), "LEFT ONE")
    page.insert_text((40, 400), "LEFT TWO")
    page.insert_text((340, 50), "RIGHT ONE")
    path = tmp_path / "FR-1937-01-05.pdf"
    doc.save(path)
    rows = list(rows_federal_register(path, "FR-1937-01-05", CUTOFF, Counter()))
    t = rows[0]["text"]
    assert t.index("LEFT ONE") < t.index("LEFT TWO") < t.index("RIGHT ONE")
    assert rows[0]["date"] == "1937-01-05"


def test_caselaw_opinions_only_and_cutoff(tmp_path):
    import json
    import zipfile

    from src.data.ingest_extra import _cap_date, rows_caselaw_access_project

    def case(cid, date, opinions):
        return {"id": cid, "decision_date": date, "name_abbreviation": f"Case {cid}",
                "citations": [{"cite": f"{cid} Cal. 8"}], "court": {"name": "Supreme Court of California"},
                "jurisdiction": {"name_long": "California"},
                "casebody": {"head_matter": "Attorneys for appellant.", "opinions": opinions}}

    path = tmp_path / "cal__210.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("json/0008-01.json", json.dumps(case(1, "1934-05-17", [
            {"type": "majority", "author": "SHENK, J.", "text": "The appeal is taken."},
            {"type": "dissent", "author": "CURTIS, J.", "text": "I dissent."}])))
        z.writestr("json/0011-01.json", json.dumps(case(2, "1939-07-03", [{"type": "majority", "text": "Late."}])))
        z.writestr("json/0012-01.json", json.dumps(case(3, "1939-06", [{"type": "majority", "text": "June."}])))
        z.writestr("metadata/CasesMetadata.json", "[]")
    stats = Counter()
    rows = list(rows_caselaw_access_project(path, "cal__210", CUTOFF, stats))
    assert [r["article_id"] for r in rows] == ["cap_1", "cap_3"]
    assert rows[0]["text"] == "The appeal is taken.\n\nI dissent." and "Attorneys" not in rows[0]["text"]
    assert rows[0]["byline"] == "SHENK, J." and rows[0]["state"] == "California"
    assert rows[1]["date"] == "1939-06-01" and stats["dropped_after_cutoff"] == 1
    assert _cap_date("1939", CUTOFF) == (None, "year") and _cap_date("1938", CUTOFF)[0] == dt.date(1938, 1, 1)
    assert _cap_date("1939-12", CUTOFF)[0] is None and _cap_date("n/a", CUTOFF) == (None, "bad")
