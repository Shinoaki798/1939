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
