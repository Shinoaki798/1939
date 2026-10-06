import datetime as dt
from collections import Counter

from src.data.ia_catalog import in_window, item_date
from src.data.ingest_extra import ia_chunks, ia_clean

LO, HI = dt.date(1915, 1, 1), dt.date(1939, 6, 30)


def test_item_date_prefers_identifier_then_date_field():
    assert item_date({"identifier": "sim_physical-review_1939-06-15_55_12"}) == (dt.date(1939, 6, 15), "day")
    assert item_date({"identifier": "sim_popular-science_1916-01_88_1"}) == (dt.date(1916, 1, 1), "month")
    assert item_date({"identifier": "x", "date": "1935-04-15T00:00:00Z"}) == (dt.date(1935, 4, 15), "day")
    assert item_date({"identifier": "x", "date": "1910-01-01T00:00:00Z"}) == (dt.date(1910, 1, 1), "year")
    assert item_date({"identifier": "x", "year": "1922"}) == (dt.date(1922, 1, 1), "year")
    assert item_date({"identifier": "x"}) == (None, "none")


def test_in_window_needs_the_whole_span():
    assert in_window(dt.date(1939, 6, 15), "day", LO, HI)
    assert not in_window(dt.date(1939, 7, 1), "day", LO, HI)
    assert in_window(dt.date(1939, 6, 1), "month", LO, HI)
    assert not in_window(dt.date(1939, 1, 1), "year", LO, HI)       # year-only 1939 is out
    assert in_window(dt.date(1938, 1, 1), "year", LO, HI)


def test_ia_clean_strips_google_boilerplate_and_unwraps():
    raw = ("This is a digital copy of a book that was preserved for generations on library shelves. "
           "About Google Book Search ... at http : //books .google .com/ \n\n"
           "THE THEORY OF \nRELATIVITY was ex- \ntended to gravitation.\n\n\nSecond para- \ngraph here.\n")
    assert ia_clean(raw) == "THE THEORY OF RELATIVITY was extended to gravitation.\n\nSecond paragraph here."


def test_ia_chunks_cut_at_paragraphs_and_fold_a_short_tail():
    paras = [" ".join(["w"] * 600)] * 7 + ["tail words"]
    chunks = ia_chunks("\n\n".join(paras), words=2000)
    assert len(chunks) == 2 and chunks[-1].endswith("tail words")
    assert sum(len(c.split()) for c in chunks) == 7 * 600 + 2


def test_rows_ia_text_drops_after_cutoff(tmp_path, monkeypatch):
    import src.data.ingest_extra as ie

    monkeypatch.setitem(ie._IA_ITEMS, "physrev_ia", {
        "a": {"date": "1939-06-15", "precision": "day", "title": "Physical Review", "volume": "55"},
        "b": {"date": "1939-07-01", "precision": "day", "title": "Physical Review", "volume": "56"},
        "c": {"date": "1939-01-01", "precision": "year", "title": "Physical Review", "volume": ""}})
    p = tmp_path / "x_djvu.txt"
    p.write_text("Nuclear fission of uranium.\n\nA second paragraph.", encoding="utf-8")
    cutoff, stats = dt.date(1939, 6, 30), Counter()
    rows = [r for k in "abc" for r in ie.rows_ia_text(p, k, cutoff, stats, source="physrev_ia")]
    assert [r["article_id"] for r in rows] == ["physrev_ia_a_c000"] and stats["dropped_after_cutoff"] == 2
    assert rows[0]["date"] == "1939-06-15" and rows[0]["newspaper"] == "Physical Review"
