from pathlib import Path

from src.data.download import load_file_table, parse_years

REPO = Path(__file__).resolve().parents[1]


def test_parse_years_ranges_order_and_dedup():
    assert parse_years("1923-1925,1922-1920") == [1923, 1924, 1925, 1922, 1921, 1920]
    assert parse_years("1939, 1939,1938") == [1939, 1938]


def test_file_table_covers_study_years_with_sha256():
    table = load_file_table(REPO / "config" / "american_stories_files.tsv")
    for year in range(1900, 1956):
        spec = table[year]
        assert spec["file"] == f"faro_{year}.tar.gz"
        assert spec["bytes"] > 0
        assert len(spec["sha256"]) == 64
