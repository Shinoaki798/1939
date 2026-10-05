import math

from src.data.filters import english_titles, is_native_english
from src.data.ingest import guess_lang, parse_scan_name, valid_date


def test_is_native_english():
    assert is_native_english("en", 0.6)
    assert not is_native_english("en", 0.2)
    assert not is_native_english("es", 0.0)
    assert not is_native_english("und", math.nan)


def test_english_titles_drops_foreign_title_whole():
    rows = [("snEN", 1923, "en")] * 9 + [("snEN", 1923, "cs")] + \
           [("snES", 1923, "es")] * 9 + [("snES", 1923, "en")]
    assert english_titles(rows) == {("snEN", 1923)}


def test_parse_scan_name_recovers_lccn_and_edition():
    name = "faro_1923/1923-03-08_p1_sn92067236_00514150266_1923030801_0256.json"
    assert parse_scan_name(name) == ("1923-03-08", "p1", "01", "sn92067236")
    assert parse_scan_name("faro_1923/README.txt") is None


def test_valid_date_rejects_impossible_and_wrong_year():
    assert valid_date("1939-06-30", 1939)
    assert not valid_date("1939-02-30", 1939)
    assert not valid_date("1938-12-31", 1939)


def test_guess_lang_separates_english_and_spanish():
    en = "The council met on Monday and the mayor said that the bridge was to be built by the city."
    es = "El consejo de la ciudad se reunio el lunes y el alcalde dijo que el puente para los vecinos."
    assert guess_lang(en)[0] == "en"
    assert guess_lang(es)[0] == "es"
