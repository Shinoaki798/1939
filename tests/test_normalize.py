"""src.data.normalize: per-source fixes of training-only text; everything else untouched."""
from collections import Counter

from src.data.normalize import normalize


def test_europeana_repeated_words():
    s = Counter()
    x = "Die Torpedobootsflottille ist am 21. Mai eingetroffen. eingetroffen. Norwegen Norwegen mit 22,1"
    assert normalize(x, "europeana_newspapers_de", "de", s) == \
        "Die Torpedobootsflottille ist am 21. Mai eingetroffen. Norwegen mit 22,1"
    assert s["repeats_removed"] == 2
    assert normalize("das das ist so", "europeana_newspapers_de", "de") == "das das ist so"   # < 4 letters kept


def test_ddb_spaced_punctuation_and_quotes():
    x = "Das Signal stand auf Halt . „ Die Sache ist sehr einfach “ , erklärte der Beamte ( vgl. oben ) ."
    assert normalize(x, "ddb_newspapers_de", "de") == \
        "Das Signal stand auf Halt. „Die Sache ist sehr einfach“, erklärte der Beamte (vgl. oben)."
    assert normalize("bodies of troops would be collected . The author", "royal_society_corpus", "en") == \
        "bodies of troops would be collected. The author"


def test_oblique_hyphen():
    s = Counter()
    x = "in schwefligsauren Alka⸗ lien und Unterhaltungs⸗ und Reichssender, Ko⸗ Feindliche"
    assert normalize(x, "voelkischer_beobachter_de", "de", s) == \
        "in schwefligsauren Alkalien und Unterhaltungs- und Reichssender, Ko- Feindliche"
    assert s["oblique_joined"] == 1 and s["oblique_to_hyphen"] == 2


def test_congressional_record_commas():
    x = ("The Senate. as in Committee of the Whole. proceeded to consider the bill. It was read at 2 p. m. and "
         "referred, etc. and so on. Mr. Smith rose.")
    assert normalize(x, "congressional_record", "en") == \
        ("The Senate, as in Committee of the Whole, proceeded to consider the bill. It was read at 2 p. m. and "
         "referred, etc. and so on. Mr. Smith rose.")


def test_other_sources_untouched():
    x = "the gladi- ators fought . bravely. and well Norwegen Norwegen ⸗"
    for src, lang in (("american_stories", "en"), ("chronicling_america", "en"), ("caselaw_access_project", "en")):
        assert normalize(x, src, lang) == x
