import math

from src.data.ocr_quality import MIN_TOKENS, dta_text, score, tokens, word_share


def test_tokens_lowercase_long_s_letters_only():
    assert tokens("Die Uebereinſtimmung, 1938 a Thür-Schloß!") == ["die", "uebereinstimmung", "thür", "schloß"]


def test_score_hit_rate_and_short_text_is_nan():
    lex = {"die", "stadt"}
    text = " ".join(["die", "stadt", "xqzv", "stadt"] * 10)
    assert abs(score(text, lex) - 0.75) < 1e-9
    assert math.isnan(score("die stadt", lex)) and MIN_TOKENS > 2


def test_dta_text_joins_hyphenation_and_drops_running_heads():
    tei = ('<TEI><teiHeader><publicationStmt><date type="publication">2025-10-24T08:36:51Z</date>'
           '</publicationStmt><sourceDesc><biblFull><date type="publication">1873</date></biblFull>'
           '</sourceDesc></teiHeader>'
           '<text><fw type="header">Seite 3</fw><p>Die Ver-<lb/>fassung des Lan&amp;des</p></text></TEI>')
    year, text = dta_text(tei)
    assert year == 1873
    assert tokens(text) == ["die", "verfassung", "des", "lan", "des"]


def test_dta_year_falls_back_to_creation_date():
    tei = ('<TEI><teiHeader><date type="publication">2025-10-24T08:36:51Z</date>'
           '<date type="creation">1849</date></teiHeader><text><p>Die Stadt</p></text></TEI>')
    assert dta_text(tei)[0] == 1849


def test_word_share_penalises_tables_and_fragments():
    prose = "Die Verſammlung des Ortsvereins findet am Sonntag im „Deutſchen Hof“ ſtatt, alle Mit-glieder ſind eingeladen. " * 3
    table = "Laurahütte 98,25 bz 60,75 G 4½ 36430 — — 5 7 * 1919. 1909, da 0,72 0 8 12 . 6 . 1/1 2.2 " * 3
    assert word_share(prose) > 0.9
    assert word_share(table) < 0.3
    assert math.isnan(word_share("zu kurz"))


def test_word_share_ignores_spaced_punctuation():
    spaced = "Beerdigte den 12 . Juli Peter Müller , Bäckermeister , alt 73 Jahre . " * 4
    assert word_share(spaced) > 0.75


def test_gate_order_and_thresholds():
    from src.data.ocr_quality import gate

    lex = {"die", "stadt", "hat", "einen", "neuen", "bahnhof"}
    good = "Die Stadt hat einen neuen Bahnhof . " * 10
    assert gate(good, lex, "de")[0] is None
    assert gate("Die Stadt hat einen neuen Bahnhof", lex, "de")[0] == "short"
    assert gate("Dle Stabt hnt elnen ncuen Bahuhof " * 10, lex, "de")[0] == "hit_rate"
    table = "Die Stadt 12,50 hat 3 einen 44 neuen 7 Bahnhof 1919 " * 6
    assert gate(table, lex, "de")[0] == "word_share"


def test_clean_segments_drops_tables_keeps_prose():
    from src.data.ocr_quality import clean_segments

    prose = "Die Versammlung des Ortsvereins findet am Sonntag im Saale statt und alle sind herzlich eingeladen worden ."
    table = "Hindenburg 279 353 Marx 371 074 Thälmann 23 246 34 294 506 281 624 68 804 35 242 657 151 405 15 000 12 9"
    text = "\n\n".join([prose, prose, table, table, prose, prose, "Ende ."])
    out, n_in, n_kept = clean_segments(text)
    assert "Thälmann" not in out and out.count("Versammlung") == 4 and out.rstrip().endswith("Ende .")
    assert n_in == len(text.split()) and n_kept == len(out.split())


def test_variant_filter_drops_ocr_variants_unless_anchored():
    from collections import Counter

    from src.data.ocr_quality import _edits1, variant_filter

    assert {"a", "cb", "cab", "abc"} <= _edits1("ab", "abc") and "ab" not in _edits1("ab", "abc")
    tf = Counter({"the": 100000, "tbe": 900, "and": 80000, "aud": 700, "have": 50000, "hare": 300,
                  "ist": 40000, "ift": 600, "sein": 60000, "fein": 500, "tho": 3000})
    pool = {"the", "tbe", "aud", "hare", "ift", "fein", "tho"}
    kept, dropped = variant_filter(pool, tf, anchor={"the", "hare", "fein"}, ratio=50, min_tf=50)
    assert kept == {"the", "hare", "fein", "tho"}          # tho: "the" is only 33x more frequent
    assert dropped == [("tbe", 900, "the", 100000), ("aud", 700, "and", 80000), ("ift", 600, "ist", 40000)]


def test_scowl_words_reads_only_chosen_categories_and_sizes(tmp_path):
    import io
    import tarfile

    from src.data.ocr_quality import scowl_words

    path = tmp_path / "scowl-x.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        for name, words in [("english-words.10", "the\nhouse\n"), ("american-words.60", "color\n"),
                            ("british-words.10", "colour\n"), ("english-words.70", "zyzzyva\n"),
                            ("english-contractions.35", "don't\n")]:
            b = words.encode("latin-1")
            info = tarfile.TarInfo(f"scowl-x/final/{name}"); info.size = len(b)
            tf.addfile(info, io.BytesIO(b))
    words, used = scowl_words(path)
    assert words == {"the", "house", "color", "don"}
    assert used == ["american-words.60", "english-contractions.35", "english-words.10"]
