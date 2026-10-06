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
