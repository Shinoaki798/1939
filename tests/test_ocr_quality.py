import math

from src.data.ocr_quality import MIN_TOKENS, dta_text, score, tokens


def test_tokens_lowercase_long_s_letters_only():
    assert tokens("Die Uebereinſtimmung, 1938 a Thür-Schloß!") == ["die", "uebereinstimmung", "thür", "schloß"]


def test_score_hit_rate_and_short_text_is_nan():
    lex = {"die", "stadt"}
    text = " ".join(["die", "stadt", "xqzv", "stadt"] * 10)
    assert abs(score(text, lex) - 0.75) < 1e-9
    assert math.isnan(score("die stadt", lex)) and MIN_TOKENS > 2


def test_dta_text_joins_hyphenation_and_drops_running_heads():
    tei = ('<TEI><teiHeader><date type="publication">1873</date></teiHeader>'
           '<text><fw type="header">Seite 3</fw><p>Die Ver-<lb/>fassung des Lan&amp;des</p></text></TEI>')
    year, text = dta_text(tei)
    assert year == 1873
    assert tokens(text) == ["die", "verfassung", "des", "lan", "des"]
