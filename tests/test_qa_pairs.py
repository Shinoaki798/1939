"""src.data.qa_pairs: column articles split into question-answer pairs, near-duplicate questions found, one copy
of a repeated question kept, and a cut passage that loses the answer dropped."""
from types import SimpleNamespace

import pandas as pd

from src.data.qa_pairs import cross_exam_share, encode, first_per_question, near_duplicates, pairs, prompt_text

COLUMN = ("You can get an answer to any question of fact by writing to the Question Editor.\n\n"
          " Q. Who was the star in the motion picture \"The Reckless Age?\"\n\n A, Reginald Denny.\n\n"
          " Q. Are tin cans made\nentirely of tin?\n\n A. No, they are made mostly of sheet steel.\n\n"
          " Q. Why?\n\n A. Too short a question.\n\n"
          " Q. This one has no question mark.\n\n A. Dropped.\n\n"
          " A. An answer without a question.")


def test_pairs_split_at_line_initial_markers():
    got = pairs(COLUMN, (8, 400), (2, 1000))
    assert got == [("Who was the star in the motion picture \"The Reckless Age?\"", "Reginald Denny."),
                   ("Are tin cans made entirely of tin?", "No, they are made mostly of sheet steel.")]
    assert pairs("Mid-line Q. is not a marker? A. no", (8, 400), (2, 1000)) == []
    assert pairs("O. Where is the Sphinx?\n A. In Egypt.", (8, 400), (2, 1000)) == [("Where is the Sphinx?", "In Egypt.")]
    assert pairs("Q. Who wrote the song?\n A. Wendell Hall. q. How much do airplanes cost?", (8, 400), (2, 1000)) == \
        [("Who wrote the song?", "Wendell Hall.")]


def test_cross_exam_share():
    court = ["And how much did you get from your doctor?", "She was drunk?", "Where is the Sphinx?"]
    assert cross_exam_share(court) == 2 / 3
    assert cross_exam_share(["Can you give me directions for feeding a monkey?", "What is the meaning of the name Tilie?"]) == 0


def test_near_duplicates_and_first_per_question():
    train = ["Are tin cans made entirely of tin?", "Who is the French Ambassador to the United States?"]
    ev = ["Are tin cans made entirely of tin ?", "Are tln cans made entirely of tin?", "How old is the Sphinx?"]
    assert near_duplicates(ev, train, 0.5).tolist() == [True, True, False]
    df = pd.DataFrame({"id": ["b", "a", "c"], "date": ["1925-01-02", "1925-01-01", "1926-01-01"],
                       "question": ["Are tin cans made of tin?", "ARE TIN CANS MADE OF TIN ?", "How old is the Sphinx?"]})
    assert sorted(first_per_question(df)["id"]) == ["a", "c"]


class CharTok:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[ord(ch) for ch in text])

    def decode(self, ids):
        return "".join(map(chr, ids))


def test_encode_prompt_answer_and_cut_passage():
    tok = CharTok()
    p, a = encode(tok, "Who?", "Denny")
    assert tok.decode(p) == prompt_text("Who?") == "Q. Who?\n\n A." and tok.decode(a) == " Denny"
    p, a = encode(tok, "Who?", "Denny", context="Denny starred.", context_tokens=100)
    assert tok.decode(p) == "Denny starred.\n\n Q. Who?\n\n A."
    assert encode(tok, "Who?", "Denny", context="It starred Denny.", context_tokens=8) is None
