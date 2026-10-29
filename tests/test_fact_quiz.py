"""src.eval.fact_quiz: the frozen quiz keeps its construction rules, continuations are formed the same way for every
candidate, and the choice is the candidate with the lowest bits per byte."""
import numpy as np
import pandas as pd
import pytest

from src.data.download import repo_path
from src.eval.fact_quiz import check, choices, continuation


def test_frozen_quiz_follows_the_rules():
    df = pd.read_csv(repo_path("probes/fact_quiz_1939.csv"), dtype=str)
    check(df)
    assert (df["tier"] != "post").sum() >= 100 and set(df["tier"]) == {"standing", "recent", "post"}


def test_check_rejects_a_misdated_or_repeated_item():
    row = {"id": "X", "tier": "recent", "as_of": "1938-07-01", "question": "Who?", "stem": "It was",
           "answer": "A", "distractor_1": "B", "distractor_2": "C", "distractor_3": "D"}
    check(pd.DataFrame([row]))
    with pytest.raises(SystemExit):
        check(pd.DataFrame([{**row, "as_of": "1939-07-15"}]))                # embargo window: no tier
    with pytest.raises(SystemExit):
        check(pd.DataFrame([{**row, "distractor_3": "A"}]))


def test_continuations_and_choice():
    assert continuation("the Pacific Ocean", "qa") == " The Pacific Ocean."
    assert continuation("the Pacific Ocean", "stem") == " the Pacific Ocean."
    assert continuation("Lakehurst, N. J.", "qa") == " Lakehurst, N. J."
    bits, nbytes = np.array([[10.0, 4.0, 9.0, 9.0], [2.0, 9.0, 9.0, 9.0]]), np.array([[20, 4, 9, 9], [4, 9, 9, 9]])
    assert choices(bits, nbytes).tolist() == [0, 0]                        # 0.5 bpb beats 1.0 despite more bits
