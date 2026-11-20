import pytest

from kd_course.text_metrics import mean, token_f1, within_word_limit, word_count
from kd_course.titles import (
    MAX_TITLE_WORDS,
    build_judge_prompt,
    build_title_prompt,
    clean_title,
    generate_titles,
    judge_titles,
    parse_verdict,
    student_prompt,
)

from .test_teachers import FakeTeacher


def test_word_count_and_limit():
    assert word_count("Card has not arrived") == 4
    assert word_count("  ") == 0
    assert within_word_limit("one two three", 3)
    assert not within_word_limit("one two three four", 3)
    assert not within_word_limit("", 3)  # an empty title never passes


def test_token_f1_is_one_for_identical_text_ignoring_case_and_punctuation():
    assert token_f1("Card not arrived.", "card not arrived") == 1.0


def test_token_f1_matches_a_hand_calculation():
    # prediction: card lost today (3) ; reference: card was lost (3) ; overlap: card, lost (2)
    # precision 2/3, recall 2/3, F1 2/3
    assert token_f1("card lost today", "card was lost") == pytest.approx(2 / 3)


def test_token_f1_is_zero_with_no_overlap_or_empty_text():
    assert token_f1("exchange rate", "card arrival") == 0.0
    assert token_f1("", "card arrival") == 0.0
    assert token_f1("card", "") == 0.0


def test_token_f1_counts_repeated_words_only_as_often_as_they_match():
    # prediction has "card" twice, reference once: overlap is 1
    assert token_f1("card card", "card") == pytest.approx(2 * (1 / 2) * 1 / (1 / 2 + 1))


def test_mean_of_nothing_is_zero():
    assert mean([]) == 0.0
    assert mean([True, False, True, True]) == 0.75


def test_title_prompt_states_the_word_limit_and_the_message():
    prompt = build_title_prompt("Where is my card?")
    assert str(MAX_TITLE_WORDS) in prompt and "Where is my card?" in prompt


def test_student_prompt_is_short():
    assert student_prompt("Where is my card?") == "Message: Where is my card?\nTitle:\n"


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Card not arrived", "Card not arrived"),
        ('"Card not arrived"', "Card not arrived"),
        ("Title: Card not arrived", "Card not arrived"),
        ("Card not arrived\n\nLet me know if you need more.", "Card not arrived"),
        ("  \n Card not arrived. ", "Card not arrived"),
        ("", ""),
    ],
)
def test_clean_title_keeps_the_first_line_without_decoration(reply, expected):
    assert clean_title(reply) == expected


def test_generate_titles_cleans_each_reply():
    teacher = FakeTeacher(['"New card missing"', "Title: Charged twice\nextra"])
    assert generate_titles(teacher, ["a", "b"]) == ["New card missing", "Charged twice"]
    assert "a" in teacher.prompts[0]


def test_judge_prompt_shows_the_message_and_the_title():
    prompt = build_judge_prompt("Where is my card?", "Card not arrived")
    assert "Where is my card?" in prompt and "Card not arrived" in prompt


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("yes", True),
        ("Yes.", True),
        (" NO ", False),
        ("No, it misses the point", False),
        ("maybe", None),
        ("", None),
    ],
)
def test_parse_verdict(reply, expected):
    assert parse_verdict(reply) is expected


def test_judge_titles_rejects_an_empty_title_without_asking():
    teacher = FakeTeacher(["yes"])
    verdicts = judge_titles(teacher, ["m1", "m2"], ["Good title", ""])
    assert verdicts == [True, False]
    assert len(teacher.prompts) == 1


def test_judge_titles_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length"):
        judge_titles(FakeTeacher([]), ["m1"], [])
