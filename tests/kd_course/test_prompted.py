import pytest

from kd_course.prompted import build_classification_prompt, parse_label

LABELS = ["card_arrival", "lost_or_stolen_card", "exchange_rate", "Refund_not_showing_up"]


def test_prompt_lists_every_label_and_the_message():
    prompt = build_classification_prompt("Where is my card?", LABELS)
    for label in LABELS:
        assert label in prompt
    assert "Where is my card?" in prompt


def test_prompt_includes_examples_when_given():
    prompt = build_classification_prompt(
        "Where is my card?", LABELS, examples=[("I lost my card", "lost_or_stolen_card")]
    )
    assert "I lost my card" in prompt
    assert prompt.index("I lost my card") < prompt.index("Where is my card?")


@pytest.mark.parametrize(
    "reply",
    [
        "card_arrival",
        " card_arrival\n",
        "Card_Arrival",
        "`card_arrival`",
        '"card_arrival".',
        "card arrival",
    ],
)
def test_parse_label_accepts_small_format_differences(reply):
    assert parse_label(reply, LABELS) == 0


def test_parse_label_matches_labels_with_capitals():
    assert parse_label("refund_not_showing_up", LABELS) == 3


def test_parse_label_finds_a_label_inside_a_sentence():
    assert parse_label("The type is exchange_rate.", LABELS) == 2


def test_parse_label_prefers_the_longest_match():
    labels = ["card", "card_arrival"]
    assert parse_label("card_arrival", labels) == 1


@pytest.mark.parametrize("reply", ["", "I am not sure", "card_arrival or exchange_rate", "42"])
def test_parse_label_returns_none_when_no_single_label_is_named(reply):
    assert parse_label(reply, LABELS) is None


def test_classify_with_llm_returns_one_result_per_text(tiny_lm):
    from kd_course.prompted import classify_with_llm

    model, tokenizer = tiny_lm
    texts = ["the cat sat", "a dog ran", "the mat"]
    predictions = classify_with_llm(
        model, tokenizer, texts, ["cat", "dog"], batch_size=2, max_new_tokens=4
    )
    assert len(predictions) == 3
    assert all(p is None or p in (0, 1) for p in predictions)


def test_classify_with_llm_handles_an_empty_list(tiny_lm):
    from kd_course.prompted import classify_with_llm

    model, tokenizer = tiny_lm
    assert classify_with_llm(model, tokenizer, [], ["cat", "dog"]) == []
