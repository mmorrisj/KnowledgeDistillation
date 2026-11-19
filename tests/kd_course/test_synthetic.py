import pytest

from kd_course.data import LabelledSet
from kd_course.synthetic import (
    build_generation_prompt,
    dedupe,
    generate_examples,
    labelled_from_predictions,
    parse_generated,
    pick_examples,
    round_trip_filter,
)

from .test_teachers import FakeTeacher


def test_generation_prompt_names_the_type_the_count_and_the_examples():
    prompt = build_generation_prompt("card_arrival", 5, ["Where is my card?"])
    assert "card_arrival" in prompt and "5" in prompt and "Where is my card?" in prompt


@pytest.mark.parametrize(
    "reply",
    [
        "1. Where is my card?\n2. My card has not come.\n3. Card still missing",
        "1) Where is my card?\n2) My card has not come.\n3) Card still missing",
        "- Where is my card?\n- My card has not come.\n- Card still missing",
        "Here are the messages:\n\n"
        '1. "Where is my card?"\n2. "My card has not come."\n3. "Card still missing"\n',
    ],
)
def test_parse_generated_reads_common_list_formats(reply):
    assert parse_generated(reply) == [
        "Where is my card?",
        "My card has not come.",
        "Card still missing",
    ]


def test_parse_generated_ignores_blank_and_non_list_lines():
    assert parse_generated("Sure!\n\n1. One message\n\nHope that helps.") == ["One message"]


def test_generate_examples_labels_each_message_with_its_type():
    teacher = FakeTeacher(["1. a one\n2. a two", "1. b one\n2. b two"])
    generated = generate_examples(teacher, ["type_a", "type_b"], n_per_class=2, seed_examples={})
    assert generated.texts == ["a one", "a two", "b one", "b two"]
    assert generated.labels == [0, 0, 1, 1]


def test_generate_examples_keeps_at_most_the_requested_number():
    teacher = FakeTeacher(["1. a\n2. b\n3. c\n4. d"])
    assert len(generate_examples(teacher, ["t"], n_per_class=2, seed_examples={})) == 2


def test_dedupe_removes_repeats_ignoring_case_and_spacing():
    data = LabelledSet(
        texts=["Where is my card?", "where is  my card?", "Another"], labels=[0, 0, 1]
    )
    kept, removed = dedupe(data)
    assert kept.texts == ["Where is my card?", "Another"]
    assert kept.labels == [0, 1]
    assert removed == 1


def test_dedupe_removes_anything_that_matches_a_protected_text():
    data = LabelledSet(texts=["A test message", "Fresh one"], labels=[0, 1])
    kept, removed = dedupe(data, against=["a TEST message"])
    assert kept.texts == ["Fresh one"]
    assert removed == 1


def test_round_trip_filter_keeps_messages_the_teacher_labels_as_intended():
    data = LabelledSet(texts=["m1", "m2", "m3"], labels=[0, 1, 0])
    teacher = FakeTeacher(["cat", "cat", "nonsense"])  # m1 agrees, m2 disagrees, m3 unusable
    kept, dropped = round_trip_filter(teacher, data, ["cat", "dog"])
    assert kept.texts == ["m1"]
    assert dropped == 2


def test_labelled_from_predictions_drops_unusable_replies():
    kept, dropped = labelled_from_predictions(["a", "b", "c"], [1, None, 0])
    assert kept.texts == ["a", "c"]
    assert kept.labels == [1, 0]
    assert dropped == 1


def test_labelled_from_predictions_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length"):
        labelled_from_predictions(["a"], [1, 2])


def test_pick_examples_takes_the_same_number_from_every_class():
    gold = LabelledSet(texts=["a1", "a2", "b1", "b2"], labels=[0, 0, 1, 1])
    examples = pick_examples(gold, ["type_a", "type_b"], per_class=1, seed=0)
    assert sorted(label for _, label in examples) == ["type_a", "type_b"]
    assert all(text in gold.texts for text, _ in examples)


def test_pick_examples_is_repeatable():
    gold = LabelledSet(texts=["a1", "a2", "b1", "b2"], labels=[0, 0, 1, 1])
    assert pick_examples(gold, ["a", "b"], seed=3) == pick_examples(gold, ["a", "b"], seed=3)
