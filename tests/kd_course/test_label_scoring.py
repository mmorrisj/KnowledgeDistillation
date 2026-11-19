import math

import pytest

torch = pytest.importorskip("torch")

from kd_course.label_scoring import load_scores, save_scores, score_labels  # noqa: E402
from kd_course.prompted import build_classification_prompt  # noqa: E402

LABELS = ["cat", "dog", "cat sat", "mat floor"]
TEXTS = ["the cat sat on", "a dog ran to the floor", "mat"]


def brute_force(model, tokenizer, text, labels):
    """Score each label with one full forward pass per label. Slow but obviously right."""
    prompt = build_classification_prompt(text, labels)
    prompt_ids = tokenizer(prompt)["input_ids"]
    scores = []
    for label in labels:
        label_ids = tokenizer(" " + label, add_special_tokens=False)["input_ids"] + [
            tokenizer.eos_token_id
        ]
        ids = torch.tensor([prompt_ids + label_ids])
        with torch.inference_mode():
            log_probs = torch.log_softmax(model(ids).logits[0], dim=-1)
        start = len(prompt_ids)
        scores.append(
            sum(log_probs[start + j - 1, token].item() for j, token in enumerate(label_ids))
        )
    total = math.log(sum(math.exp(s) for s in scores))
    return [s - total for s in scores]


def test_scores_match_the_brute_force_calculation(tiny_lm):
    model, tokenizer = tiny_lm
    fast = score_labels(model, tokenizer, TEXTS, LABELS)
    for text, row in zip(TEXTS, fast, strict=True):
        assert row == pytest.approx(brute_force(model, tokenizer, text, LABELS), abs=1e-4)


def test_each_row_is_a_probability_distribution(tiny_lm):
    model, tokenizer = tiny_lm
    for row in score_labels(model, tokenizer, TEXTS, LABELS):
        assert len(row) == len(LABELS)
        assert sum(math.exp(v) for v in row) == pytest.approx(1.0, abs=1e-5)


def test_scoring_one_text_alone_gives_the_same_answer_as_in_a_list(tiny_lm):
    model, tokenizer = tiny_lm
    together = score_labels(model, tokenizer, TEXTS, LABELS)
    alone = score_labels(model, tokenizer, [TEXTS[1]], LABELS)
    assert alone[0] == pytest.approx(together[1], abs=1e-5)


def test_empty_input_gives_empty_output(tiny_lm):
    model, tokenizer = tiny_lm
    assert score_labels(model, tokenizer, [], LABELS) == []


def test_scores_round_trip_through_a_file(tmp_path):
    scores = [[-0.1, -2.5], [-1.0, -0.5]]
    path = save_scores(tmp_path / "teacher" / "pool.json", ["a", "b"], scores)
    assert load_scores(path, ["a", "b"]) == scores


def test_loading_scores_for_different_texts_is_an_error(tmp_path):
    path = save_scores(tmp_path / "pool.json", ["a", "b"], [[-0.1], [-0.2]])
    with pytest.raises(ValueError, match="different"):
        load_scores(path, ["a", "c"])


def test_top_labels_picks_the_most_likely_class_per_row():
    from kd_course.label_scoring import top_labels

    assert top_labels([[-0.1, -3.0], [-2.0, -0.2], [-1.0, -1.0]]) == [0, 1, 0]


def test_score_or_load_scores_once_and_then_reads_the_file(tiny_lm, tmp_path):
    from kd_course.label_scoring import score_or_load

    model, tokenizer = tiny_lm
    path = tmp_path / "scores.json"
    first = score_or_load(path, model, tokenizer, TEXTS, LABELS)
    assert path.is_file()
    # With no model, the second call can only succeed by reading the saved file.
    second = score_or_load(path, None, None, TEXTS, LABELS)
    assert second == first
