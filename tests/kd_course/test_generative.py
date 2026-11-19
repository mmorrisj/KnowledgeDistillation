import copy

import pytest

torch = pytest.importorskip("torch")

from kd_course.data import LabelledSet  # noqa: E402
from kd_course.generative import (  # noqa: E402
    build_training_batch,
    format_prompt,
    predict_generative,
    train_generative_classifier,
)

LABELS = ["cat", "dog"]
TRAIN = LabelledSet(
    texts=[
        "the cat sat on the mat",
        "a cat sat",
        "the cat on a mat",
        "cat sat on floor",
        "the dog ran to the floor",
        "a dog ran",
        "dog ran to a mat",
        "the dog ran",
    ],
    labels=[0, 0, 0, 0, 1, 1, 1, 1],
)


def test_prompt_is_short_and_has_no_list_of_types():
    prompt = format_prompt("Where is my card?")
    assert "Where is my card?" in prompt
    assert prompt.endswith("Type:")
    assert len(prompt) < 60


def test_training_batch_only_scores_the_answer_tokens(tiny_lm):
    _, tokenizer = tiny_lm
    batch = build_training_batch(tokenizer, ["the cat sat", "a dog"], ["cat", "dog"], max_length=32)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    for row in range(2):
        scored = batch["labels"][row][batch["labels"][row] != -100].tolist()
        # the answer is one word here, followed by the end-of-text token
        assert len(scored) == 2
        assert scored[-1] == tokenizer.eos_token_id
    # padding is never scored and never attended to
    padded = batch["attention_mask"] == 0
    assert (batch["labels"][padded] == -100).all()


def test_training_lowers_the_loss_and_learns_the_toy_task(tiny_lm):
    model, tokenizer = copy.deepcopy(tiny_lm[0]), tiny_lm[1]
    history = train_generative_classifier(
        model, tokenizer, TRAIN, LABELS, epochs=40, learning_rate=5e-3, batch_size=4, seed=0
    )
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert predict_generative(model, tokenizer, TRAIN.texts, LABELS) == TRAIN.labels


def test_history_has_one_entry_per_epoch(tiny_lm):
    model, tokenizer = copy.deepcopy(tiny_lm[0]), tiny_lm[1]
    history = train_generative_classifier(model, tokenizer, TRAIN, LABELS, epochs=3, batch_size=4)
    assert [row["epoch"] for row in history] == [1, 2, 3]


def test_predict_returns_none_for_an_unusable_reply(tiny_lm):
    # An untrained random model will not write a label.
    model, tokenizer = tiny_lm
    predictions = predict_generative(model, tokenizer, ["the cat sat"], ["zebra", "giraffe"])
    assert predictions == [None]


def test_predict_handles_an_empty_list(tiny_lm):
    model, tokenizer = tiny_lm
    assert predict_generative(model, tokenizer, [], LABELS) == []


def test_training_rejects_an_empty_training_set(tiny_lm):
    model, tokenizer = tiny_lm
    with pytest.raises(ValueError, match="empty"):
        train_generative_classifier(model, tokenizer, LabelledSet(), LABELS)


def test_only_lora_weights_change_when_training_a_lora_model(tiny_lm):
    pytest.importorskip("peft")
    from kd_course.lora import add_lora

    model, tokenizer = copy.deepcopy(tiny_lm[0]), tiny_lm[1]
    lora_model = add_lora(model, rank=4, alpha=8, target_modules=["c_attn"])
    before = {n: p.detach().clone() for n, p in lora_model.named_parameters()}
    train_generative_classifier(
        lora_model, tokenizer, TRAIN, LABELS, epochs=2, learning_rate=1e-2, batch_size=4
    )
    changed = {n for n, p in lora_model.named_parameters() if not torch.equal(p, before[n])}
    assert changed and all("lora" in name for name in changed)


def test_measure_training_records_time_and_no_memory_on_cpu(tiny_lm):
    from kd_course.generative import measure_training

    with measure_training(tiny_lm[0]) as measured:
        sum(range(1000))
    assert measured.cost.seconds >= 0
    assert measured.cost.peak_memory_gb is None
