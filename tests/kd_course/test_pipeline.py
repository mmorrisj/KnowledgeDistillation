"""The Module 3 lab flow, end to end, on tiny CPU models. Checks the pieces fit together."""

import pytest

pytest.importorskip("torch")

from kd_course.classifier import predict, train_classifier  # noqa: E402
from kd_course.data import LabelledSet  # noqa: E402
from kd_course.distill import DistillSet, train_distilled_classifier  # noqa: E402
from kd_course.label_scoring import score_or_load, top_labels  # noqa: E402
from kd_course.metrics import accuracy, gap_closed, paired_bootstrap, summarize  # noqa: E402
from kd_course.results import format_results, load_results, require_result, save_result  # noqa: E402

from .test_classifier import TRAIN, tiny_encoder_dir  # noqa: E402, F401


def test_white_box_lab_flow(tiny_lm, tiny_encoder_dir, tmp_path):  # noqa: F811
    teacher, teacher_tokenizer = tiny_lm
    labels = ["cat", "dog"]
    gold = LabelledSet(texts=TRAIN.texts[:4], labels=TRAIN.labels[:4])
    pool = LabelledSet(texts=TRAIN.texts[4:], labels=TRAIN.labels[4:])
    test = TRAIN

    args = {"model": teacher, "tokenizer": teacher_tokenizer, "label_names": labels}
    test_scores = score_or_load(tmp_path / "teacher_scores_test.json", texts=test.texts, **args)
    gold_scores = score_or_load(tmp_path / "teacher_scores_gold.json", texts=gold.texts, **args)
    pool_scores = score_or_load(tmp_path / "teacher_scores_pool.json", texts=pool.texts, **args)

    teacher_pred = top_labels(test_scores)
    save_result(
        "teacher_scored",
        summarize(test.labels, teacher_pred, 2, n_resamples=50),
        teacher_pred,
        tmp_path,
    )

    baseline = train_classifier(tiny_encoder_dir, gold, gold, num_labels=2, epochs=1, device="cpu")
    baseline_pred = predict(baseline, test.texts)
    save_result(
        "baseline_gold_616",
        summarize(test.labels, baseline_pred, 2, n_resamples=50),
        baseline_pred,
        tmp_path,
    )

    hard = LabelledSet(texts=gold.texts + pool.texts, labels=gold.labels + top_labels(pool_scores))
    hard_pred = predict(
        train_classifier(tiny_encoder_dir, hard, gold, num_labels=2, epochs=1, device="cpu"),
        test.texts,
    )

    soft = DistillSet(
        texts=gold.texts + pool.texts,
        teacher_log_probs=gold_scores + pool_scores,
        labels=gold.labels + [None] * len(pool),
    )
    student = train_distilled_classifier(
        tiny_encoder_dir, soft, gold, num_labels=2, epochs=1, device="cpu"
    )
    soft_pred = predict(student, test.texts)
    soft_result = summarize(test.labels, soft_pred, 2, n_resamples=50)
    save_result("distilled_probabilities_T2", soft_result, soft_pred, tmp_path)

    saved_baseline = require_result("baseline_gold_616", tmp_path)
    table = format_results(load_results(tmp_path))
    assert "teacher_scored" in table and "distilled_probabilities_T2" in table
    gap = gap_closed(
        saved_baseline["metrics"]["accuracy"],
        soft_result["accuracy"],
        accuracy(test.labels, teacher_pred),
    )
    assert gap is None or isinstance(gap, float)
    comparison = paired_bootstrap(
        accuracy, test.labels, saved_baseline["predictions"], soft_pred, n_resamples=50
    )
    assert comparison.low <= comparison.difference <= comparison.high
    assert len(hard_pred) == len(soft_pred) == len(test)


def test_black_box_lab_flow(tiny_encoder_dir, tmp_path):  # noqa: F811
    from kd_course.metrics import teacher_verdict
    from kd_course.synthetic import (
        dedupe,
        generate_examples,
        labelled_from_predictions,
        pick_examples,
        round_trip_filter,
    )
    from kd_course.teachers import classify_with_teacher

    labels = ["lost_card", "exchange_rate"]

    class KeywordTeacher:
        """A stand-in black-box teacher: answers by keyword, and can write new messages."""

        def complete(self, prompts, max_new_tokens=16):
            replies = []
            for prompt in prompts:
                if prompt.startswith("Write"):
                    kind = "card lost" if "lost_card" in prompt else "euro rate"
                    replies.append(f"1. my {kind}\n2. the {kind}\n3. my {kind}")
                else:
                    message = prompt.rsplit("Message:", 1)[1]
                    replies.append("lost_card" if "card" in message else "exchange_rate")
            return replies

    teacher = KeywordTeacher()
    gold = LabelledSet(texts=TRAIN.texts[:2] + TRAIN.texts[4:6], labels=[0, 0, 1, 1])
    val = TRAIN
    pool_texts = TRAIN.texts[2:4] + TRAIN.texts[6:]

    examples = pick_examples(gold, labels, per_class=1, seed=0)
    val_pred = classify_with_teacher(teacher, val.texts, labels, examples)
    assert teacher_verdict(accuracy(val.labels, val_pred), 0.5, len(val)).split(":")[0] in {
        "better",
        "unclear",
        "worse",
    }

    pool_labelled, dropped = labelled_from_predictions(
        pool_texts, classify_with_teacher(teacher, pool_texts, labels, examples)
    )
    assert dropped == 0 and len(pool_labelled) == len(pool_texts)

    seeds = {
        label: [t for t, y in zip(gold.texts, gold.labels, strict=True) if y == label]
        for label in (0, 1)
    }
    generated = generate_examples(teacher, labels, n_per_class=3, seed_examples=seeds)
    generated, repeats = dedupe(generated, against=val.texts)
    generated, off_type = round_trip_filter(teacher, generated, labels, examples)
    assert repeats >= 2  # each type's reply repeats a message
    assert off_type == 0

    train = LabelledSet(
        texts=gold.texts + pool_labelled.texts + generated.texts,
        labels=gold.labels + pool_labelled.labels + generated.labels,
    )
    student = train_classifier(tiny_encoder_dir, train, val, num_labels=2, epochs=1, device="cpu")
    predictions = predict(student, val.texts)
    save_result(
        "blackbox_plus_synthetic",
        summarize(val.labels, predictions, 2, n_resamples=50),
        predictions,
        tmp_path,
    )
    assert "blackbox_plus_synthetic" in load_results(tmp_path)
