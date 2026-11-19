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
