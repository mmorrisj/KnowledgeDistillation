import pytest

np = pytest.importorskip("numpy")

from kd_course.metrics import (  # noqa: E402
    accuracy,
    bootstrap_ci,
    gap_closed,
    macro_f1,
    paired_bootstrap,
    per_class_f1,
)


def test_accuracy():
    assert accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75


def test_accuracy_counts_missing_predictions_as_wrong():
    assert accuracy([0, 1], [0, None]) == 0.5


def test_per_class_f1():
    # class 0: precision 2/3, recall 1 -> 0.8 ; class 1: precision 1, recall 1/2 -> 2/3
    scores = per_class_f1([0, 0, 1, 1], [0, 0, 0, 1], n_classes=2)
    assert scores == pytest.approx([0.8, 2 / 3])


def test_macro_f1_is_the_plain_average_over_classes():
    assert macro_f1([0, 0, 1, 1], [0, 0, 0, 1], n_classes=2) == pytest.approx((0.8 + 2 / 3) / 2)


def test_macro_f1_punishes_ignoring_a_rare_class():
    y_true = [0] * 95 + [1] * 5
    always_zero = [0] * 100
    assert accuracy(y_true, always_zero) == 0.95
    assert macro_f1(y_true, always_zero, n_classes=2) < 0.5


def test_a_class_never_predicted_and_never_present_scores_zero_not_nan():
    assert per_class_f1([0, 0], [0, 0], n_classes=2) == [1.0, 0.0]


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="length"):
        accuracy([0, 1], [0])


def test_metrics_reject_empty_input():
    with pytest.raises(ValueError, match="empty"):
        accuracy([], [])


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=400).tolist()
    y_pred = [t if rng.random() < 0.8 else 1 - t for t in y_true]
    low, high = bootstrap_ci(accuracy, y_true, y_pred, seed=0)
    assert low < accuracy(y_true, y_pred) < high
    assert 0.03 < high - low < 0.12


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(1)

    def width(n):
        y_true = rng.integers(0, 2, size=n).tolist()
        y_pred = [t if rng.random() < 0.8 else 1 - t for t in y_true]
        low, high = bootstrap_ci(accuracy, y_true, y_pred, seed=0)
        return high - low

    assert width(2000) < width(200)


def test_bootstrap_ci_is_repeatable_with_a_seed():
    y_true, y_pred = [0, 1] * 50, [0, 0, 1, 1] * 25
    assert bootstrap_ci(accuracy, y_true, y_pred, seed=3) == bootstrap_ci(
        accuracy, y_true, y_pred, seed=3
    )


def test_paired_bootstrap_detects_a_clear_improvement():
    y_true = [0, 1] * 250
    worse = [t if i % 5 else 1 - t for i, t in enumerate(y_true)]  # 80% right
    better = [t if i % 20 else 1 - t for i, t in enumerate(y_true)]  # 95% right
    result = paired_bootstrap(accuracy, y_true, worse, better, seed=0)
    assert result.difference == pytest.approx(0.15)
    assert result.low > 0
    assert result.probability_b_better > 0.99


def test_paired_bootstrap_is_unsure_about_identical_models():
    y_true = [0, 1] * 100
    predictions = [t if i % 5 else 1 - t for i, t in enumerate(y_true)]
    result = paired_bootstrap(accuracy, y_true, predictions, predictions, seed=0)
    assert result.difference == 0
    assert result.low <= 0 <= result.high


def test_gap_closed():
    assert gap_closed(baseline=0.60, distilled=0.72, teacher=0.80) == pytest.approx(0.6)


def test_gap_closed_can_exceed_one_or_go_negative():
    assert gap_closed(0.60, 0.85, 0.80) == pytest.approx(1.25)
    assert gap_closed(0.60, 0.55, 0.80) == pytest.approx(-0.25)


def test_gap_closed_is_undefined_when_the_teacher_is_no_better_than_the_baseline():
    assert gap_closed(0.80, 0.82, 0.80) is None
    assert gap_closed(0.80, 0.82, 0.75) is None


def test_summarize_reports_scores_intervals_and_unanswered_share():
    from kd_course.metrics import summarize

    y_true = [0, 1, 0, 1] * 50
    y_pred = [0, 1, 0, None] * 50
    summary = summarize(y_true, y_pred, n_classes=2, seed=0)
    assert summary["n"] == 200
    assert summary["accuracy"] == 0.75
    assert summary["unanswered"] == 0.25
    assert summary["accuracy_ci"][0] < 0.75 < summary["accuracy_ci"][1]
    assert summary["macro_f1_ci"][0] < summary["macro_f1"] < summary["macro_f1_ci"][1]
    assert all(isinstance(v, float) for v in summary["accuracy_ci"])
