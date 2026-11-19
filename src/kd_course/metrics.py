"""Scores for classification, and ways to tell whether a difference between two models is real."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

Prediction = int | None
Metric = Callable[[Sequence[int], Sequence[Prediction]], float]


def _check(y_true: Sequence[int], y_pred: Sequence[Prediction]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Labels and predictions differ in length: {len(y_true)} and {len(y_pred)}."
        )
    if len(y_true) == 0:
        raise ValueError("Cannot score an empty set of predictions.")


def accuracy(y_true: Sequence[int], y_pred: Sequence[Prediction]) -> float:
    """Share of predictions that are right. A missing prediction (None) counts as wrong."""
    _check(y_true, y_pred)
    return sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)


def per_class_f1(
    y_true: Sequence[int], y_pred: Sequence[Prediction], n_classes: int
) -> list[float]:
    """F1 for each class. A class with no true and no predicted examples scores 0."""
    _check(y_true, y_pred)
    true_positive = [0] * n_classes
    predicted = [0] * n_classes
    actual = [0] * n_classes
    for t, p in zip(y_true, y_pred, strict=True):
        actual[t] += 1
        if p is not None:
            predicted[p] += 1
            if p == t:
                true_positive[t] += 1
    return [
        2 * tp / (pred + act) if (pred + act) else 0.0
        for tp, pred, act in zip(true_positive, predicted, actual, strict=True)
    ]


def macro_f1(y_true: Sequence[int], y_pred: Sequence[Prediction], n_classes: int) -> float:
    """The plain average of per-class F1, so every class counts the same however rare."""
    scores = per_class_f1(y_true, y_pred, n_classes)
    return sum(scores) / n_classes


def _resample_indices(n: int, n_resamples: int, seed: int):
    import numpy as np

    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


def bootstrap_ci(
    metric: Metric,
    y_true: Sequence[int],
    y_pred: Sequence[Prediction],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """A confidence interval for a score, by resampling the test set with replacement."""
    import numpy as np

    _check(y_true, y_pred)
    scores = [
        metric([y_true[i] for i in row], [y_pred[i] for i in row])
        for row in _resample_indices(len(y_true), n_resamples, seed)
    ]
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(scores, [tail, 100 - tail])
    return float(low), float(high)


@dataclass(frozen=True)
class PairedResult:
    difference: float  # score of B minus score of A on the full test set
    low: float
    high: float
    probability_b_better: float


def paired_bootstrap(
    metric: Metric,
    y_true: Sequence[int],
    pred_a: Sequence[Prediction],
    pred_b: Sequence[Prediction],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedResult:
    """Compare two models on the same test messages.

    Each resample scores both models on the same messages, so the result reflects
    how they differ, not how hard a particular sample happened to be.
    """
    import numpy as np

    _check(y_true, pred_a)
    _check(y_true, pred_b)
    differences = []
    for row in _resample_indices(len(y_true), n_resamples, seed):
        truth = [y_true[i] for i in row]
        differences.append(
            metric(truth, [pred_b[i] for i in row]) - metric(truth, [pred_a[i] for i in row])
        )
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(differences, [tail, 100 - tail])
    return PairedResult(
        difference=metric(y_true, pred_b) - metric(y_true, pred_a),
        low=float(low),
        high=float(high),
        probability_b_better=float(np.mean(np.array(differences) > 0)),
    )


def gap_closed(baseline: float, distilled: float, teacher: float) -> float | None:
    """How much of the distance from the baseline to the teacher the distilled student covered.

    1.0 means the student matched the teacher; 0 means distillation did nothing.
    Returns None when the teacher is no better than the baseline: there is no gap to close.
    """
    if teacher <= baseline:
        return None
    return (distilled - baseline) / (teacher - baseline)


def summarize(
    y_true: Sequence[int],
    y_pred: Sequence[Prediction],
    n_classes: int,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, float | int | list[float]]:
    """Accuracy and macro-F1 with their 95% intervals, ready to save or print."""

    def macro(truth: Sequence[int], predicted: Sequence[Prediction]) -> float:
        return macro_f1(truth, predicted, n_classes)

    return {
        "n": len(y_true),
        "accuracy": accuracy(y_true, y_pred),
        "accuracy_ci": list(bootstrap_ci(accuracy, y_true, y_pred, n_resamples, seed=seed)),
        "macro_f1": macro(y_true, y_pred),
        "macro_f1_ci": list(bootstrap_ci(macro, y_true, y_pred, n_resamples, seed=seed)),
        "unanswered": sum(p is None for p in y_pred) / len(y_pred),
    }


def teacher_verdict(teacher: float, baseline: float | None, n: int) -> str:
    """Is this teacher worth learning from? Compares two accuracies measured on n messages.

    Returns a sentence starting with "better", "unclear", "worse" or "unknown".
    Uses a simple 95% margin for the difference, treating the two scores as independent.
    """
    if baseline is None:
        return "unknown: no baseline validation score was saved, so run Lab 2 again to compare."
    margin = 1.96 * ((teacher * (1 - teacher) + baseline * (1 - baseline)) / n) ** 0.5
    difference = teacher - baseline
    detail = f"teacher {teacher:.1%}, baseline {baseline:.1%}, margin {margin:.1%}"
    if difference > margin:
        return f"better: the teacher clearly beats the baseline ({detail})."
    if difference < -margin:
        return f"worse: the teacher is clearly behind the baseline ({detail})."
    return f"unclear: the two cannot be told apart on this many messages ({detail})."
