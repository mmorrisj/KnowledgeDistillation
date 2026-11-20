"""Student first, teacher as a fallback: send only the uncertain messages to the big model."""

from __future__ import annotations

from collections.abc import Sequence


def cascade_stats(
    confidences: Sequence[float],
    student_correct: Sequence[bool],
    teacher_correct: Sequence[bool],
    threshold: float,
) -> dict[str, float]:
    """Send every message the student is less than `threshold` sure of to the teacher.

    Returns the share of messages sent to the teacher and the accuracy of the combined system.
    """
    if not (len(confidences) == len(student_correct) == len(teacher_correct)):
        raise ValueError("confidences, student_correct and teacher_correct differ in length.")
    sent = [confidence < threshold for confidence in confidences]
    right = [
        teacher if to_teacher else student
        for to_teacher, student, teacher in zip(sent, student_correct, teacher_correct, strict=True)
    ]
    n = len(confidences)
    return {"threshold": threshold, "sent_to_teacher": sum(sent) / n, "accuracy": sum(right) / n}


def sweep_thresholds(
    confidences: Sequence[float],
    student_correct: Sequence[bool],
    teacher_correct: Sequence[bool],
    thresholds: Sequence[float],
) -> list[dict[str, float]]:
    """`cascade_stats` at each threshold, in the order given."""
    return [
        cascade_stats(confidences, student_correct, teacher_correct, threshold)
        for threshold in thresholds
    ]
