"""Simple, checkable measures for short generated text."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def word_count(text: str) -> int:
    return len(text.split())


def within_word_limit(text: str, limit: int) -> bool:
    """True if the text is not empty and has at most `limit` words."""
    return 0 < word_count(text) <= limit


def token_f1(prediction: str, reference: str) -> float:
    """Word overlap between two texts, from 0 to 1. Ignores case and punctuation.

    Combines precision (how much of the prediction is in the reference) and recall
    (how much of the reference is in the prediction).
    """
    predicted, expected = _words(prediction), _words(reference)
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def mean(values: Iterable[float | bool]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
