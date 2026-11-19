"""Build training data from a black-box teacher: label what you have, or write new examples."""

from __future__ import annotations

import random
import re
from collections import defaultdict

from kd_course.data import LabelledSet, _normalize
from kd_course.teachers import Teacher, classify_with_teacher


def pick_examples(
    gold: LabelledSet, label_names: list[str], per_class: int = 1, seed: int = 0
) -> list[tuple[str, str]]:
    """Example messages to show the teacher in its prompt, the same number for every type.

    Take them from the gold training set only. Never from validation or test.
    """
    by_class: dict[int, list[str]] = defaultdict(list)
    for text, label in zip(gold.texts, gold.labels, strict=True):
        by_class[label].append(text)
    rng = random.Random(seed)
    examples: list[tuple[str, str]] = []
    for label in sorted(by_class):
        for text in rng.sample(by_class[label], min(per_class, len(by_class[label]))):
            examples.append((text, label_names[label]))
    return examples


def labelled_from_predictions(
    texts: list[str], predictions: list[int | None]
) -> tuple[LabelledSet, int]:
    """Turn teacher answers into a training set. Texts with an unusable answer are dropped."""
    if len(texts) != len(predictions):
        raise ValueError(
            f"Texts and predictions differ in length: {len(texts)} and {len(predictions)}."
        )
    kept = LabelledSet()
    for text, prediction in zip(texts, predictions, strict=True):
        if prediction is not None:
            kept.texts.append(text)
            kept.labels.append(prediction)
    return kept, len(texts) - len(kept)


def build_generation_prompt(label: str, n: int, seed_examples: list[str]) -> str:
    """Ask the teacher to write new customer messages of one type."""
    lines = [
        f"Write {n} different messages that customers might send to a bank's support team.",
        f"Every message must be of this type: {label}",
        "Vary the wording, length and tone. Write the way real customers type.",
        "Reply with a numbered list and nothing else.",
    ]
    if seed_examples:
        lines += ["", "Examples of this type:"] + [f"- {example}" for example in seed_examples]
    return "\n".join(lines)


_LIST_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+(.*\S)\s*$")


def parse_generated(reply: str) -> list[str]:
    """The items of a numbered or bulleted list, without numbering or surrounding quotes."""
    items = []
    for line in reply.splitlines():
        match = _LIST_ITEM.match(line)
        if match:
            item = match.group(1).strip().strip("\"'“”").strip()
            if item:
                items.append(item)
    return items


def generate_examples(
    teacher: Teacher,
    label_names: list[str],
    n_per_class: int,
    seed_examples: dict[int, list[str]],
    max_new_tokens: int = 400,
) -> LabelledSet:
    """Have the teacher write up to `n_per_class` new messages for every type."""
    prompts = [
        build_generation_prompt(name, n_per_class, seed_examples.get(label, []))
        for label, name in enumerate(label_names)
    ]
    replies = teacher.complete(prompts, max_new_tokens=max_new_tokens)
    generated = LabelledSet()
    for label, reply in enumerate(replies):
        for text in parse_generated(reply)[:n_per_class]:
            generated.texts.append(text)
            generated.labels.append(label)
    return generated


def dedupe(data: LabelledSet, against: list[str] | None = None) -> tuple[LabelledSet, int]:
    """Drop repeats, and anything that matches a text in `against` (for example the test set).

    Matching ignores case and spacing.
    """
    seen = {_normalize(text) for text in against or []}
    kept = LabelledSet()
    for text, label in zip(data.texts, data.labels, strict=True):
        key = _normalize(text)
        if key not in seen:
            seen.add(key)
            kept.texts.append(text)
            kept.labels.append(label)
    return kept, len(data) - len(kept)


def round_trip_filter(
    teacher: Teacher,
    data: LabelledSet,
    label_names: list[str],
    examples: list[tuple[str, str]] | None = None,
) -> tuple[LabelledSet, int]:
    """Keep generated messages only if the teacher, asked fresh, gives the intended type."""
    predictions = classify_with_teacher(teacher, data.texts, label_names, examples)
    kept = LabelledSet()
    for text, label, prediction in zip(data.texts, data.labels, predictions, strict=True):
        if prediction == label:
            kept.texts.append(text)
            kept.labels.append(label)
    return kept, len(data) - len(kept)
