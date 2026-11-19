"""Use a text-generation model as a classifier by asking it in a prompt."""

from __future__ import annotations

import re
from typing import Any

INSTRUCTIONS = (
    "You sort customer messages for a bank into types. "
    "Reply with exactly one type from the list, copied exactly, and nothing else."
)


def build_classification_prompt(
    text: str, label_names: list[str], examples: list[tuple[str, str]] | None = None
) -> str:
    """The prompt for one message. `examples` are (message, type) pairs shown before it."""
    parts = [INSTRUCTIONS, "", "Types:", *label_names, ""]
    for example_text, example_label in examples or []:
        parts += [f"Message: {example_text}", f"Type: {example_label}", ""]
    parts += [f"Message: {text}", "Type:"]
    return "\n".join(parts)


def parse_label(reply: str, label_names: list[str]) -> int | None:
    """Find the one label a reply names. Returns None if it names none, or more than one.

    Tolerates case, surrounding quotes or punctuation, and spaces in place of underscores.
    """
    lowered = reply.strip().lower()
    candidates = {lowered, re.sub(r"\s+", "_", lowered.strip(" \t\n\"'`.,:;!"))}
    # Compare on letters, digits and underscores only: one Banking77 label ends in "?".
    names = [re.sub(r"[^a-z0-9_]", "", name.lower()) for name in label_names]
    matched: set[int] = set()
    for index, name in enumerate(names):
        pattern = rf"(?<![a-z0-9_]){re.escape(name)}(?![a-z0-9_])"
        if any(re.search(pattern, candidate) for candidate in candidates):
            matched.add(index)
    # "card_arrival" also contains "card": keep only the longest of nested matches.
    longest = {
        i
        for i in matched
        if not any(i != j and label_names[i].lower() in label_names[j].lower() for j in matched)
    }
    return longest.pop() if len(longest) == 1 else None


def classify_with_llm(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    label_names: list[str],
    examples: list[tuple[str, str]] | None = None,
    batch_size: int = 16,
    max_new_tokens: int = 16,
) -> list[int | None]:
    """Classify each text by prompting a local model. An unusable reply gives None."""
    from kd_course.teachers import LocalTeacher, classify_with_teacher

    teacher = LocalTeacher(model, tokenizer, batch_size=batch_size)
    return classify_with_teacher(teacher, texts, label_names, examples, max_new_tokens)
