"""The Module 6 task: write a short ticket title for a customer message."""

from __future__ import annotations

import re
from typing import Any

from kd_course.teachers import Teacher

MAX_TITLE_WORDS = 8


def build_title_prompt(message: str) -> str:
    """The teacher's prompt: full instructions every time."""
    return (
        "You write titles for customer support tickets at a bank.\n"
        f"Write a title of at most {MAX_TITLE_WORDS} words that says what the customer needs.\n"
        "Be specific and neutral. Do not use the customer's exact sentence.\n"
        "Reply with the title only.\n\n"
        f"Customer message: {message}"
    )


def student_prompt(message: str) -> str:
    """The student's prompt. A trained student needs no instructions."""
    return f"Message: {message}\nTitle:\n"


def clean_title(reply: str) -> str:
    """The first non-empty line of a reply, without a 'Title:' lead-in, quotes or a final stop."""
    for line in reply.splitlines():
        line = line.strip()
        if line:
            line = re.sub(r"^title\s*:\s*", "", line, flags=re.IGNORECASE)
            return line.strip().strip("\"'“”").strip().rstrip(".").strip()
    return ""


def generate_titles(teacher: Teacher, messages: list[str], max_new_tokens: int = 32) -> list[str]:
    """Ask a teacher for a title for each message."""
    prompts = [build_title_prompt(message) for message in messages]
    return [
        clean_title(reply) for reply in teacher.complete(prompts, max_new_tokens=max_new_tokens)
    ]


def teacher_chat_prompts(tokenizer: Any, messages: list[str]) -> list[str]:
    """The teacher's prompts in its chat format, ready for reading its token probabilities."""
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": build_title_prompt(message)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]


def build_judge_prompt(message: str, title: str) -> str:
    return (
        "You are checking titles written for customer support tickets at a bank.\n"
        "A good title correctly says what the customer needs.\n"
        "A bad title names the wrong need, is too vague to act on, or is not a title at all.\n\n"
        f"Customer message: {message}\n"
        f"Title: {title}\n\n"
        "Is this a good title? Answer yes or no."
    )


def parse_verdict(reply: str) -> bool | None:
    """True for yes, False for no, None if the reply starts with neither."""
    match = re.match(r"\s*(yes|no)\b", reply, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower() == "yes"


def judge_titles(teacher: Teacher, messages: list[str], titles: list[str]) -> list[bool | None]:
    """Have a model judge each title. An empty title is rejected without asking."""
    if len(messages) != len(titles):
        raise ValueError(
            f"Messages and titles differ in length: {len(messages)} and {len(titles)}."
        )
    to_ask = [i for i, title in enumerate(titles) if title.strip()]
    prompts = [build_judge_prompt(messages[i], titles[i]) for i in to_ask]
    replies = teacher.complete(prompts, max_new_tokens=4) if prompts else []
    verdicts: list[bool | None] = [False] * len(titles)
    for index, reply in zip(to_ask, replies, strict=True):
        verdicts[index] = parse_verdict(reply)
    return verdicts


def usable_for_training(message: str, title: str) -> bool:
    """A title is kept as a training example only if it passes every rule a program can check."""
    from kd_course.text_metrics import within_word_limit

    def plain(text: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()

    return within_word_limit(title, MAX_TITLE_WORDS) and plain(title) != plain(message)


def score_titles(titles: list[str], references: list[str]) -> dict[str, float]:
    """Share of titles within the word limit, and average word overlap with the reference titles."""
    from kd_course.text_metrics import mean, token_f1, within_word_limit

    if len(titles) != len(references):
        raise ValueError(
            f"Titles and references differ in length: {len(titles)} and {len(references)}."
        )
    return {
        "n": len(titles),
        "within_limit": mean(within_word_limit(title, MAX_TITLE_WORDS) for title in titles),
        "overlap_with_teacher": mean(
            token_f1(title, reference) for title, reference in zip(titles, references, strict=True)
        ),
    }


def acceptance(verdicts: list[bool | None]) -> dict[str, float]:
    """Share of titles the judge accepted. A verdict that is neither yes nor no does not count."""
    from kd_course.text_metrics import mean

    return {
        "accepted": mean(verdict is True for verdict in verdicts),
        "unanswered": mean(verdict is None for verdict in verdicts),
    }
