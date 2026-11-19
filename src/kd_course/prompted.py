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
    matched: set[int] = set()
    for index, name in enumerate(label_names):
        pattern = rf"(?<![a-z0-9_]){re.escape(name.lower())}(?![a-z0-9_])"
        if any(re.search(pattern, candidate) for candidate in candidates):
            matched.add(index)
    # "card_arrival" also contains "card": keep only the longest of nested matches.
    longest = {
        i
        for i in matched
        if not any(i != j and label_names[i].lower() in label_names[j].lower() for j in matched)
    }
    return longest.pop() if len(longest) == 1 else None


def _as_model_input(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    return prompt


def classify_with_llm(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    label_names: list[str],
    examples: list[tuple[str, str]] | None = None,
    batch_size: int = 16,
    max_new_tokens: int = 16,
) -> list[int | None]:
    """Classify each text by prompting. A reply that names no single label gives None."""
    import torch

    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # generation continues from the right-hand end
    predictions: list[int | None] = []
    try:
        for start in range(0, len(texts), batch_size):
            prompts = [
                _as_model_input(tokenizer, build_classification_prompt(text, label_names, examples))
                for text in texts[start : start + batch_size]
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id
                )
            new_tokens = output[:, inputs["input_ids"].shape[1] :]
            replies = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            predictions.extend(parse_label(reply, label_names) for reply in replies)
    finally:
        tokenizer.padding_side = original_side
    return predictions
