"""Read a language model's probability for every class, not just its top answer.

This is what makes distillation "white-box": it needs the model's weights, because
an API that returns text does not expose these numbers.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from kd_course.prompted import build_classification_prompt

_PLACEHOLDER = "\u0000MESSAGE\u0000"


def _split_prompt(tokenizer: Any, label_names: list[str], examples: Any) -> tuple[str, str, bool]:
    """The prompt text before and after the message, and whether a chat format is used."""
    prompt = build_classification_prompt(_PLACEHOLDER, label_names, examples)
    chat = bool(getattr(tokenizer, "chat_template", None))
    if chat:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    before, after = prompt.split(_PLACEHOLDER)
    return before, after, chat


def score_labels(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    label_names: list[str],
    examples: list[tuple[str, str]] | None = None,
    progress_every: int = 0,
) -> list[list[float]]:
    """For each text, the log-probability the model gives each label.

    Each row is normalized over the labels, so its probabilities add up to 1.

    How it stays fast: the instructions and the list of types are the same for every
    message, so the model reads them once and that work is reused. For each message
    the model then reads only the message, and all labels are scored in one batch.
    """
    import torch

    if not texts:
        return []
    before, after, chat = _split_prompt(tokenizer, label_names, examples)
    # Keep the space after "Message:" attached to the message, where a tokenizer expects it.
    stripped = before.rstrip(" ")
    lead = before[len(stripped) :]
    device = model.device

    end = tokenizer.eos_token_id
    label_lead = "" if chat else " "
    label_ids = [
        tokenizer(label_lead + name, add_special_tokens=False)["input_ids"] + [end]
        for name in label_names
    ]
    n_labels = len(label_ids)
    longest = max(len(ids) for ids in label_ids)
    # All labels in one batch, right-padded. Padding is masked out below.
    batch_ids = torch.zeros((n_labels, longest), dtype=torch.long, device=device)
    batch_mask = torch.zeros((n_labels, longest), dtype=torch.long, device=device)
    for row, ids in enumerate(label_ids):
        batch_ids[row, : len(ids)] = torch.tensor(ids, device=device)
        batch_mask[row, : len(ids)] = 1

    results: list[list[float]] = []
    with torch.inference_mode():
        prefix_ids = tokenizer(stripped, return_tensors="pt")["input_ids"].to(device)
        prefix_cache = model(prefix_ids, use_cache=True).past_key_values

        for index, text in enumerate(texts):
            message_ids = tokenizer(
                lead + text + after, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(device)
            out = model(message_ids, past_key_values=copy.deepcopy(prefix_cache), use_cache=True)
            # Probability of each label's first token, straight after the prompt.
            first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)

            cache = out.past_key_values
            cache.batch_repeat_interleave(n_labels)
            context = prefix_ids.shape[1] + message_ids.shape[1]
            mask = torch.cat(
                [torch.ones((n_labels, context), dtype=torch.long, device=device), batch_mask],
                dim=1,
            )
            logits = model(batch_ids, past_key_values=cache, attention_mask=mask).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)

            # A label's score is the sum over its tokens of log P(token | everything before it).
            scores = first[batch_ids[:, 0]]
            following = log_probs[:, :-1].gather(2, batch_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            scores = scores + (following * batch_mask[:, 1:]).sum(dim=1)
            results.append(torch.log_softmax(scores, dim=-1).tolist())

            if progress_every and (index + 1) % progress_every == 0:
                print(f"scored {index + 1} of {len(texts)}")
    return results


def _fingerprint(texts: list[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()


def save_scores(path: str | Path, texts: list[str], scores: list[list[float]]) -> Path:
    """Save teacher scores, tagged with the texts they belong to."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"texts": _fingerprint(texts), "scores": scores}))
    return path


def load_scores(path: str | Path, texts: list[str]) -> list[list[float]]:
    """Load saved teacher scores, checking they were computed for these exact texts."""
    saved = json.loads(Path(path).read_text())
    if saved["texts"] != _fingerprint(texts):
        raise ValueError(
            f"The scores in {path} were computed for different texts. "
            "Delete the file and score again."
        )
    return saved["scores"]


def top_labels(scores: list[list[float]]) -> list[int]:
    """The most likely class in each row. Ties go to the lower class number."""
    return [max(range(len(row)), key=row.__getitem__) for row in scores]


def score_or_load(
    path: str | Path,
    model: Any,
    tokenizer: Any,
    texts: list[str],
    label_names: list[str],
    progress_every: int = 0,
) -> list[list[float]]:
    """Score the texts and save the result, or read it back if it was saved before.

    Scoring is the slow step of white-box distillation, so it should run once.
    """
    path = Path(path)
    if path.is_file():
        return load_scores(path, texts)
    scores = score_labels(model, tokenizer, texts, label_names, progress_every=progress_every)
    save_scores(path, texts, scores)
    return scores
