"""A text-generation model as the student: it learns to write the type name.

Works the same for a fully fine-tuned model and for one with LoRA adapters, because
the optimizer is given only the weights that are allowed to change.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from kd_course.classifier import _schedule
from kd_course.data import LabelledSet
from kd_course.prompted import parse_label

IGNORE = -100


def format_prompt(text: str) -> str:
    """The student's prompt. A trained student knows the types, so the list is not needed."""
    return f"Message: {text}\nType:"


def _pad_id(tokenizer: Any) -> int:
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


def build_training_batch(
    tokenizer: Any, texts: list[str], answers: list[str], max_length: int = 96
) -> dict[str, Any]:
    """Token ids for prompt + answer, with the loss switched off everywhere but the answer."""
    import torch

    rows = []
    for text, answer in zip(texts, answers, strict=True):
        answer_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
        answer_ids = answer_ids + [tokenizer.eos_token_id]
        prompt_ids = tokenizer(format_prompt(text))["input_ids"][: max_length - len(answer_ids)]
        rows.append((prompt_ids, answer_ids))

    longest = max(len(p) + len(a) for p, a in rows)
    pad = _pad_id(tokenizer)
    input_ids = torch.full((len(rows), longest), pad, dtype=torch.long)
    labels = torch.full((len(rows), longest), IGNORE, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), longest), dtype=torch.long)
    for row, (prompt_ids, answer_ids) in enumerate(rows):
        n_prompt, n = len(prompt_ids), len(prompt_ids) + len(answer_ids)
        input_ids[row, :n] = torch.tensor(prompt_ids + answer_ids)
        labels[row, n_prompt:n] = torch.tensor(answer_ids)
        attention_mask[row, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train_generative_classifier(
    model: Any,
    tokenizer: Any,
    train: LabelledSet,
    label_names: list[str],
    epochs: int = 2,
    learning_rate: float = 2e-5,
    batch_size: int = 32,
    max_length: int = 96,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Train whichever weights of `model` are trainable to write the right type name.

    For full fine-tuning a learning rate near 2e-5 is typical; for LoRA, near 2e-4.
    Returns one row per epoch with the average training loss.
    """
    if len(train) == 0:
        raise ValueError("The training set is empty.")
    import torch

    torch.manual_seed(seed)
    device = model.device
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    steps_per_epoch = (len(train) + batch_size - 1) // batch_size
    scheduler = _schedule(optimizer, epochs * steps_per_epoch)
    shuffle = torch.Generator().manual_seed(seed)
    on_gpu = device.type == "cuda"
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=shuffle).tolist()
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            batch = build_training_batch(
                tokenizer,
                [train.texts[i] for i in rows],
                [label_names[train.labels[i]] for i in rows],
                max_length,
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=on_gpu):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item() * len(rows)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train)})

    model.eval()
    return history


def predict_generative(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    label_names: list[str],
    batch_size: int = 64,
    max_new_tokens: int = 12,
) -> list[int | None]:
    """Have the student write a type for each text. An unusable reply gives None."""
    import torch

    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    predictions: list[int | None] = []
    try:
        for start in range(0, len(texts), batch_size):
            prompts = [format_prompt(text) for text in texts[start : start + batch_size]]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=_pad_id(tokenizer),
                )
            replies = tokenizer.batch_decode(
                output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            predictions.extend(parse_label(reply, label_names) for reply in replies)
    finally:
        tokenizer.padding_side = original_side
    return predictions


@dataclass(frozen=True)
class TrainingCost:
    seconds: float
    peak_memory_gb: float | None


class measure_training:  # noqa: N801 - used as `with measure_training(model) as cost:`
    """Time a block of training and record the peak GPU memory it used."""

    def __init__(self, model: Any):
        self._device = model.device
        self.cost: TrainingCost | None = None

    def __enter__(self) -> measure_training:
        import torch

        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        import torch

        peak = None
        if self._device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(self._device) / 1024**3
        self.cost = TrainingCost(seconds=time.perf_counter() - self._start, peak_memory_gb=peak)
