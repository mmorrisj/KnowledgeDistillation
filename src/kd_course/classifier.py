"""Train a small encoder as a classifier, with a training loop you can read end to end."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kd_course.data import LabelledSet
from kd_course.models import resolve_model_path

STUDENT_ENCODER_ID = "answerdotai/ModernBERT-base"


@dataclass
class TrainedClassifier:
    model: Any
    tokenizer: Any
    max_length: int
    history: list[dict[str, float]] = field(default_factory=list)


def _batches(tokenizer: Any, texts: list[str], batch_size: int, max_length: int, device: Any):
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        yield tokenizer(
            chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)


def _schedule(optimizer: Any, total_steps: int) -> Any:
    """Learning rate climbs for the first tenth of training, then falls steadily to zero."""
    import torch

    warmup = max(1, total_steps // 10)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            (step + 1) / warmup
            if step < warmup
            else max(0.0, (total_steps - step) / max(1, total_steps - warmup))
        ),
    )


def predict(trained: TrainedClassifier, texts: list[str], batch_size: int = 64) -> list[int]:
    """The most likely label for each text."""
    import torch

    model = trained.model.eval()
    predictions: list[int] = []
    with torch.inference_mode():
        for batch in _batches(
            trained.tokenizer, texts, batch_size, trained.max_length, model.device
        ):
            predictions.extend(model(**batch).logits.argmax(dim=-1).tolist())
    return predictions


def train_classifier(
    model_id: str,
    train: LabelledSet,
    val: LabelledSet,
    num_labels: int,
    epochs: int = 10,
    learning_rate: float = 5e-5,
    batch_size: int = 16,
    max_length: int = 64,
    device: str = "cuda",
    seed: int = 0,
) -> TrainedClassifier:
    """Fine-tune every weight of an encoder on `train`, keeping the epoch that is best on `val`."""
    if len(train) == 0:
        raise ValueError("The training set is empty.")
    if any(not 0 <= label < num_labels for label in train.labels):
        raise ValueError(f"Every label must be between 0 and {num_labels - 1}.")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    path = resolve_model_path(model_id)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=num_labels)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    steps_per_epoch = (len(train) + batch_size - 1) // batch_size
    scheduler = _schedule(optimizer, epochs * steps_per_epoch)
    shuffle = torch.Generator().manual_seed(seed)
    trained = TrainedClassifier(model=model, tokenizer=tokenizer, max_length=max_length)
    best_accuracy, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=shuffle).tolist()
        texts = [train.texts[i] for i in order]
        labels = torch.tensor([train.labels[i] for i in order], device=device)
        total_loss = 0.0
        batches = _batches(tokenizer, texts, batch_size, max_length, device)
        for step, batch in enumerate(batches):
            target = labels[step * batch_size : (step + 1) * batch_size]
            logits = model(**batch).logits  # 1. predict
            loss = torch.nn.functional.cross_entropy(logits.float(), target)  # 2. compare
            loss.backward()  # 3. work out how each weight should move
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  # 4. move the weights
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item() * len(target)

        val_predictions = predict(trained, val.texts)
        val_accuracy = sum(p == t for p, t in zip(val_predictions, val.labels, strict=True)) / max(
            1, len(val)
        )
        trained.history.append(
            {"epoch": epoch, "train_loss": total_loss / len(train), "val_accuracy": val_accuracy}
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return trained
