"""White-box distillation for a classifier: learn from a teacher's probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kd_course.classifier import TrainedClassifier, _batches, _schedule, predict
from kd_course.data import LabelledSet
from kd_course.models import resolve_model_path

NO_LABEL = -100


@dataclass
class DistillSet:
    """Training data for distillation.

    teacher_log_probs: one row per text, the teacher's log-probability for each class.
    labels: the human label, or None where nobody has labelled the text.
    """

    texts: list[str]
    teacher_log_probs: list[list[float]]
    labels: list[int | None]

    def __len__(self) -> int:
        return len(self.texts)


def soften(teacher_log_probs: Any, temperature: float) -> Any:
    """The teacher's distribution at a temperature: flatter above 1, same order always."""
    import torch

    return torch.softmax(teacher_log_probs / temperature, dim=-1)


def kd_loss(
    student_logits: Any,
    teacher_log_probs: Any,
    hard_labels: Any,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> Any:
    """alpha * (match the teacher) + (1 - alpha) * (match the human labels).

    hard_labels uses -100 for examples with no human label; those take part only in
    the teacher term.
    """
    import torch.nn.functional as F  # noqa: N812

    if temperature <= 0:
        raise ValueError("temperature must be above zero.")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    soft_targets = soften(teacher_log_probs, temperature)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    # Softening shrinks gradients by about 1/T^2; multiplying back keeps the two terms comparable.
    teacher_term = F.kl_div(student_log_probs, soft_targets, reduction="batchmean") * temperature**2

    if (hard_labels != NO_LABEL).any():
        label_term = F.cross_entropy(student_logits, hard_labels, ignore_index=NO_LABEL)
    else:
        label_term = student_logits.sum() * 0.0
    return alpha * teacher_term + (1 - alpha) * label_term


def train_distilled_classifier(
    model_id: str,
    train: DistillSet,
    val: LabelledSet,
    num_labels: int,
    temperature: float = 2.0,
    alpha: float = 0.5,
    epochs: int = 5,
    learning_rate: float = 5e-5,
    batch_size: int = 32,
    max_length: int = 64,
    device: str = "cuda",
    seed: int = 0,
) -> TrainedClassifier:
    """Fine-tune an encoder on teacher probabilities, plus human labels where they exist."""
    if len(train) == 0:
        raise ValueError("The training set is empty.")
    if not (len(train.teacher_log_probs) == len(train.labels) == len(train.texts)):
        raise ValueError("texts, teacher scores and labels must have the same length.")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(seed)
    path = resolve_model_path(model_id)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=num_labels)
    model.to(device)

    teacher = torch.tensor(train.teacher_log_probs, dtype=torch.float32, device=device)
    labels = torch.tensor(
        [NO_LABEL if label is None else label for label in train.labels], device=device
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    steps_per_epoch = (len(train) + batch_size - 1) // batch_size
    scheduler = _schedule(optimizer, epochs * steps_per_epoch)
    shuffle = torch.Generator().manual_seed(seed)
    trained = TrainedClassifier(model=model, tokenizer=tokenizer, max_length=max_length)
    best_accuracy, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=shuffle)
        texts = [train.texts[i] for i in order.tolist()]
        order = order.to(device)
        total_loss = 0.0
        for step, batch in enumerate(_batches(tokenizer, texts, batch_size, max_length, device)):
            rows = order[step * batch_size : (step + 1) * batch_size]
            logits = model(**batch).logits.float()
            # The only line that differs from ordinary fine-tuning: the target.
            loss = kd_loss(logits, teacher[rows], labels[rows], temperature, alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item() * len(rows)

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
