"""Save a trained student as a fixed artifact, load it for serving, and shrink it."""

from __future__ import annotations

import copy
import io
import json
from pathlib import Path
from typing import Any

from kd_course.classifier import TrainedClassifier

_META = "student.json"


class ClassifierPredictor:
    """Wraps a classifier for serving: texts in, (label index, confidence) out."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_length: int,
        name: str = "student",
        batch_size: int = 64,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.name = name
        self.batch_size = batch_size

    def predict(self, texts: list[str]) -> list[tuple[int, float]]:
        import torch

        results: list[tuple[int, float]] = []
        device = next(self.model.parameters(), torch.zeros(1)).device
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    texts[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)
                probabilities = torch.softmax(self.model(**batch).logits.float(), dim=-1)
                confidence, label = probabilities.max(dim=-1)
                results.extend(zip(label.tolist(), confidence.tolist(), strict=True))
        return results


def save_classifier(trained: TrainedClassifier, label_names: list[str], path: str | Path) -> Path:
    """Write the model, its tokenizer and its label names to one folder."""
    if len(label_names) != trained.model.config.num_labels:
        raise ValueError(
            f"The model has {trained.model.config.num_labels} outputs "
            f"but {len(label_names)} labels were given."
        )
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    trained.model.save_pretrained(folder)
    trained.tokenizer.save_pretrained(folder)
    meta = {"labels": label_names, "max_length": trained.max_length}
    (folder / _META).write_text(json.dumps(meta, indent=2))
    return folder


def quantize_int8(model: Any) -> Any:
    """A copy of the model with its linear layers stored in 8 bits. Runs on CPU only."""
    import torch

    cpu_copy = copy.deepcopy(model).cpu().float().eval()
    return torch.ao.quantization.quantize_dynamic(cpu_copy, {torch.nn.Linear}, dtype=torch.qint8)


def model_size_mb(model: Any) -> float:
    """Size of the model's weights when saved, in megabytes."""
    import torch

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1_048_576


def load_classifier(
    path: str | Path, device: str = "cuda", quantize: bool = False
) -> tuple[ClassifierPredictor, list[str]]:
    """Load a student saved by `save_classifier`. Returns a predictor and the label names."""
    folder = Path(path)
    if not (folder / _META).is_file():
        raise FileNotFoundError(
            f"{folder} is not a saved student. Run the Module 7 lab to create one, "
            "or set KD_STUDENT_DIR to the folder that holds it."
        )
    if quantize and device != "cpu":
        raise ValueError("An 8-bit model runs on CPU. Use device='cpu' with quantize=True.")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    meta = json.loads((folder / _META).read_text())
    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(folder)
    name = folder.name
    if quantize:
        model, name = quantize_int8(model), f"{name}-int8"
    else:
        model = model.to(device)
    return ClassifierPredictor(model, tokenizer, meta["max_length"], name=name), meta["labels"]
