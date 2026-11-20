"""Module 8: the same distillation recipe, with images in place of text.

The teacher is a model that matches images to class names with no training.
The student is a small vision transformer. The loss is the one from Module 3.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kd_course.classifier import _schedule
from kd_course.models import resolve_model_path

PETS_ID = "timm/oxford-iiit-pet"
VISION_TEACHER_ID = "openai/clip-vit-large-patch14"
VISION_STUDENT_ID = "facebook/deit-tiny-patch16-224"
VISION_STUDENT_LARGE_ID = "google/vit-base-patch16-224-in21k"
PET_TEMPLATE = "a photo of a {}, a type of pet"
NO_LABEL = -100


@dataclass
class ImageSet:
    """Images with their labels and ids, kept in step. `images` may load lazily."""

    images: Sequence[Any]
    labels: list[int]
    ids: list[str]

    def __len__(self) -> int:
        return len(self.labels)


@dataclass
class ImageBundle:
    train: ImageSet
    test: ImageSet
    label_names: list[str]


@dataclass
class ImageSplits:
    """The scarce-label scenario for images. Same meaning as the text splits in Module 2."""

    gold_train: ImageSet
    gold_val: ImageSet
    pool: ImageSet
    test: ImageSet


class _View(Sequence):
    """Some items of another sequence, by position, without copying them."""

    def __init__(self, source: Sequence[Any], indices: list[int]):
        self._source, self._indices = source, indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, position):
        if isinstance(position, slice):
            return [self._source[i] for i in self._indices[position]]
        return self._source[self._indices[position]]


class _ImageColumn(Sequence):
    """The images of a Hugging Face dataset, decoded one at a time when asked for."""

    def __init__(self, dataset: Any):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, position):
        if isinstance(position, slice):
            return [self[i] for i in range(*position.indices(len(self)))]
        return self._dataset[int(position)]["image"].convert("RGB")


def load_pets(data_dir: str | None = None) -> ImageBundle:
    """Load the Oxford-IIIT Pet dataset: photos of 37 cat and dog breeds.

    Reads from Hugging Face, or from `<data_dir>/oxford-iiit-pet/{train,test}.parquet`.
    `data_dir` defaults to the `KD_COURSE_DATA_DIR` environment variable.
    """
    from datasets import load_dataset

    data_dir = data_dir or os.environ.get("KD_COURSE_DATA_DIR")
    local = Path(data_dir) / "oxford-iiit-pet" if data_dir else None
    if local and (local / "train.parquet").is_file():
        files = {split: str(local / f"{split}.parquet") for split in ("train", "test")}
        dataset = load_dataset("parquet", data_files=files)
    else:
        dataset = load_dataset(PETS_ID)

    def as_set(split: str) -> ImageSet:
        rows = dataset[split]
        return ImageSet(
            images=_ImageColumn(rows), labels=list(rows["label"]), ids=list(rows["image_id"])
        )

    names = list(dataset["train"].features["label"].names)
    return ImageBundle(train=as_set("train"), test=as_set("test"), label_names=names)


def _subset(data: ImageSet, indices: list[int]) -> ImageSet:
    return ImageSet(
        images=_View(data.images, indices),
        labels=[data.labels[i] for i in indices],
        ids=[data.ids[i] for i in indices],
    )


def make_image_splits(
    train: ImageSet,
    test: ImageSet,
    labels_per_class: int = 4,
    val_per_class: int = 2,
    test_size: int = 1000,
    seed: int = 13,
) -> ImageSplits:
    """Cut the data into gold_train, gold_val, pool and test. The same seed gives the same cut."""
    if test_size > len(test):
        raise ValueError(
            f"Asked for {test_size} test images but the test split has only {len(test)}."
        )
    rng = random.Random(seed)
    test_order = list(range(len(test)))
    rng.shuffle(test_order)

    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(train.labels):
        by_class[label].append(index)

    need = labels_per_class + val_per_class
    gold, val, pool = [], [], []
    for label in sorted(by_class):
        indices = by_class[label]
        if len(indices) < need:
            raise ValueError(
                f"class {label} has {len(indices)} training images; {need} are needed "
                f"({labels_per_class} to train, {val_per_class} to validate)."
            )
        rng.shuffle(indices)
        gold += indices[:labels_per_class]
        val += indices[labels_per_class:need]
        pool += indices[need:]

    return ImageSplits(
        gold_train=_subset(train, gold),
        gold_val=_subset(train, val),
        pool=_subset(train, pool),
        test=_subset(test, test_order[:test_size]),
    )


def to_pixels(image_processor: Any, images: Sequence[Any], batch_size: int = 256) -> Any:
    """Resize and normalize images into one tensor, ready for a model. Done once per model."""
    import torch

    chunks = [
        image_processor(images=list(images[start : start + batch_size]), return_tensors="pt")[
            "pixel_values"
        ]
        for start in range(0, len(images), batch_size)
    ]
    return torch.cat(chunks) if chunks else torch.empty(0)


def prompt_for(label_name: str, template: str = PET_TEMPLATE) -> str:
    """The sentence a class name is turned into for the teacher."""
    return template.format(label_name.replace("_", " "))


def load_zero_shot_teacher(
    model_id: str = VISION_TEACHER_ID, device: str = "cuda"
) -> tuple[Any, Any, Any]:
    """Load an image-and-text model. Returns the model, its image processor and its tokenizer."""
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

    path = resolve_model_path(model_id)
    model = AutoModel.from_pretrained(path).to(device).eval()
    return model, AutoImageProcessor.from_pretrained(path), AutoTokenizer.from_pretrained(path)


def zero_shot_scores(
    model: Any,
    image_processor: Any,
    tokenizer: Any,
    images: Sequence[Any],
    label_names: list[str],
    template: str = PET_TEMPLATE,
    batch_size: int = 64,
    progress_every: int = 0,
) -> list[list[float]]:
    """For each image, the teacher's log-probability for each class name.

    The teacher turns every class name into a sentence, and scores how well each image
    matches each sentence. No training is involved. Each row adds up to 100%.
    """
    import torch

    if len(images) == 0:
        return []
    device = next(model.parameters()).device
    prompts = [prompt_for(name, template) for name in label_names]
    text = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    scores: list[list[float]] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = list(images[start : start + batch_size])
            pixels = image_processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
            logits = model(pixel_values=pixels, **text).logits_per_image.float()
            scores.extend(torch.log_softmax(logits, dim=-1).tolist())
            done = start + len(batch)
            if progress_every and done % progress_every < batch_size:
                print(f"scored {done} of {len(images)}")
    return scores


def attention_targets(model: Any) -> list[str]:
    """The names of a vision transformer's query and value layers, for attaching LoRA.

    Library versions name these layers differently, so they are looked up, not assumed.
    """
    names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    for pair in (("q_proj", "v_proj"), ("query", "value")):
        if set(pair) <= names:
            return list(pair)
    raise ValueError("Could not find this model's attention layers. Pass lora_targets yourself.")


@dataclass
class TrainedImageClassifier:
    model: Any
    history: list[dict[str, float]] = field(default_factory=list)


def predict_images(
    trained: TrainedImageClassifier, pixels: Any, batch_size: int = 128
) -> list[int]:
    """The most likely class for each image."""
    import torch

    model = trained.model.eval()
    device = next(model.parameters()).device
    on_gpu = device.type == "cuda"
    predictions: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(pixels), batch_size):
            batch = pixels[start : start + batch_size].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=on_gpu):
                logits = model(pixel_values=batch).logits
            predictions.extend(logits.argmax(dim=-1).tolist())
    return predictions


def train_image_classifier(
    model_id: str,
    pixels: Any,
    labels: Sequence[int | None],
    val_pixels: Any,
    val_labels: Sequence[int],
    num_labels: int,
    teacher_log_probs: Sequence[Sequence[float]] | None = None,
    temperature: float = 2.0,
    alpha: float = 0.5,
    epochs: int = 5,
    learning_rate: float = 5e-5,
    batch_size: int = 64,
    device: str = "cuda",
    seed: int = 0,
    lora_rank: int | None = None,
    lora_targets: Sequence[str] | None = None,
) -> TrainedImageClassifier:
    """Fine-tune a vision transformer on human labels, teacher probabilities, or both.

    `labels` holds None for images nobody has labelled. With `teacher_log_probs` the loss
    is the Module 3 distillation loss; without them it is ordinary cross-entropy.
    Pass `lora_rank` to freeze the model and train LoRA adapters plus the new output layer.
    """
    if len(labels) != len(pixels):
        raise ValueError(f"Images and labels differ in length: {len(pixels)} and {len(labels)}.")
    if teacher_log_probs is None and any(label is None for label in labels):
        raise ValueError("Some images have no label, and no teacher scores were given for them.")
    if teacher_log_probs is not None and len(teacher_log_probs) != len(pixels):
        raise ValueError(
            "Images and teacher scores differ in length: "
            f"{len(pixels)} and {len(teacher_log_probs)}."
        )

    import torch
    from transformers import AutoModelForImageClassification

    from kd_course.distill import kd_loss

    torch.manual_seed(seed)
    model = AutoModelForImageClassification.from_pretrained(
        resolve_model_path(model_id), num_labels=num_labels, ignore_mismatched_sizes=True
    )
    if lora_rank is not None:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            lora_dropout=0.05,
            target_modules=list(lora_targets or attention_targets(model)),
            # The output layer is new, so it has to be trained in full.
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, config)
    model.to(device)

    hard = torch.tensor([NO_LABEL if label is None else label for label in labels], device=device)
    teacher = (
        None
        if teacher_log_probs is None
        else torch.tensor(teacher_log_probs, dtype=torch.float32, device=device)
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    steps_per_epoch = (len(pixels) + batch_size - 1) // batch_size
    scheduler = _schedule(optimizer, epochs * steps_per_epoch)
    shuffle = torch.Generator().manual_seed(seed)
    on_gpu = torch.device(device).type == "cuda"
    trained = TrainedImageClassifier(model=model)
    best_accuracy, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(pixels), generator=shuffle)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            batch = pixels[rows].to(device)
            # Mirror half the images left to right: a cheap way to show more variety.
            flip = torch.rand(len(rows), generator=shuffle) < 0.5
            batch[flip] = batch[flip].flip(-1)
            rows = rows.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=on_gpu):
                logits = model(pixel_values=batch).logits
            logits = logits.float()
            if teacher is None:
                loss = torch.nn.functional.cross_entropy(logits, hard[rows])
            else:
                loss = kd_loss(logits, teacher[rows], hard[rows], temperature, alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item() * len(rows)

        val_predictions = predict_images(trained, val_pixels)
        val_accuracy = sum(p == t for p, t in zip(val_predictions, val_labels, strict=True)) / max(
            1, len(val_labels)
        )
        trained.history.append(
            {"epoch": epoch, "train_loss": total_loss / len(pixels), "val_accuracy": val_accuracy}
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
                if lora_rank is None or "lora" in name or "classifier" in name
            }

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    model.eval()
    return trained


def zero_shot_or_load(
    path: str | Path,
    data: ImageSet,
    model: Any,
    image_processor: Any,
    tokenizer: Any,
    label_names: list[str],
    template: str = PET_TEMPLATE,
    progress_every: int = 0,
) -> list[list[float]]:
    """Score the images with the teacher and save the result, or read it back if saved before.

    Saved scores are tagged with the image ids and refuse to load for a different set.
    """
    from kd_course.label_scoring import load_scores, save_scores

    path = Path(path)
    if path.is_file():
        return load_scores(path, data.ids)
    scores = zero_shot_scores(
        model,
        image_processor,
        tokenizer,
        data.images,
        label_names,
        template,
        progress_every=progress_every,
    )
    save_scores(path, data.ids, scores)
    return scores
