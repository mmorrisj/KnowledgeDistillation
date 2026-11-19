"""Load the course dataset and cut it into the fixed splits every module uses."""

from __future__ import annotations

import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

BANKING77_ID = "PolyAI/banking77"


@dataclass
class LabelledSet:
    texts: list[str] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.texts)


@dataclass
class DatasetBundle:
    train: LabelledSet
    test: LabelledSet
    label_names: list[str]


@dataclass
class Splits:
    """The course's scarce-label scenario.

    gold_train / gold_val: the few messages a person has labelled.
    pool: messages with no human label. `pool.labels` holds the true labels so
        later modules can measure the teacher's mistakes. Never train on them.
    test: held out. Used only to score finished models.
    """

    gold_train: LabelledSet
    gold_val: LabelledSet
    pool: LabelledSet
    test: LabelledSet
    removed_duplicates: int = 0


def load_banking77(data_dir: str | None = None) -> DatasetBundle:
    """Load Banking77 from Hugging Face, or from `<data_dir>/banking77/{train,test}.parquet`.

    `data_dir` defaults to the `KD_COURSE_DATA_DIR` environment variable.
    """
    from datasets import load_dataset

    data_dir = data_dir or os.environ.get("KD_COURSE_DATA_DIR")
    local = Path(data_dir) / "banking77" if data_dir else None
    if local and (local / "train.parquet").is_file():
        files = {split: str(local / f"{split}.parquet") for split in ("train", "test")}
        dataset = load_dataset("parquet", data_files=files)
    else:
        dataset = load_dataset(BANKING77_ID)

    def as_set(split: str) -> LabelledSet:
        return LabelledSet(texts=list(dataset[split]["text"]), labels=list(dataset[split]["label"]))

    return DatasetBundle(
        train=as_set("train"),
        test=as_set("test"),
        label_names=list(dataset["train"].features["label"].names),
    )


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def make_splits(
    train: LabelledSet,
    test: LabelledSet,
    labels_per_class: int = 8,
    val_per_class: int = 4,
    test_size: int = 1000,
    seed: int = 13,
) -> Splits:
    """Cut the data into gold_train, gold_val, pool and test. The same seed gives the same cut."""
    if test_size > len(test):
        raise ValueError(
            f"Asked for {test_size} test messages but the test split has only {len(test)}."
        )
    rng = random.Random(seed)

    test_order = list(range(len(test)))
    rng.shuffle(test_order)
    chosen = test_order[:test_size]
    test_set = LabelledSet([test.texts[i] for i in chosen], [test.labels[i] for i in chosen])

    # A training message that also appears in the test set would leak the answer.
    test_texts = {_normalize(text) for text in test_set.texts}
    by_class: dict[int, list[int]] = defaultdict(list)
    removed = 0
    for index, (text, label) in enumerate(zip(train.texts, train.labels, strict=True)):
        if _normalize(text) in test_texts:
            removed += 1
        else:
            by_class[label].append(index)

    need = labels_per_class + val_per_class
    gold_train, gold_val, pool = LabelledSet(), LabelledSet(), LabelledSet()
    for label in sorted(by_class):
        indices = by_class[label]
        if len(indices) < need:
            raise ValueError(
                f"class {label} has {len(indices)} training messages; "
                f"{need} are needed ({labels_per_class} for training, {val_per_class} for validation)."
            )
        rng.shuffle(indices)
        parts = (
            (gold_train, indices[:labels_per_class]),
            (gold_val, indices[labels_per_class:need]),
            (pool, indices[need:]),
        )
        for target, part in parts:
            target.texts.extend(train.texts[i] for i in part)
            target.labels.extend(train.labels[i] for i in part)

    return Splits(gold_train, gold_val, pool, test_set, removed_duplicates=removed)
