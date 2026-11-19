from collections import Counter

import pytest

from kd_course.data import LabelledSet, make_splits

N_CLASSES = 5


def toy(n_per_class: int, prefix: str) -> LabelledSet:
    texts, labels = [], []
    for label in range(N_CLASSES):
        for i in range(n_per_class):
            texts.append(f"{prefix} class {label} message {i}")
            labels.append(label)
    return LabelledSet(texts=texts, labels=labels)


@pytest.fixture
def splits():
    return make_splits(
        toy(40, "train"), toy(30, "test"), labels_per_class=8, val_per_class=4, test_size=50
    )


def test_gold_train_has_the_requested_labels_per_class(splits):
    assert Counter(splits.gold_train.labels) == {c: 8 for c in range(N_CLASSES)}


def test_gold_val_has_the_requested_labels_per_class(splits):
    assert Counter(splits.gold_val.labels) == {c: 4 for c in range(N_CLASSES)}


def test_pool_is_everything_else(splits):
    assert len(splits.pool.texts) == N_CLASSES * (40 - 8 - 4)


def test_train_val_and_pool_do_not_overlap(splits):
    train, val, pool = (
        set(splits.gold_train.texts),
        set(splits.gold_val.texts),
        set(splits.pool.texts),
    )
    assert not train & val
    assert not train & pool
    assert not val & pool


def test_test_set_has_the_requested_size_and_comes_only_from_test(splits):
    assert len(splits.test.texts) == 50
    assert all(text.startswith("test") for text in splits.test.texts)


def test_no_training_text_appears_in_the_test_set(splits):
    training = set(splits.gold_train.texts) | set(splits.gold_val.texts) | set(splits.pool.texts)
    assert not training & set(splits.test.texts)


def test_same_seed_gives_the_same_split():
    a = make_splits(toy(40, "train"), toy(30, "test"), test_size=50, seed=13)
    b = make_splits(toy(40, "train"), toy(30, "test"), test_size=50, seed=13)
    assert a.gold_train.texts == b.gold_train.texts
    assert a.test.texts == b.test.texts


def test_different_seed_gives_a_different_split():
    a = make_splits(toy(40, "train"), toy(30, "test"), test_size=50, seed=13)
    b = make_splits(toy(40, "train"), toy(30, "test"), test_size=50, seed=14)
    assert a.gold_train.texts != b.gold_train.texts


def test_duplicate_of_a_test_message_is_removed_from_training():
    train = toy(40, "train")
    test = toy(30, "test")
    train.texts[0] = "  " + test.texts[0].upper() + " "  # same message, different case and spacing
    splits = make_splits(train, test, test_size=N_CLASSES * 30)
    training = splits.gold_train.texts + splits.gold_val.texts + splits.pool.texts
    assert train.texts[0] not in training
    assert splits.removed_duplicates == 1


def test_a_class_with_too_few_examples_is_an_error():
    with pytest.raises(ValueError, match="class 0"):
        make_splits(
            toy(5, "train"), toy(30, "test"), labels_per_class=8, val_per_class=4, test_size=50
        )


def test_asking_for_more_test_messages_than_exist_is_an_error():
    with pytest.raises(ValueError, match="test"):
        make_splits(toy(40, "train"), toy(2, "test"), test_size=50)


def test_load_banking77_from_a_local_folder(tmp_path):
    datasets = pytest.importorskip("datasets")
    from kd_course.data import load_banking77

    features = datasets.Features(
        {
            "text": datasets.Value("string"),
            "label": datasets.ClassLabel(names=["card_arrival", "lost_card"]),
        }
    )
    folder = tmp_path / "banking77"
    folder.mkdir()
    for split, rows in {"train": ["a", "b", "c"], "test": ["d"]}.items():
        dataset = datasets.Dataset.from_dict(
            {"text": rows, "label": [0, 1, 0][: len(rows)]}, features=features
        )
        dataset.to_parquet(str(folder / f"{split}.parquet"))
    bundle = load_banking77(data_dir=str(tmp_path))
    assert bundle.train.texts == ["a", "b", "c"]
    assert bundle.train.labels == [0, 1, 0]
    assert bundle.test.texts == ["d"]
    assert bundle.label_names == ["card_arrival", "lost_card"]
