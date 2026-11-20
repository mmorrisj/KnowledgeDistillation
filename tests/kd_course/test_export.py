import pytest

torch = pytest.importorskip("torch")

from kd_course.classifier import TrainedClassifier, predict, train_classifier  # noqa: E402
from kd_course.serving.export import (  # noqa: E402
    ClassifierPredictor,
    load_classifier,
    model_size_mb,
    quantize_int8,
    save_classifier,
)

from .test_classifier import TRAIN, tiny_encoder_dir  # noqa: E402, F401

LABELS = ["lost_card", "exchange_rate"]


@pytest.fixture(scope="module")
def trained(tiny_encoder_dir):  # noqa: F811
    return train_classifier(
        tiny_encoder_dir,
        TRAIN,
        TRAIN,
        num_labels=2,
        epochs=30,
        learning_rate=5e-3,
        batch_size=4,
        device="cpu",
        seed=0,
    )


def test_predictor_returns_a_label_and_a_confidence_between_0_and_1(trained):
    predictor = ClassifierPredictor(
        trained.model, trained.tokenizer, trained.max_length, name="tiny"
    )
    results = predictor.predict(TRAIN.texts)
    assert [label for label, _ in results] == predict(trained, TRAIN.texts)
    assert all(0.5 <= confidence <= 1.0 for _, confidence in results)
    assert predictor.predict([]) == []


def test_saved_classifier_loads_back_with_the_same_answers(trained, tmp_path):
    save_classifier(trained, LABELS, tmp_path / "student")
    predictor, labels = load_classifier(tmp_path / "student", device="cpu")
    assert labels == LABELS
    assert [label for label, _ in predictor.predict(TRAIN.texts)] == predict(trained, TRAIN.texts)
    assert predictor.name == "student"


def test_save_rejects_a_label_list_of_the_wrong_length(trained, tmp_path):
    with pytest.raises(ValueError, match="labels"):
        save_classifier(trained, ["only_one"], tmp_path / "student")


def test_loading_a_folder_that_is_not_a_saved_student_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="saved student"):
        load_classifier(tmp_path / "nothing", device="cpu")


def test_int8_model_is_smaller_and_still_answers_the_same(trained):
    quantized = quantize_int8(trained.model)
    assert model_size_mb(quantized) < model_size_mb(trained.model)
    small = TrainedClassifier(
        model=quantized, tokenizer=trained.tokenizer, max_length=trained.max_length
    )
    assert predict(small, TRAIN.texts) == predict(trained, TRAIN.texts)


def test_quantizing_does_not_change_the_original_model(trained):
    before = model_size_mb(trained.model)
    quantize_int8(trained.model)
    assert model_size_mb(trained.model) == before


def test_loading_with_quantize_gives_a_working_int8_predictor(trained, tmp_path):
    save_classifier(trained, LABELS, tmp_path / "student")
    predictor, _ = load_classifier(tmp_path / "student", device="cpu", quantize=True)
    assert [label for label, _ in predictor.predict(TRAIN.texts)] == predict(trained, TRAIN.texts)
    assert predictor.name == "student-int8"


def test_quantizing_on_a_gpu_device_is_refused(trained, tmp_path):
    save_classifier(trained, LABELS, tmp_path / "student")
    with pytest.raises(ValueError, match="CPU"):
        load_classifier(tmp_path / "student", device="cuda", quantize=True)
