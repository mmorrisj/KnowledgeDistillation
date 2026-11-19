import math

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from kd_course.distill import DistillSet, kd_loss, soften, train_distilled_classifier  # noqa: E402

from .test_classifier import TRAIN, tiny_encoder_dir  # noqa: E402, F401

NO_LABEL = -100


def test_soften_at_temperature_one_returns_the_teacher_distribution():
    log_probs = torch.log(torch.tensor([[0.7, 0.2, 0.1]]))
    assert soften(log_probs, 1.0) == pytest.approx(torch.tensor([[0.7, 0.2, 0.1]]))


def test_soften_flattens_as_temperature_rises_but_keeps_the_order():
    log_probs = torch.log(torch.tensor([[0.7, 0.2, 0.1]]))
    hot = soften(log_probs, 4.0)[0]
    assert hot[0] < 0.7 and hot[2] > 0.1
    assert hot[0] > hot[1] > hot[2]
    assert hot.sum().item() == pytest.approx(1.0)


def test_loss_is_zero_when_the_student_matches_the_teacher_and_alpha_is_one():
    teacher = torch.log(torch.tensor([[0.7, 0.2, 0.1]]))
    loss = kd_loss(teacher.clone(), teacher, torch.tensor([NO_LABEL]), temperature=1.0, alpha=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_matches_a_hand_calculation():
    student = torch.tensor([[2.0, 0.0, -1.0], [0.5, 0.5, 0.0]])
    teacher = torch.log(torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]]))
    labels = torch.tensor([0, NO_LABEL])
    T, alpha = 2.0, 0.3
    soft = torch.softmax(teacher / T, dim=-1)
    kd = F.kl_div(F.log_softmax(student / T, dim=-1), soft, reduction="batchmean") * T**2
    ce = F.cross_entropy(student[:1], labels[:1])
    expected = alpha * kd + (1 - alpha) * ce
    assert kd_loss(student, teacher, labels, T, alpha).item() == pytest.approx(expected.item())


def test_alpha_zero_is_plain_cross_entropy_on_the_labelled_examples():
    student = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    teacher = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]]))
    labels = torch.tensor([0, 1])
    assert kd_loss(student, teacher, labels, 2.0, 0.0).item() == pytest.approx(
        F.cross_entropy(student, labels).item()
    )


def test_a_batch_with_no_human_labels_gives_a_finite_loss():
    student = torch.tensor([[2.0, 0.0]], requires_grad=True)
    teacher = torch.log(torch.tensor([[0.5, 0.5]]))
    loss = kd_loss(student, teacher, torch.tensor([NO_LABEL]), 2.0, 0.5)
    loss.backward()
    assert math.isfinite(loss.item())
    assert torch.isfinite(student.grad).all()


@pytest.mark.parametrize("bad", [{"temperature": 0.0}, {"alpha": 1.5}, {"alpha": -0.1}])
def test_invalid_settings_are_rejected(bad):
    args = {"temperature": 2.0, "alpha": 0.5, **bad}
    with pytest.raises(ValueError):
        kd_loss(torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([0]), **args)


def confident(label: int) -> list[float]:
    return [math.log(0.9), math.log(0.1)] if label == 0 else [math.log(0.1), math.log(0.9)]


def test_a_student_learns_from_teacher_probabilities_alone(tiny_encoder_dir):  # noqa: F811
    train = DistillSet(
        texts=TRAIN.texts,
        teacher_log_probs=[confident(label) for label in TRAIN.labels],
        labels=[None] * len(TRAIN.texts),
    )
    from kd_course.classifier import predict

    trained = train_distilled_classifier(
        tiny_encoder_dir,
        train,
        TRAIN,
        num_labels=2,
        temperature=1.0,
        alpha=1.0,
        epochs=30,
        learning_rate=5e-3,
        batch_size=4,
        device="cpu",
        seed=0,
    )
    assert predict(trained, TRAIN.texts) == TRAIN.labels
    assert trained.history[-1]["train_loss"] < trained.history[0]["train_loss"]


def test_training_rejects_misaligned_teacher_scores(tiny_encoder_dir):  # noqa: F811
    bad = DistillSet(texts=["my card"], teacher_log_probs=[], labels=[None])
    with pytest.raises(ValueError, match="teacher"):
        train_distilled_classifier(tiny_encoder_dir, bad, TRAIN, num_labels=2, device="cpu")
