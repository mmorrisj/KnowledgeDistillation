import copy

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from kd_course.generative import generate_replies, train_text_pairs  # noqa: E402
from kd_course.token_kd import (  # noqa: E402
    token_kd_loss,
    topk_targets,
    train_token_kd,
)

PROMPTS = ["the cat sat on", "a dog ran to"]
TARGETS = ["the mat", "the floor"]


def test_train_text_pairs_learns_to_continue_prompts(tiny_lm):
    model, tokenizer = copy.deepcopy(tiny_lm[0]), tiny_lm[1]
    history = train_text_pairs(
        model,
        tokenizer,
        PROMPTS * 4,
        TARGETS * 4,
        epochs=40,
        learning_rate=5e-3,
        batch_size=4,
        seed=0,
    )
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert generate_replies(model, tokenizer, PROMPTS, max_new_tokens=4) == TARGETS


def test_generate_replies_handles_an_empty_list(tiny_lm):
    assert generate_replies(tiny_lm[0], tiny_lm[1], []) == []


def test_topk_targets_match_a_direct_calculation(tiny_lm):
    model, tokenizer = tiny_lm
    records = topk_targets(model, tokenizer, PROMPTS, TARGETS, k=5)
    assert len(records) == 2
    for prompt, target, record in zip(PROMPTS, TARGETS, records, strict=True):
        prompt_ids = tokenizer(prompt)["input_ids"]
        target_ids = tokenizer(" " + target, add_special_tokens=False)["input_ids"] + [
            tokenizer.eos_token_id
        ]
        assert record.target_ids == target_ids
        with torch.inference_mode():
            logits = model(torch.tensor([prompt_ids + target_ids])).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        for position in range(len(target_ids)):
            expected = log_probs[len(prompt_ids) + position - 1].topk(5)
            assert record.top_ids[position].tolist() == expected.indices.tolist()
            assert record.top_log_probs[position] == pytest.approx(expected.values, abs=1e-4)


def test_topk_targets_shapes(tiny_lm):
    model, tokenizer = tiny_lm
    record = topk_targets(model, tokenizer, PROMPTS[:1], TARGETS[:1], k=3)[0]
    assert record.top_ids.shape == (len(record.target_ids), 3)
    assert record.top_log_probs.shape == (len(record.target_ids), 3)


def uniform_case(vocab=6, positions=2, k=6):
    torch.manual_seed(0)
    teacher_logits = torch.randn(1, positions, vocab)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    top = teacher_log_probs.topk(k, dim=-1)
    return teacher_logits, teacher_log_probs, top.indices, top.values


def test_loss_is_zero_when_the_student_matches_the_teacher():
    teacher_logits, _, ids, log_probs = uniform_case()
    targets = ids[..., 0]
    mask = torch.ones(1, 2)
    loss = token_kd_loss(teacher_logits, ids, log_probs, targets, mask, temperature=1.0, alpha=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_with_every_token_kept_the_loss_is_the_full_kl_divergence():
    teacher_logits, teacher_log_probs, ids, log_probs = uniform_case()
    student_logits = torch.randn(1, 2, 6)
    mask = torch.ones(1, 2)
    expected = F.kl_div(
        F.log_softmax(student_logits, dim=-1).view(-1, 6),
        teacher_log_probs.exp().view(-1, 6),
        reduction="batchmean",
    )
    loss = token_kd_loss(
        student_logits, ids, log_probs, ids[..., 0], mask, temperature=1.0, alpha=1.0
    )
    assert loss.item() == pytest.approx(expected.item(), abs=1e-5)


def test_alpha_zero_is_cross_entropy_on_the_teacher_tokens():
    _, _, ids, log_probs = uniform_case()
    student_logits = torch.randn(1, 2, 6)
    targets = ids[..., 0]
    mask = torch.ones(1, 2)
    loss = token_kd_loss(student_logits, ids, log_probs, targets, mask, temperature=2.0, alpha=0.0)
    assert loss.item() == pytest.approx(
        F.cross_entropy(student_logits.view(-1, 6), targets.view(-1)).item(), abs=1e-5
    )


def test_masked_positions_do_not_affect_the_loss():
    _, _, ids, log_probs = uniform_case()
    student = torch.randn(1, 2, 6)
    mask = torch.tensor([[1.0, 0.0]])
    first = token_kd_loss(student, ids, log_probs, ids[..., 0], mask, 2.0, 0.5)
    changed = student.clone()
    changed[0, 1] += 5.0
    second = token_kd_loss(changed, ids, log_probs, ids[..., 0], mask, 2.0, 0.5)
    assert first.item() == pytest.approx(second.item())


def test_top_k_teacher_probabilities_are_renormalized():
    # Keeping k=2 of 6: the two kept teacher probabilities are rescaled to add up to 1,
    # and a student that puts the same relative weight on them scores zero on the teacher term.
    teacher_logits, _, _, _ = uniform_case()
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    top = teacher_log_probs.topk(2, dim=-1)
    mask = torch.ones(1, 2)
    loss = token_kd_loss(
        teacher_logits, top.indices, top.values, top.indices[..., 0], mask, 1.0, 1.0
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_train_token_kd_pulls_a_student_toward_its_teacher(tiny_lm):
    teacher, tokenizer = tiny_lm
    torch.manual_seed(1)
    student = copy.deepcopy(teacher)
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    records = topk_targets(teacher, tokenizer, PROMPTS, TARGETS, k=8)
    history = train_token_kd(
        student,
        tokenizer,
        PROMPTS,
        records,
        temperature=1.0,
        alpha=1.0,
        epochs=30,
        learning_rate=2e-3,
        batch_size=2,
    )
    assert history[-1]["train_loss"] < history[0]["train_loss"] * 0.5


def test_train_token_kd_rejects_misaligned_inputs(tiny_lm):
    with pytest.raises(ValueError, match="length"):
        train_token_kd(tiny_lm[0], tiny_lm[1], PROMPTS, [])
