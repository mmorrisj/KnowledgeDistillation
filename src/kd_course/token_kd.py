"""Token-level distillation: the student matches the teacher's next-token probabilities.

A language model has a probability for every token in its vocabulary at every position,
which is far too much to store. Only the teacher's top few tokens per position are kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kd_course.classifier import _schedule
from kd_course.generative import _pad_id, answer_token_ids


@dataclass
class TokenTargets:
    """What the teacher thought at each position of one answer.

    target_ids: the answer's tokens, ending with the end-of-text token.
    top_ids / top_log_probs: for each of those positions, the teacher's k most likely
        tokens and their log-probabilities. Shape: (answer length, k).
    """

    target_ids: list[int]
    top_ids: Any
    top_log_probs: Any


def topk_targets(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    answers: list[str],
    k: int = 20,
    answer_prefix: str = " ",
) -> list[TokenTargets]:
    """Read the teacher's top-k next-token probabilities along each answer.

    The teacher is shown its prompt followed by the answer, and at every answer position
    its distribution over the next token is recorded. Nothing is generated.
    """
    import torch

    records: list[TokenTargets] = []
    with torch.inference_mode():
        for prompt, answer in zip(prompts, answers, strict=True):
            prompt_ids = tokenizer(prompt)["input_ids"]
            target_ids = answer_token_ids(tokenizer, answer, answer_prefix)
            ids = torch.tensor([prompt_ids + target_ids], device=model.device)
            logits = model(ids).logits[0].float()
            # The prediction for answer token i is made at the position just before it.
            start = len(prompt_ids) - 1
            log_probs = torch.log_softmax(logits[start : start + len(target_ids)], dim=-1)
            top = log_probs.topk(k, dim=-1)
            records.append(
                TokenTargets(
                    target_ids=target_ids,
                    top_ids=top.indices.cpu(),
                    top_log_probs=top.values.cpu(),
                )
            )
    return records


def token_kd_loss(
    student_logits: Any,
    top_ids: Any,
    top_log_probs: Any,
    target_ids: Any,
    mask: Any,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> Any:
    """alpha * (match the teacher's top-k distribution) + (1 - alpha) * (predict its token).

    Shapes: student_logits (batch, length, vocabulary); top_ids and top_log_probs
    (batch, length, k); target_ids and mask (batch, length). Positions where mask is 0
    are ignored.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    count = mask.sum().clamp(min=1)

    # Teacher: its top-k probabilities, softened and rescaled to add up to 1.
    teacher = torch.softmax(top_log_probs / temperature, dim=-1)
    # Student: its probabilities for the same k tokens, rescaled the same way.
    student_all = F.log_softmax(student_logits / temperature, dim=-1)
    student = student_all.gather(-1, top_ids)
    student = student - torch.logsumexp(student, dim=-1, keepdim=True)

    kl = (teacher * (torch.log(teacher.clamp(min=1e-12)) - student)).sum(dim=-1)
    teacher_term = (kl * mask).sum() / count * temperature**2

    vocabulary = student_logits.shape[-1]
    token_loss = F.cross_entropy(
        student_logits.reshape(-1, vocabulary), target_ids.reshape(-1), reduction="none"
    ).reshape(target_ids.shape)
    label_term = (token_loss * mask).sum() / count
    return alpha * teacher_term + (1 - alpha) * label_term


def _kd_batch(tokenizer: Any, prompts: list[str], records: list[TokenTargets], device: Any):
    """Pad one batch. Returns model inputs, and for each row where its answer starts."""
    import torch

    prompt_ids = [tokenizer(prompt)["input_ids"] for prompt in prompts]
    lengths = [len(p) + len(r.target_ids) for p, r in zip(prompt_ids, records, strict=True)]
    longest, longest_answer = max(lengths), max(len(r.target_ids) for r in records)
    k = records[0].top_ids.shape[-1]
    n = len(prompts)

    input_ids = torch.full((n, longest), _pad_id(tokenizer), dtype=torch.long)
    attention = torch.zeros((n, longest), dtype=torch.long)
    top_ids = torch.zeros((n, longest_answer, k), dtype=torch.long)
    top_log_probs = torch.zeros((n, longest_answer, k))
    targets = torch.zeros((n, longest_answer), dtype=torch.long)
    mask = torch.zeros((n, longest_answer))
    for row, (p_ids, record) in enumerate(zip(prompt_ids, records, strict=True)):
        size = len(record.target_ids)
        input_ids[row, : lengths[row]] = torch.tensor(p_ids + record.target_ids)
        attention[row, : lengths[row]] = 1
        top_ids[row, :size] = record.top_ids
        top_log_probs[row, :size] = record.top_log_probs
        targets[row, :size] = torch.tensor(record.target_ids)
        mask[row, :size] = 1
    starts = [len(p) - 1 for p in prompt_ids]
    tensors = (input_ids, attention, top_ids, top_log_probs, targets, mask)
    return [t.to(device) for t in tensors], starts


def train_token_kd(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    records: list[TokenTargets],
    temperature: float = 2.0,
    alpha: float = 0.5,
    epochs: int = 2,
    learning_rate: float = 2e-5,
    batch_size: int = 32,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Train the student to match the teacher's top-k distribution at every answer position.

    `prompts` are the student's own prompts; they can differ from the teacher's. The answer
    tokens must be the same for both, which needs a shared tokenizer.
    """
    if len(prompts) != len(records):
        raise ValueError(
            f"Prompts and teacher records differ in length: {len(prompts)} and {len(records)}."
        )
    if not prompts:
        raise ValueError("The training set is empty.")
    import torch

    torch.manual_seed(seed)
    device = model.device
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    steps_per_epoch = (len(prompts) + batch_size - 1) // batch_size
    scheduler = _schedule(optimizer, epochs * steps_per_epoch)
    shuffle = torch.Generator().manual_seed(seed)
    on_gpu = device.type == "cuda"
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(prompts), generator=shuffle).tolist()
        total_loss = 0.0
        for begin in range(0, len(order), batch_size):
            rows = order[begin : begin + batch_size]
            tensors, starts = _kd_batch(
                tokenizer, [prompts[i] for i in rows], [records[i] for i in rows], device
            )
            input_ids, attention, top_ids, top_log_probs, targets, mask = tensors
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=on_gpu):
                logits = model(input_ids=input_ids, attention_mask=attention).logits
            # Line up each row's answer positions at the left of a padded block.
            answer_logits = logits.new_zeros((len(rows), targets.shape[1], logits.shape[-1]))
            for row, start in enumerate(starts):
                size = int(mask[row].sum().item())
                answer_logits[row, :size] = logits[row, start : start + size]
            loss = token_kd_loss(
                answer_logits.float(), top_ids, top_log_probs, targets, mask, temperature, alpha
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item() * len(rows)
        history.append({"epoch": epoch, "train_loss": total_loss / len(prompts)})

    model.eval()
    return history


def storage_gb(n_tokens: int, k: int, vocabulary: int | None = None) -> float:
    """Disk space for teacher targets: k (id, probability) pairs per token, or the full vocabulary.

    Assumes 4 bytes per token id and 2 bytes per probability.
    """
    per_token = vocabulary * 2 if vocabulary else k * 6
    return n_tokens * per_token / 1024**3
