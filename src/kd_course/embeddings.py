"""Distil a text-embedding model: the student learns to arrange texts the way the teacher does.

An embedding model has no classes and writes no text. It turns each text into a list of
numbers, and what matters is which texts end up close together.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from kd_course.classifier import _schedule
from kd_course.models import resolve_model_path

EMBED_TEACHER_ID = "BAAI/bge-large-en-v1.5"  # 335M weights, 1,024 numbers per text
EMBED_STUDENT_ID = "BAAI/bge-small-en-v1.5"  # 33M weights, 384 numbers per text


def load_embedder(model_id: str, device: str = "cuda") -> tuple[Any, Any]:
    """Load an embedding model and its tokenizer."""
    from transformers import AutoModel, AutoTokenizer

    path = resolve_model_path(model_id)
    return AutoModel.from_pretrained(path).to(device).eval(), AutoTokenizer.from_pretrained(path)


def _encode(model: Any, tokenizer: Any, texts: list[str], max_length: int) -> Any:
    """One unit-length vector per text, taken from the model's first output position."""
    import torch

    device = next(model.parameters()).device
    batch = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    ).to(device)
    hidden = model(**batch).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(hidden.float(), dim=-1)


def embed_texts(
    model: Any, tokenizer: Any, texts: Sequence[str], batch_size: int = 128, max_length: int = 64
) -> Any:
    """Embed every text. Returns a CPU tensor with one unit-length row per text."""
    import torch

    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunks.append(
                _encode(model, tokenizer, list(texts[start : start + batch_size]), max_length).cpu()
            )
    return torch.cat(chunks) if chunks else torch.empty(0)


def similarity_matching_loss(student: Any, teacher: Any) -> Any:
    """How differently the student and the teacher arrange one batch of texts.

    For every pair of texts in the batch, compare the student's similarity with the teacher's.
    The two models can use different numbers of dimensions, because only similarities
    between texts are compared, never the vectors themselves.
    """
    import torch

    n = student.shape[0]
    if n < 2:
        raise ValueError("At least two texts are needed to compare similarities.")
    student_sims = student @ student.T
    teacher_sims = teacher @ teacher.T
    different = ~torch.eye(
        n, dtype=torch.bool, device=student.device
    )  # leave out each text with itself
    return ((student_sims - teacher_sims)[different] ** 2).mean()


def train_embedding_student(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    teacher_embeddings: Any,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 128,
    max_length: int = 64,
    device: str = "cuda",
    seed: int = 0,
) -> list[dict[str, float]]:
    """Train the student so that texts the teacher puts close together end up close together.

    `teacher_embeddings` are computed once beforehand, one row per text. No labels are used.
    """
    if len(texts) != len(teacher_embeddings):
        raise ValueError(
            "Texts and teacher embeddings differ in length: "
            f"{len(texts)} and {len(teacher_embeddings)}."
        )
    import torch

    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    steps_per_epoch = (len(texts) + batch_size - 1) // batch_size
    scheduler = _schedule(optimizer, epochs * steps_per_epoch)
    shuffle = torch.Generator().manual_seed(seed)
    device_type = torch.device(device).type
    on_gpu = device_type == "cuda"
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(texts), generator=shuffle)
        total, seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            if len(rows) < 2:
                continue  # a single text has no pair to compare
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=on_gpu):
                student = _encode(model, tokenizer, [texts[i] for i in rows.tolist()], max_length)
            teacher = teacher_embeddings[rows].to(device).float()
            loss = similarity_matching_loss(student, teacher)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += loss.item() * len(rows)
            seen += len(rows)
        history.append({"epoch": epoch, "train_loss": total / max(1, seen)})

    model.eval()
    return history


def top_neighbours(
    queries: Any, corpus: Any, k: int = 10, exclude_self: bool = False
) -> list[list[int]]:
    """For each query, the positions of its k most similar corpus vectors, nearest first.

    This is an exact search: every query is compared with every corpus vector.
    Set `exclude_self` when the queries are the corpus itself.
    """
    import torch

    available = corpus.shape[0] - (1 if exclude_self else 0)
    if k > available:
        raise ValueError(f"k is {k} but only {available} neighbours are available.")
    similarities = queries.float() @ corpus.float().T
    if exclude_self:
        similarities.fill_diagonal_(-torch.inf)
    return similarities.topk(k, dim=-1).indices.tolist()


def per_query_precision(
    neighbours: Sequence[Sequence[int]], query_labels: Sequence[int], corpus_labels: Sequence[int]
) -> list[float]:
    """For each query, the share of its results that have the query's own label."""
    return [
        sum(corpus_labels[i] == label for i in found) / len(found)
        for found, label in zip(neighbours, query_labels, strict=True)
    ]


def retrieval_scores(
    neighbours: Sequence[Sequence[int]], query_labels: Sequence[int], corpus_labels: Sequence[int]
) -> dict[str, float]:
    """Search quality when "relevant" means "has the same label as the query".

    precision_at_k: share of the k results that are relevant, averaged over queries.
    top_1_same_class: share of queries whose first result is relevant.
    """
    precision, top_1 = 0.0, 0.0
    for found, label in zip(neighbours, query_labels, strict=True):
        relevant = [corpus_labels[i] == label for i in found]
        precision += sum(relevant) / len(relevant)
        top_1 += relevant[0]
    n = len(query_labels)
    return {"precision_at_k": precision / n, "top_1_same_class": top_1 / n, "k": len(neighbours[0])}


def neighbour_overlap(first: Sequence[Sequence[int]], second: Sequence[Sequence[int]]) -> float:
    """Average share of results two models have in common, query by query. Needs no labels."""
    if len(first) != len(second):
        raise ValueError(f"The two result lists differ in length: {len(first)} and {len(second)}.")
    shares = [len(set(a) & set(b)) / len(a) for a, b in zip(first, second, strict=True)]
    return sum(shares) / len(shares)
