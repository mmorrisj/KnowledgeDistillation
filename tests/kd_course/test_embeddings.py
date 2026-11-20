import copy

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from kd_course.embeddings import (  # noqa: E402
    embed_texts,
    neighbour_overlap,
    retrieval_scores,
    similarity_matching_loss,
    top_neighbours,
    train_embedding_student,
)

from .test_classifier import tiny_encoder_dir  # noqa: E402, F401

TEXTS = [
    "my card lost",
    "the card stolen",
    "lost my card",
    "exchange rate euro",
    "the euro rate",
    "rate exchange",
]


@pytest.fixture(scope="module")
def tiny_embedder(tiny_encoder_dir):  # noqa: F811
    model = transformers.AutoModel.from_pretrained(tiny_encoder_dir).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny_encoder_dir)
    return model, tokenizer


def test_embeddings_have_one_unit_length_row_per_text(tiny_embedder):
    model, tokenizer = tiny_embedder
    vectors = embed_texts(model, tokenizer, TEXTS, batch_size=4)
    assert vectors.shape == (6, 32)
    assert vectors.norm(dim=-1) == pytest.approx(torch.ones(6), abs=1e-5)


def test_embedding_no_texts_gives_an_empty_result(tiny_embedder):
    model, tokenizer = tiny_embedder
    assert embed_texts(model, tokenizer, []).shape[0] == 0


def test_embeddings_do_not_depend_on_batching(tiny_embedder):
    model, tokenizer = tiny_embedder
    together = embed_texts(model, tokenizer, TEXTS, batch_size=6)
    one_by_one = embed_texts(model, tokenizer, TEXTS, batch_size=1)
    assert together == pytest.approx(one_by_one, abs=1e-5)


def unit(rows):
    return torch.nn.functional.normalize(torch.tensor(rows, dtype=torch.float32), dim=-1)


def test_similarity_loss_is_zero_when_the_student_arranges_texts_like_the_teacher():
    teacher = unit([[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]])
    # A student with fewer dimensions and a rotated space, but the same angles between texts.
    student = unit([[0, 1], [-1, 0], [-1, 1]])
    assert similarity_matching_loss(student, teacher).item() == pytest.approx(0.0, abs=1e-6)


def test_similarity_loss_is_positive_when_the_arrangement_differs():
    teacher = unit([[1, 0], [0, 1], [1, 1]])
    student = unit([[1, 0], [1, 0.1], [0, 1]])
    assert similarity_matching_loss(student, teacher).item() > 0.01


def test_similarity_loss_matches_a_hand_calculation():
    teacher = unit([[1, 0], [0, 1]])  # similarity between the two texts: 0
    student = unit([[1, 0], [1, 0]])  # similarity between the two texts: 1
    # one pair, squared difference (1 - 0)^2 = 1
    assert similarity_matching_loss(student, teacher).item() == pytest.approx(1.0)


def test_similarity_loss_needs_at_least_two_texts():
    with pytest.raises(ValueError, match="two"):
        similarity_matching_loss(unit([[1, 0]]), unit([[1, 0]]))


def test_top_neighbours_are_ordered_by_similarity():
    corpus = unit([[1, 0], [0.9, 0.1], [0, 1]])
    queries = unit([[1, 0.05]])
    assert top_neighbours(queries, corpus, k=3) == [[0, 1, 2]]


def test_top_neighbours_can_leave_out_the_query_itself():
    corpus = unit([[1, 0], [0.9, 0.1], [0, 1]])
    assert top_neighbours(corpus, corpus, k=1, exclude_self=True) == [[1], [0], [1]]


def test_k_larger_than_the_corpus_is_an_error():
    with pytest.raises(ValueError, match="k"):
        top_neighbours(unit([[1, 0]]), unit([[1, 0]]), k=5)


def test_retrieval_scores_with_a_hand_checked_example():
    # one query of class 0; neighbours in order have classes 0, 1, 0
    scores = retrieval_scores([[0, 1, 2]], query_labels=[0], corpus_labels=[0, 1, 0])
    assert scores["precision_at_k"] == pytest.approx(2 / 3)
    assert scores["top_1_same_class"] == 1.0
    assert scores["k"] == 3


def test_retrieval_scores_average_over_queries():
    scores = retrieval_scores([[0, 1], [0, 1]], query_labels=[0, 1], corpus_labels=[0, 0])
    assert scores["precision_at_k"] == pytest.approx(0.5)
    assert scores["top_1_same_class"] == pytest.approx(0.5)


def test_neighbour_overlap():
    assert neighbour_overlap([[1, 2, 3]], [[3, 2, 9]]) == pytest.approx(2 / 3)
    assert neighbour_overlap([[1, 2]], [[1, 2]]) == 1.0
    with pytest.raises(ValueError, match="length"):
        neighbour_overlap([[1]], [[1], [2]])


def test_training_pulls_the_student_toward_the_teachers_arrangement(tiny_embedder):
    teacher_model, tokenizer = tiny_embedder
    torch.manual_seed(3)
    student = copy.deepcopy(teacher_model)
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.2)
    teacher_vectors = embed_texts(teacher_model, tokenizer, TEXTS)
    before = similarity_matching_loss(
        embed_texts(student, tokenizer, TEXTS), teacher_vectors
    ).item()
    history = train_embedding_student(
        student,
        tokenizer,
        TEXTS,
        teacher_vectors,
        epochs=40,
        learning_rate=2e-3,
        batch_size=6,
        device="cpu",
    )
    after = similarity_matching_loss(embed_texts(student, tokenizer, TEXTS), teacher_vectors).item()
    assert after < before * 0.5
    assert [row["epoch"] for row in history] == list(range(1, 41))


def test_training_rejects_misaligned_inputs(tiny_embedder):
    model, tokenizer = tiny_embedder
    with pytest.raises(ValueError, match="length"):
        train_embedding_student(model, tokenizer, TEXTS, torch.zeros(2, 4), device="cpu")
