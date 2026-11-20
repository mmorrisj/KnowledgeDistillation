import os

import pytest

from kd_course.vector_store import MemoryStore, open_store

VECTORS = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
TEXTS = ["card lost", "card stolen", "exchange rate", "top up"]
LABELS = [0, 0, 1, 2]
DATABASE_URL = os.environ.get("KD_COURSE_DATABASE_URL")


def fill(store):
    store.create()
    store.add(ids=[10, 11, 12, 13], texts=TEXTS, embeddings=VECTORS, labels=LABELS)
    return store


@pytest.fixture(params=["memory", "postgres"])
def store(request):
    if request.param == "memory":
        yield fill(MemoryStore(dim=3))
        return
    if not DATABASE_URL:
        pytest.skip("Set KD_COURSE_DATABASE_URL to run the pgvector tests.")
    from kd_course.vector_store import PgVectorStore

    pg = PgVectorStore(DATABASE_URL, table="kd_test_vectors", dim=3)
    pg.drop()
    yield fill(pg)
    pg.drop()


def test_count(store):
    assert store.count() == 4


def test_search_returns_nearest_first_with_text_label_and_distance(store):
    hits = store.search([1.0, 0.05, 0.0], k=2)
    assert [hit.id for hit in hits] == [10, 11]
    assert hits[0].text == "card lost" and hits[0].label == 0
    assert 0 <= hits[0].distance < hits[1].distance < 0.1


def test_search_ignores_vector_length(store):
    # Cosine distance compares direction: a longer query pointing the same way gives the same order.
    assert [hit.id for hit in store.search([5.0, 0.25, 0.0], k=2)] == [10, 11]


def test_search_with_k_larger_than_the_store_returns_everything(store):
    assert len(store.search([1.0, 0.0, 0.0], k=50)) == 4


def test_a_vector_of_the_wrong_size_is_rejected(store):
    with pytest.raises(ValueError, match="3"):
        store.search([1.0, 0.0], k=1)
    with pytest.raises(ValueError, match="3"):
        store.add(ids=[99], texts=["x"], embeddings=[[1.0, 0.0]], labels=[0])


def test_adding_mismatched_lists_is_rejected(store):
    with pytest.raises(ValueError, match="length"):
        store.add(ids=[1, 2], texts=["x"], embeddings=[[1.0, 0.0, 0.0]], labels=[0])


def test_index_does_not_change_results_on_a_tiny_table(store):
    before = [hit.id for hit in store.search([0.0, 1.0, 0.1], k=3)]
    store.create_index()
    assert [hit.id for hit in store.search([0.0, 1.0, 0.1], k=3)] == before


def test_size_in_bytes_is_positive(store):
    assert store.size_bytes() > 0


def test_create_again_starts_empty(store):
    store.create()
    assert store.count() == 0


def test_open_store_uses_memory_when_no_database_is_configured(monkeypatch):
    monkeypatch.delenv("KD_COURSE_DATABASE_URL", raising=False)
    assert isinstance(open_store(dim=3), MemoryStore)


@pytest.mark.parametrize("name", ["bad name", "drop;table", "", "1starts_with_digit", "x" * 70])
def test_unsafe_table_names_are_rejected(name):
    from kd_course.vector_store import PgVectorStore

    with pytest.raises(ValueError, match="table"):
        PgVectorStore("postgresql+psycopg://u:p@localhost/db", table=name, dim=3)


def test_open_store_uses_postgres_when_configured():
    if not DATABASE_URL:
        pytest.skip("Set KD_COURSE_DATABASE_URL to run the pgvector tests.")
    from kd_course.vector_store import PgVectorStore

    opened = open_store(dim=3, table="kd_test_open")
    assert isinstance(opened, PgVectorStore)
    opened.drop()


def test_search_in_an_empty_store_finds_nothing():
    empty = MemoryStore(dim=3)
    empty.create()
    assert empty.search([1.0, 0.0, 0.0], k=5) == []


def test_search_effort_is_accepted_by_both_stores(store):
    assert [hit.id for hit in store.search([1.0, 0.05, 0.0], k=2, search_effort=200)] == [10, 11]


@pytest.mark.parametrize("effort", [0, -5, 5000, 2.5, "40; DROP TABLE x"])
def test_an_invalid_search_effort_is_rejected(store, effort):
    with pytest.raises(ValueError, match="search_effort"):
        store.search([1.0, 0.0, 0.0], k=1, search_effort=effort)


def test_more_search_effort_finds_more_true_neighbours():
    if not DATABASE_URL:
        pytest.skip("Set KD_COURSE_DATABASE_URL to run the pgvector tests.")
    import numpy as np

    from kd_course.embeddings import neighbour_overlap
    from kd_course.vector_store import PgVectorStore

    rng = np.random.default_rng(0)
    n, dim = 4000, 32
    vectors = rng.normal(size=(n, dim)).astype("float32")
    queries = rng.normal(size=(40, dim)).tolist()
    pg = PgVectorStore(DATABASE_URL, table="kd_test_effort", dim=dim)
    pg.create()
    pg.add(list(range(n)), ["t"] * n, vectors.tolist(), [0] * n)
    exact = [[hit.id for hit in pg.search(q, k=10)] for q in queries]
    pg.create_index()
    low = neighbour_overlap(
        exact, [[hit.id for hit in pg.search(q, k=10, search_effort=10)] for q in queries]
    )
    high = neighbour_overlap(
        exact, [[hit.id for hit in pg.search(q, k=10, search_effort=400)] for q in queries]
    )
    pg.drop()
    assert high > low
    assert high > 0.95


def hit(row_id, distance):
    from kd_course.vector_store import Hit

    return Hit(row_id, "t", 0, distance)


def test_index_recall_counts_results_as_good_as_the_true_ones():
    from kd_course.vector_store import index_recall

    exact = [[hit(1, 0.10), hit(2, 0.20), hit(3, 0.30)]]
    assert index_recall(exact, [[hit(1, 0.10), hit(2, 0.20), hit(3, 0.30)]]) == 1.0
    assert index_recall(exact, [[hit(1, 0.10), hit(2, 0.20), hit(9, 0.90)]]) == pytest.approx(2 / 3)


def test_index_recall_treats_ties_as_correct():
    from kd_course.vector_store import index_recall

    # Row 7 is a different row at exactly the same distance as row 3: an equally good answer.
    exact = [[hit(1, 0.10), hit(2, 0.20), hit(3, 0.30)]]
    assert index_recall(exact, [[hit(1, 0.10), hit(2, 0.20), hit(7, 0.30)]]) == 1.0


def test_index_recall_rejects_mismatched_input():
    from kd_course.vector_store import index_recall

    with pytest.raises(ValueError, match="length"):
        index_recall([[hit(1, 0.1)]], [])
