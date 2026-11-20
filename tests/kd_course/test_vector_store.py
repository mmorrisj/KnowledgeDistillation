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
