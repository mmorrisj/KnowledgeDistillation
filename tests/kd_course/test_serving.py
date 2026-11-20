import pytest
from fastapi.testclient import TestClient

from kd_course.serving.app import MAX_BATCH, MAX_CHARS, create_student_app

LABELS = ["card_arrival", "exchange_rate"]


class FakePredictor:
    name = "fake-student"

    def __init__(self):
        self.calls = []

    def predict(self, texts):
        self.calls.append(list(texts))
        return [(1, 0.9) if "rate" in text else (0, 0.75) for text in texts]


@pytest.fixture
def predictor():
    return FakePredictor()


@pytest.fixture
def client(predictor):
    return TestClient(create_student_app(predictor, LABELS))


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_info_names_the_model_and_its_labels(client):
    info = client.get("/info").json()
    assert info["model"] == "fake-student"
    assert info["labels"] == LABELS
    assert info["max_batch"] == MAX_BATCH


def test_classify_returns_a_label_and_confidence_for_each_text(client):
    response = client.post(
        "/classify", json={"texts": ["where is my card", "what rate do you use"]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == [
        {"label": "card_arrival", "confidence": 0.75},
        {"label": "exchange_rate", "confidence": 0.9},
    ]
    assert body["model"] == "fake-student"
    assert body["latency_ms"] >= 0


def test_classify_sends_the_whole_request_to_the_model_as_one_batch(client, predictor):
    client.post("/classify", json={"texts": ["a", "b", "c"]})
    assert predictor.calls == [["a", "b", "c"]]


@pytest.mark.parametrize(
    "payload",
    [
        {"texts": []},
        {"texts": ["x"] * (MAX_BATCH + 1)},
        {"texts": ["x" * (MAX_CHARS + 1)]},
        {"texts": [""]},
        {"texts": ["   "]},
        {"texts": "not a list"},
        {},
    ],
)
def test_classify_rejects_bad_requests(client, payload):
    assert client.post("/classify", json=payload).status_code == 422


def test_a_model_failure_is_a_500_without_internal_details(predictor):
    def boom(texts):
        raise RuntimeError("CUDA exploded at /secret/path")

    predictor.predict = boom
    client = TestClient(create_student_app(predictor, LABELS), raise_server_exceptions=False)
    response = client.post("/classify", json={"texts": ["a"]})
    assert response.status_code == 500
    assert "secret" not in response.text
    assert response.json() == {"detail": "The model failed to answer this request."}


def test_a_label_index_out_of_range_is_a_500(predictor):
    predictor.predict = lambda texts: [(7, 0.5) for _ in texts]
    client = TestClient(create_student_app(predictor, LABELS), raise_server_exceptions=False)
    assert client.post("/classify", json={"texts": ["a"]}).status_code == 500
