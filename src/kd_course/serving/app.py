"""A FastAPI service around a trained student classifier."""

from __future__ import annotations

import logging
import threading
import time
from typing import Annotated, Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StringConstraints

logger = logging.getLogger(__name__)

MAX_BATCH = 64  # texts per request
MAX_CHARS = 2000  # characters per text

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_CHARS)]


class Predictor(Protocol):
    """Anything that can turn texts into (label index, confidence) pairs."""

    name: str

    def predict(self, texts: list[str]) -> list[tuple[int, float]]: ...


class ClassifyRequest(BaseModel):
    texts: list[Text] = Field(min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    label: str
    confidence: float


class ClassifyResponse(BaseModel):
    predictions: list[Prediction]
    model: str
    latency_ms: float


def create_student_app(predictor: Predictor, label_names: list[str]) -> FastAPI:
    """Build the service. The model is loaded once, before this is called, and reused."""
    app = FastAPI(title="Distilled student", docs_url="/docs", redoc_url=None)
    # One request at a time reaches the model. A GPU does not run two batches faster than one,
    # and this keeps memory use predictable. Throughput comes from batching inside a request.
    model_lock = threading.Lock()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/info")
    def info() -> dict[str, object]:
        return {
            "model": predictor.name,
            "labels": label_names,
            "max_batch": MAX_BATCH,
            "max_chars": MAX_CHARS,
        }

    # A plain `def` endpoint runs in a worker thread, so a slow model call does not
    # freeze the server's event loop and /health keeps answering.
    @app.post("/classify")
    def classify(request: ClassifyRequest) -> ClassifyResponse:
        start = time.perf_counter()
        try:
            with model_lock:
                results = predictor.predict(request.texts)
            predictions = [
                Prediction(label=label_names[index], confidence=round(float(confidence), 4))
                for index, confidence in results
            ]
        except Exception:
            # Log the cause for the operator; tell the caller nothing about internals.
            logger.exception("The model failed on a request of %d texts", len(request.texts))
            raise HTTPException(
                status_code=500, detail="The model failed to answer this request."
            ) from None
        return ClassifyResponse(
            predictions=predictions,
            model=predictor.name,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    return app


def run() -> None:
    """Console entry point: `kd-student-serve`.

    Settings come from the environment:
    KD_STUDENT_DIR (default: results/students/classifier), KD_STUDENT_DEVICE (cuda or cpu),
    KD_STUDENT_QUANTIZE (1 to load an 8-bit model on CPU), HOST, PORT.
    """
    import os

    import uvicorn

    from kd_course.results import default_results_dir
    from kd_course.serving.export import load_classifier

    folder = os.environ.get("KD_STUDENT_DIR") or str(
        default_results_dir() / "students" / "classifier"
    )
    quantize = os.environ.get("KD_STUDENT_QUANTIZE", "0") == "1"
    device = os.environ.get("KD_STUDENT_DEVICE", "cpu" if quantize else "cuda")
    try:
        predictor, labels = load_classifier(folder, device=device, quantize=quantize)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    uvicorn.run(
        create_student_app(predictor, labels),
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8100")),
    )
