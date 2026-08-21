from __future__ import annotations

from src.retrieval.pipeline import RAGPipeline


def test_model_unavailable_error_is_retryable() -> None:
    error = RuntimeError(
        "Error code: 404 - model_not_found: This model is unavailable for free."
    )

    assert RAGPipeline._is_model_availability_error(error)


def test_unrelated_error_is_not_retryable() -> None:
    assert not RAGPipeline._is_model_availability_error(RuntimeError("Qdrant unavailable"))
