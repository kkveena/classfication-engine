"""Tests for the FastAPI inference service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from classifier_runtime.models.classification.distilbert import inference_service


@pytest.fixture
def mock_model():
    """Create a mock ONNX model."""
    model = MagicMock()
    model.predict.return_value = {
        "text": "test text",
        "predicted_class": 1,
        "predicted_label": "general_inquiry",
        "probabilities": [0.1, 0.8, 0.1],
        "logits": [-1.5, 2.3, -1.2],
    }
    model.batch_predict.return_value = [
        {
            "text": "text a",
            "predicted_class": 0,
            "predicted_label": "billing",
            "probabilities": [0.9, 0.05, 0.05],
            "logits": [3.0, -1.0, -1.0],
        },
        {
            "text": "text b",
            "predicted_class": 2,
            "predicted_label": "order_status",
            "probabilities": [0.05, 0.05, 0.9],
            "logits": [-1.0, -1.0, 3.0],
        },
    ]
    return model


@pytest.fixture
def client(mock_model):
    """Create test client with mocked model."""
    inference_service._model = mock_model
    client = TestClient(inference_service.app)
    yield client
    inference_service._model = None


class TestHealthEndpoint:
    def test_health_with_model(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_without_model(self):
        inference_service._model = None
        client = TestClient(inference_service.app)
        response = client.get("/health")
        assert response.status_code == 503


class TestPredictEndpoint:
    def test_predict_success(self, client):
        response = client.post("/predict", json={"text": "test text"})
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "test text"
        assert data["predicted_class"] == 1
        assert data["predicted_label"] == "general_inquiry"
        assert "probabilities" in data
        assert "logits" in data

    def test_predict_missing_text(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_no_model(self):
        inference_service._model = None
        client = TestClient(inference_service.app)
        response = client.post("/predict", json={"text": "test"})
        assert response.status_code == 503


class TestPredictBatchEndpoint:
    def test_batch_predict_success(self, client):
        response = client.post(
            "/predict-batch", json={"texts": ["text a", "text b"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2

    def test_batch_predict_empty_list(self, client, mock_model):
        mock_model.batch_predict.return_value = []
        response = client.post("/predict-batch", json={"texts": []})
        assert response.status_code == 200

    def test_batch_predict_no_model(self):
        inference_service._model = None
        client = TestClient(inference_service.app)
        response = client.post(
            "/predict-batch", json={"texts": ["test"]}
        )
        assert response.status_code == 503
