"""Shared test fixtures for DistilBERT classification tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "This is a short test.",
        "Please classify this customer inquiry about their recent order.",
        "I need help with my account billing.",
    ]


@pytest.fixture
def sample_label_to_id():
    """Sample label-to-id mapping."""
    return {
        "billing": 0,
        "general_inquiry": 1,
        "order_status": 2,
    }


@pytest.fixture
def sample_id_to_label(sample_label_to_id):
    """Sample id-to-label mapping."""
    return {str(v): k for k, v in sample_label_to_id.items()}


@pytest.fixture
def tmp_model_dir(tmp_path, sample_label_to_id, sample_id_to_label):
    """Create a temporary model directory with minimal required files."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    # Create minimal config.json
    config = {
        "architectures": ["DistilBertForSequenceClassification"],
        "model_type": "distilbert",
        "num_labels": 3,
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    # Create dummy model file
    (model_dir / "pytorch_model.bin").write_bytes(b"dummy")

    # Create tokenizer files
    tokenizer_config = {"model_type": "distilbert"}
    with open(model_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f)
    (model_dir / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n")

    # Create label metadata
    with open(model_dir / "label_to_id.json", "w") as f:
        json.dump(sample_label_to_id, f)
    with open(model_dir / "id_to_label.json", "w") as f:
        json.dump(sample_id_to_label, f)

    return model_dir


@pytest.fixture
def tmp_onnx_dir(tmp_path, sample_label_to_id, sample_id_to_label):
    """Create a temporary ONNX model directory with minimal required files."""
    onnx_dir = tmp_path / "onnx_model"
    onnx_dir.mkdir()

    # Create dummy ONNX file
    (onnx_dir / "model.onnx").write_bytes(b"dummy_onnx")

    # Create config
    config = {
        "architectures": ["DistilBertForSequenceClassification"],
        "model_type": "distilbert",
        "num_labels": 3,
    }
    with open(onnx_dir / "config.json", "w") as f:
        json.dump(config, f)

    # Create tokenizer files
    tokenizer_config = {"model_type": "distilbert"}
    with open(onnx_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f)
    (onnx_dir / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n")

    # Create label metadata
    with open(onnx_dir / "label_to_id.json", "w") as f:
        json.dump(sample_label_to_id, f)
    with open(onnx_dir / "id_to_label.json", "w") as f:
        json.dump(sample_id_to_label, f)

    return onnx_dir


@pytest.fixture
def mock_pytorch_predict():
    """Mock PyTorch prediction output."""
    return {
        "text": "test input",
        "predicted_class": 1,
        "predicted_label": "general_inquiry",
        "probabilities": [0.1, 0.8, 0.1],
        "logits": [-1.5, 2.3, -1.2],
    }


@pytest.fixture
def mock_onnx_predict():
    """Mock ONNX prediction output matching PyTorch."""
    return {
        "text": "test input",
        "predicted_class": 1,
        "predicted_label": "general_inquiry",
        "probabilities": [0.1001, 0.7999, 0.1000],
        "logits": [-1.5001, 2.2999, -1.2001],
    }
