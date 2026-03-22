"""Tests for the ONNX Runtime model wrapper."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from classifier_runtime.models.classification.distilbert.onnx_model import (
    LocalDistilBertOnnxModel,
)


class TestLocalDistilBertOnnxModel:
    """Tests for ONNX model loading and inference."""

    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError, match="ONNX model directory not found"):
            LocalDistilBertOnnxModel(model_dir="/nonexistent/onnx_path")

    def test_missing_onnx_file_raises(self, tmp_path):
        onnx_dir = tmp_path / "no_onnx"
        onnx_dir.mkdir()
        (onnx_dir / "config.json").write_text("{}")
        (onnx_dir / "vocab.txt").write_text("[PAD]\n")

        with pytest.raises(FileNotFoundError, match="No .onnx file found"):
            LocalDistilBertOnnxModel(model_dir=str(onnx_dir))

    @patch("classifier_runtime.models.classification.distilbert.onnx_model.ort.InferenceSession")
    @patch("classifier_runtime.models.classification.distilbert.onnx_model.AutoTokenizer.from_pretrained")
    def test_predict_returns_expected_keys(
        self, mock_tokenizer_cls, mock_session_cls, tmp_onnx_dir
    ):
        """Test that predict returns all expected fields."""
        # Setup mock session
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input_ids"
        mock_input2 = MagicMock()
        mock_input2.name = "attention_mask"
        mock_session.get_inputs.return_value = [mock_input, mock_input2]

        mock_output = MagicMock()
        mock_output.name = "logits"
        mock_session.get_outputs.return_value = [mock_output]

        # Return logits shape (1, 3)
        mock_session.run.return_value = [np.array([[1.0, 2.0, -1.0]])]
        mock_session_cls.return_value = mock_session

        # Setup mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
        }
        mock_tokenizer_cls.return_value = mock_tokenizer

        model = LocalDistilBertOnnxModel(model_dir=str(tmp_onnx_dir))
        result = model.predict("test text")

        assert "text" in result
        assert "predicted_class" in result
        assert "predicted_label" in result
        assert "probabilities" in result
        assert "logits" in result
        assert result["predicted_class"] == 1  # argmax of [1.0, 2.0, -1.0]

    @patch("classifier_runtime.models.classification.distilbert.onnx_model.ort.InferenceSession")
    @patch("classifier_runtime.models.classification.distilbert.onnx_model.AutoTokenizer.from_pretrained")
    def test_batch_predict_length(
        self, mock_tokenizer_cls, mock_session_cls, tmp_onnx_dir
    ):
        """Test that batch_predict returns correct number of results."""
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input_ids"
        mock_input2 = MagicMock()
        mock_input2.name = "attention_mask"
        mock_session.get_inputs.return_value = [mock_input, mock_input2]

        mock_output = MagicMock()
        mock_output.name = "logits"
        mock_session.get_outputs.return_value = [mock_output]

        mock_session.run.return_value = [
            np.array([[1.0, 2.0, -1.0], [0.5, -0.5, 3.0]])
        ]
        mock_session_cls.return_value = mock_session

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1], [1, 1, 1]], dtype=np.int64),
        }
        mock_tokenizer_cls.return_value = mock_tokenizer

        model = LocalDistilBertOnnxModel(model_dir=str(tmp_onnx_dir))
        results = model.batch_predict(["text a", "text b"], batch_size=4)

        assert len(results) == 2
        assert results[0]["predicted_class"] == 1
        assert results[1]["predicted_class"] == 2
