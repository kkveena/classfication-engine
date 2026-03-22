"""Tests for parity verification between PyTorch and ONNX outputs."""

from __future__ import annotations

import numpy as np
import pytest

from classifier_runtime.models.classification.distilbert.verify_parity import (
    check_single_parity,
)


class TestCheckSingleParity:
    """Tests for single-prediction parity checks."""

    def test_matching_outputs_pass(self, mock_pytorch_predict, mock_onnx_predict):
        result = check_single_parity(
            mock_pytorch_predict, mock_onnx_predict, atol=1e-3, rtol=1e-3
        )
        assert result.passed is True
        assert result.class_match is True
        assert result.logits_close is True
        assert result.probabilities_close is True

    def test_mismatched_class_fails(self):
        pt = {
            "text": "test",
            "predicted_class": 0,
            "predicted_label": "billing",
            "probabilities": [0.9, 0.05, 0.05],
            "logits": [3.0, -1.0, -1.0],
        }
        ox = {
            "text": "test",
            "predicted_class": 1,
            "predicted_label": "general_inquiry",
            "probabilities": [0.1, 0.8, 0.1],
            "logits": [-1.0, 3.0, -1.0],
        }
        result = check_single_parity(pt, ox)
        assert result.passed is False
        assert result.class_match is False

    def test_large_logit_diff_fails(self):
        pt = {
            "text": "test",
            "predicted_class": 1,
            "predicted_label": "general_inquiry",
            "probabilities": [0.1, 0.8, 0.1],
            "logits": [-1.5, 2.3, -1.2],
        }
        ox = {
            "text": "test",
            "predicted_class": 1,
            "predicted_label": "general_inquiry",
            "probabilities": [0.1, 0.8, 0.1],
            "logits": [-1.5, 2.5, -1.2],  # 0.2 diff in second logit
        }
        result = check_single_parity(pt, ox, atol=1e-4, rtol=1e-4)
        assert result.passed is False
        assert result.logits_close is False

    def test_no_labels_skips_label_check(self):
        pt = {
            "text": "test",
            "predicted_class": 0,
            "predicted_label": None,
            "probabilities": [0.9, 0.05, 0.05],
            "logits": [3.0, -1.0, -1.0],
        }
        ox = {
            "text": "test",
            "predicted_class": 0,
            "predicted_label": None,
            "probabilities": [0.9, 0.05, 0.05],
            "logits": [3.0, -1.0, -1.0],
        }
        result = check_single_parity(pt, ox)
        assert result.passed is True
        assert result.label_match is None
