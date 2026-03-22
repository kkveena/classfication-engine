"""Tests for the ONNX export utility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from classifier_runtime.models.classification.distilbert.export_model import (
    validate_output_directory,
    validate_source_directory,
)
from classifier_runtime.models.classification.distilbert.utils import (
    ensure_local_model_dir,
)


class TestValidateSourceDirectory:
    """Tests for source directory validation."""

    def test_valid_source_directory(self, tmp_model_dir):
        """Should not raise for a valid directory."""
        validate_source_directory(tmp_model_dir)

    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            validate_source_directory(Path("/nonexistent/model"))

    def test_missing_config_raises(self, tmp_path):
        model_dir = tmp_path / "bad_model"
        model_dir.mkdir()
        (model_dir / "pytorch_model.bin").write_bytes(b"dummy")
        (model_dir / "vocab.txt").write_text("[PAD]\n")

        with pytest.raises(FileNotFoundError):
            validate_source_directory(model_dir)

    def test_missing_model_weights_raises(self, tmp_path):
        model_dir = tmp_path / "no_weights"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type": "distilbert"}')
        (model_dir / "vocab.txt").write_text("[PAD]\n")

        with pytest.raises(FileNotFoundError, match="No model weight file"):
            validate_source_directory(model_dir)


class TestValidateOutputDirectory:
    """Tests for ONNX output directory validation."""

    def test_valid_output_directory(self, tmp_onnx_dir):
        assert validate_output_directory(tmp_onnx_dir) is True

    def test_missing_onnx_file(self, tmp_path):
        out_dir = tmp_path / "no_onnx"
        out_dir.mkdir()
        (out_dir / "config.json").write_text("{}")
        (out_dir / "vocab.txt").write_text("[PAD]\n")

        assert validate_output_directory(out_dir) is False

    def test_missing_config(self, tmp_path):
        out_dir = tmp_path / "no_config"
        out_dir.mkdir()
        (out_dir / "model.onnx").write_bytes(b"dummy")
        (out_dir / "vocab.txt").write_text("[PAD]\n")

        assert validate_output_directory(out_dir) is False
