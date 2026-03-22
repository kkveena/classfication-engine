"""Tests for the PyTorch reference model wrapper."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from classifier_runtime.models.classification.distilbert.pytorch_model import (
    LocalDistilBertModel,
    TextClassificationDataset,
    resolve_device,
)


class TestResolveDevice:
    """Tests for device resolution logic."""

    def test_explicit_cpu(self):
        assert resolve_device("cpu") == "cpu"

    def test_explicit_cuda(self):
        assert resolve_device("cuda") == "cuda"

    @patch("classifier_runtime.models.classification.distilbert.pytorch_model.torch.cuda.is_available", return_value=False)
    @patch("classifier_runtime.models.classification.distilbert.pytorch_model.DEVICE_TYPE", "auto")
    def test_auto_no_cuda(self, mock_cuda):
        assert resolve_device() == "cpu"


class TestTextClassificationDataset:
    """Tests for the custom dataset class."""

    def test_string_labels_converted(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }

        ds = TextClassificationDataset(
            texts=["text a", "text b", "text c"],
            labels=["cat", "dog", "cat"],
            tokenizer=mock_tokenizer,
            max_length=32,
        )

        assert ds.label_to_id is not None
        assert "cat" in ds.label_to_id
        assert "dog" in ds.label_to_id
        assert len(ds) == 3

    def test_int_labels_preserved(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }

        ds = TextClassificationDataset(
            texts=["text a", "text b"],
            labels=[0, 1],
            tokenizer=mock_tokenizer,
            max_length=32,
        )

        assert ds.labels == [0, 1]


class TestLocalDistilBertModel:
    """Tests for model loading and validation."""

    def test_missing_directory_raises(self):
        with pytest.raises(FileNotFoundError):
            LocalDistilBertModel(model_path="/nonexistent/path", device="cpu")

    def test_missing_config_raises(self, tmp_path):
        model_dir = tmp_path / "empty_model"
        model_dir.mkdir()
        (model_dir / "pytorch_model.bin").write_bytes(b"dummy")
        (model_dir / "vocab.txt").write_text("[PAD]\n")

        with pytest.raises(FileNotFoundError):
            LocalDistilBertModel(model_path=str(model_dir), device="cpu")

    def test_label_metadata_roundtrip(self, tmp_path):
        """Verify label metadata is saved and can be read back."""
        label_to_id = {"billing": 0, "order": 1, "general": 2}

        save_dir = tmp_path / "saved_model"
        save_dir.mkdir()

        # Save label files
        with open(save_dir / "label_to_id.json", "w") as f:
            json.dump(label_to_id, f, indent=2, sort_keys=True)

        id_to_label = {int(v): k for k, v in label_to_id.items()}
        with open(save_dir / "id_to_label.json", "w") as f:
            json.dump({str(k): v for k, v in id_to_label.items()}, f, indent=2, sort_keys=True)

        # Read back
        with open(save_dir / "label_to_id.json") as f:
            loaded = json.load(f)

        assert loaded == label_to_id

        with open(save_dir / "id_to_label.json") as f:
            loaded_id = json.load(f)

        assert loaded_id == {str(k): v for k, v in id_to_label.items()}
