"""Shared utility helpers for DistilBERT classification."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities from logits.

    Args:
        logits: Raw model output logits, shape (batch, num_classes) or (num_classes,).

    Returns:
        Probability array with same shape as input.
    """
    if logits.ndim == 1:
        logits = logits[np.newaxis, :]

    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def load_json(path: str | Path) -> dict:
    """Load a JSON file from disk.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON as a dictionary.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, payload: dict) -> None:
    """Save a dictionary as a JSON file.

    Args:
        path: Destination file path.
        payload: Data to serialize.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def ensure_local_model_dir(path: str | Path) -> Path:
    """Validate that a local model directory exists and contains required files.

    Args:
        path: Path to the model directory.

    Returns:
        Resolved Path object.

    Raises:
        FileNotFoundError: If directory or required files are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model directory not found: {path}")

    required_files = ["config.json"]
    for req in required_files:
        if not (path / req).exists():
            raise FileNotFoundError(f"Required file not found: {path / req}")

    model_files = ["pytorch_model.bin", "model.safetensors"]
    if not any((path / mf).exists() for mf in model_files):
        raise FileNotFoundError(
            f"No model weight file found in {path}. "
            f"Expected one of: {model_files}"
        )

    tokenizer_files = ["tokenizer.json", "vocab.txt", "tokenizer_config.json"]
    if not any((path / tf).exists() for tf in tokenizer_files):
        raise FileNotFoundError(
            f"No tokenizer file found in {path}. "
            f"Expected at least one of: {tokenizer_files}"
        )

    logger.info("Model directory validated: %s", path)
    return path


def copy_required_artifacts(src: str | Path, dst: str | Path) -> None:
    """Copy tokenizer and label metadata from source to destination directory.

    Args:
        src: Source model directory.
        dst: Destination directory (typically ONNX output).
    """
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    artifact_patterns = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
        "label_to_id.json",
        "id_to_label.json",
        "config.json",
    ]

    for pattern in artifact_patterns:
        src_file = src / pattern
        if src_file.exists():
            shutil.copy2(str(src_file), str(dst / pattern))
            logger.info("Copied %s to %s", pattern, dst)
