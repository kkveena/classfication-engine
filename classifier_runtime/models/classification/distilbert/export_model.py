"""ONNX export utility for local DistilBERT text classification model.

Exports a locally saved PyTorch model to ONNX format for CPU serving.
All operations are offline — no remote model lookups are performed.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from classifier_runtime.models.classification.distilbert.utils import (
    copy_required_artifacts,
    ensure_local_model_dir,
)

logger = logging.getLogger(__name__)


def validate_source_directory(model_dir: Path) -> None:
    """Validate that the source directory contains all required artifacts.

    Args:
        model_dir: Path to the local model directory.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    ensure_local_model_dir(model_dir)

    # Check label metadata
    label_files = ["label_to_id.json", "id_to_label.json"]
    for lf in label_files:
        if not (model_dir / lf).exists():
            logger.warning(
                "Label metadata file not found: %s. "
                "Export will proceed but labels may not be available in ONNX output.",
                model_dir / lf,
            )


def validate_output_directory(output_dir: Path) -> bool:
    """Validate the ONNX output directory is complete.

    Args:
        output_dir: Path to the ONNX output directory.

    Returns:
        True if directory contains required files.
    """
    required = ["config.json"]
    onnx_files = list(output_dir.glob("*.onnx"))

    if not onnx_files:
        logger.error("No .onnx file found in output directory: %s", output_dir)
        return False

    for req in required:
        if not (output_dir / req).exists():
            logger.error("Required file missing from output: %s", req)
            return False

    # Check tokenizer
    tokenizer_files = ["tokenizer.json", "vocab.txt", "tokenizer_config.json"]
    if not any((output_dir / tf).exists() for tf in tokenizer_files):
        logger.error("No tokenizer files found in output directory")
        return False

    logger.info("Output directory validation passed: %s", output_dir)
    return True


def export_via_cli(
    model_dir: str,
    output_dir: str,
    task: str = "text-classification",
    opset: int = 18,
    device: str = "cuda",
) -> bool:
    """Export model to ONNX using optimum-cli.

    Args:
        model_dir: Path to local PyTorch model directory.
        output_dir: Path to write ONNX artifacts.
        task: Export task type.
        opset: ONNX opset version.
        device: Device for export (cuda or cpu).

    Returns:
        True if export succeeded.
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)

    # Validate source
    validate_source_directory(model_dir)

    # Set offline environment
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "optimum.exporters.onnx",
        "--model",
        str(model_dir),
        "--task",
        task,
        "--opset",
        str(opset),
        "--device",
        device,
        str(output_dir),
    ]

    logger.info("Running export command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("Export stdout: %s", result.stdout)
        if result.stderr:
            logger.warning("Export stderr: %s", result.stderr)

    except subprocess.CalledProcessError as e:
        logger.error("CLI export failed: %s", e.stderr)
        logger.info("Falling back to programmatic export...")
        return export_programmatic(
            str(model_dir), str(output_dir), task, opset, device
        )

    # Copy label metadata and tokenizer artifacts
    copy_required_artifacts(model_dir, output_dir)

    # Validate output
    if not validate_output_directory(output_dir):
        logger.error("Output validation failed after export")
        return False

    logger.info("ONNX export completed successfully to: %s", output_dir)
    return True


def export_programmatic(
    model_dir: str,
    output_dir: str,
    task: str = "text-classification",
    opset: int = 18,
    device: str = "cpu",
) -> bool:
    """Export model to ONNX programmatically as a fallback.

    Args:
        model_dir: Path to local PyTorch model directory.
        output_dir: Path to write ONNX artifacts.
        task: Export task type (for documentation).
        opset: ONNX opset version.
        device: Device for export.

    Returns:
        True if export succeeded.
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load model and tokenizer locally
        config = AutoConfig.from_pretrained(
            str(model_dir), local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), config=config, local_files_only=True
        )

        export_device = torch.device(device)
        model.to(export_device)
        model.eval()

        # Create dummy inputs
        dummy_text = "This is a sample text for ONNX export."
        encoded = tokenizer(
            dummy_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        dummy_input_ids = encoded["input_ids"].to(export_device)
        dummy_attention_mask = encoded["attention_mask"].to(export_device)

        onnx_path = output_dir / "model.onnx"

        # Export
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            str(onnx_path),
            opset_version=opset,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
        )

        logger.info("Programmatic ONNX export completed: %s", onnx_path)

        # Copy artifacts
        copy_required_artifacts(model_dir, output_dir)

        # Save tokenizer to output dir
        tokenizer.save_pretrained(str(output_dir))

        # Validate
        if not validate_output_directory(output_dir):
            logger.error("Output validation failed after programmatic export")
            return False

        return True

    except Exception as e:
        logger.error("Programmatic export failed: %s", e)
        return False


def main():
    """CLI entry point for ONNX export."""
    parser = argparse.ArgumentParser(
        description="Export DistilBERT model to ONNX format"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to local PyTorch model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to write ONNX artifacts",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="text-classification",
        help="Export task type",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for export (cuda or cpu)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    success = export_via_cli(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        task=args.task,
        opset=args.opset,
        device=args.device,
    )

    if success:
        logger.info("Export completed successfully!")
        sys.exit(0)
    else:
        logger.error("Export failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
