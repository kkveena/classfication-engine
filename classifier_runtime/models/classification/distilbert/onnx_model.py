"""ONNX Runtime wrapper for local DistilBERT text classification.

This module provides CPU inference using a previously exported ONNX model.
It is a separate runtime path from the PyTorch reference wrapper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from classifier_runtime.models.classification.distilbert.utils import (
    load_json,
    softmax,
)

logger = logging.getLogger(__name__)


class LocalDistilBertOnnxModel:
    """ONNX Runtime wrapper for DistilBERT text classification.

    Loads an exported ONNX model and provides CPU inference
    with tokenization, softmax, and label mapping.
    """

    def __init__(
        self,
        model_dir: str,
        provider: str = "CPUExecutionProvider",
        max_length: int = 512,
    ):
        """Initialize ONNX model from local directory.

        Args:
            model_dir: Path to directory containing ONNX model and tokenizer.
            provider: ONNX Runtime execution provider.
            max_length: Maximum sequence length for tokenization.
        """
        self.model_dir = Path(model_dir)
        self.provider = provider
        self.max_length = max_length

        self.session: Optional[ort.InferenceSession] = None
        self.tokenizer = None
        self.label_to_id: Optional[Dict[str, int]] = None
        self.id_to_label: Optional[Dict[int, str]] = None

        self._load_model()

    def _load_model(self) -> None:
        """Load ONNX model, tokenizer, and label metadata from local directory."""
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"ONNX model directory not found: {self.model_dir}"
            )

        # Find ONNX model file
        onnx_files = list(self.model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(
                f"No .onnx file found in {self.model_dir}"
            )
        onnx_path = str(onnx_files[0])
        logger.info("Loading ONNX model from: %s", onnx_path)

        # Create inference session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=[self.provider],
        )
        logger.info(
            "ONNX session created with provider: %s", self.provider
        )

        # Load tokenizer from local directory
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True
        )
        logger.info("Loaded tokenizer from: %s", self.model_dir)

        # Load label metadata if present
        label_path = self.model_dir / "label_to_id.json"
        if label_path.exists():
            self.label_to_id = load_json(label_path)
            self.id_to_label = {
                int(v): k for k, v in self.label_to_id.items()
            }
            logger.info("Loaded label metadata: %d labels", len(self.label_to_id))

        # Log model input/output info
        input_names = [inp.name for inp in self.session.get_inputs()]
        output_names = [out.name for out in self.session.get_outputs()]
        logger.info("Model inputs: %s", input_names)
        logger.info("Model outputs: %s", output_names)

    def _tokenize(
        self, texts: str | List[str]
    ) -> Dict[str, np.ndarray]:
        """Tokenize input text(s) and convert to numpy arrays.

        Args:
            texts: Single text string or list of texts.

        Returns:
            Dictionary of numpy int64 arrays for ONNX Runtime.
        """
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="np",
        )

        ort_inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }

        # Pass token_type_ids if the model expects them
        model_input_names = {inp.name for inp in self.session.get_inputs()}
        if (
            "token_type_ids" in model_input_names
            and "token_type_ids" in encoded
        ):
            ort_inputs["token_type_ids"] = encoded[
                "token_type_ids"
            ].astype(np.int64)

        return ort_inputs

    def predict(self, text: str) -> Dict:
        """Run single text classification inference.

        Args:
            text: Input text to classify.

        Returns:
            Dictionary with predicted_class, predicted_label,
            probabilities, and logits.
        """
        ort_inputs = self._tokenize(text)

        # Run inference
        output_names = [out.name for out in self.session.get_outputs()]
        outputs = self.session.run(output_names, ort_inputs)

        logits = outputs[0]  # shape: (1, num_classes)
        probabilities = softmax(logits)

        predicted_class = int(np.argmax(probabilities, axis=-1)[0])
        logits_list = logits[0].tolist()
        probs_list = probabilities[0].tolist()

        predicted_label = None
        if self.id_to_label is not None:
            predicted_label = self.id_to_label.get(predicted_class)

        return {
            "text": text,
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "probabilities": probs_list,
            "logits": logits_list,
        }

    def batch_predict(
        self, texts: List[str], batch_size: int = 8
    ) -> List[Dict]:
        """Run batch text classification inference.

        Args:
            texts: List of input texts to classify.
            batch_size: Size of processing batches.

        Returns:
            List of prediction dictionaries.
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            ort_inputs = self._tokenize(batch_texts)

            output_names = [out.name for out in self.session.get_outputs()]
            outputs = self.session.run(output_names, ort_inputs)

            logits = outputs[0]  # shape: (batch, num_classes)
            probabilities = softmax(logits)
            predicted_classes = np.argmax(probabilities, axis=-1)

            for j in range(len(batch_texts)):
                predicted_label = None
                if self.id_to_label is not None:
                    predicted_label = self.id_to_label.get(
                        int(predicted_classes[j])
                    )

                results.append(
                    {
                        "text": batch_texts[j],
                        "predicted_class": int(predicted_classes[j]),
                        "predicted_label": predicted_label,
                        "probabilities": probabilities[j].tolist(),
                        "logits": logits[j].tolist(),
                    }
                )

        return results
