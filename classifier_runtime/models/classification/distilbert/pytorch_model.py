"""PyTorch reference wrapper for local DistilBERT text classification.

This module is the source of truth for training-time behavior.
It loads a locally saved DistilBERT model and provides single and batch
inference, as well as training and model persistence.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)

from classifier_runtime.models.classification.distilbert.utils import (
    load_json,
    save_json,
)

logger = logging.getLogger(__name__)

# Default model subdirectory relative to project
_MODEL_DIR = Path("artifacts") / "distilbert" / "final_model"

# Device type constants
DEVICE_TYPE_CPU = "cpu"
DEVICE_TYPE_CUDA = "cuda"
DEVICE_TYPE_AUTO = "auto"

DEVICE_TYPE = os.environ.get("DEVICE_TYPE", DEVICE_TYPE_AUTO)


def resolve_device(device_override: Optional[str] = None) -> str:
    """Resolve the compute device to use for model operations.

    Priority order:
        1. device_override passed explicitly by the caller
        2. DEVICE_TYPE environment variable (via os.environ)
        3. Automatic detection: CUDA when available, otherwise CPU.

    Args:
        device_override: Explicit device string ("cpu", "cuda", or None).

    Returns:
        Resolved device string ("cpu" or "cuda").
    """
    if device_override is not None and device_override in (
        DEVICE_TYPE_CPU,
        DEVICE_TYPE_CUDA,
    ):
        resolved = device_override
    elif DEVICE_TYPE != DEVICE_TYPE_AUTO:
        resolved = DEVICE_TYPE
    else:
        resolved = DEVICE_TYPE_CUDA if torch.cuda.is_available() else DEVICE_TYPE_CPU

    logger.info(
        "Device resolved to %s (override=%s, env_DEVICE_TYPE=%s, cuda_available=%s)",
        resolved,
        device_override,
        DEVICE_TYPE,
        torch.cuda.is_available(),
    )
    return resolved


class TextClassificationDataset(Dataset):
    """Custom dataset for text classification."""

    def __init__(
        self,
        texts: List[str],
        labels: List,
        tokenizer,
        max_length: int = 512,
        label_to_id: Optional[Dict] = None,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.labels, self.label_to_id = self._convert_labels_to_int(
            labels, label_to_id
        )

    def _convert_labels_to_int(
        self,
        labels: List,
        case_insensitive: bool = True,
        label_to_id: Optional[Dict] = None,
    ):
        """Convert string labels to integers."""
        if not labels:
            return labels

        if case_insensitive and any(
            isinstance(label, str) for label in labels
        ):
            labels = [label.lower() for label in labels]

        # Check if labels are strings
        if isinstance(labels[0], str):
            if label_to_id is None:
                # Create label mapping
                unique_labels = sorted(list(set(labels)))
                label_to_id = {
                    label: idx
                    for idx, label in enumerate(unique_labels)
                }

            logger.info("Label mapping: %s", label_to_id)
            # Convert to integers
            return [label_to_id[label] for label in labels], label_to_id

        # If already integers, ensure they're ints
        return [int(label) for label in labels], None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class LocalDistilBertModel:
    """Wrapper class for loading DistilBERT model from local directory."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
    ):
        """Initialize DistilBERT model from local directory.

        Args:
            model_path: Path to local model directory.
            device: Device to load model on ('cpu', 'cuda', or None for auto).
        """
        self.model_path = Path(model_path)
        self.device = resolve_device(device)

        # Load model components
        self.config = None
        self.tokenizer = None
        self.model = None
        self.label_to_id: Optional[Dict[str, int]] = None

        self._load_model()

    def _load_model(self):
        """Load model, tokenizer, and config from local directory."""
        try:
            # Check if model directory exists
            if not self.model_path.exists():
                logger.error("Model path not found: %s", self.model_path)
                raise FileNotFoundError(
                    f"Model directory not found: {self.model_path}"
                )

            # Load configuration
            config_path = self.model_path / "config.json"
            if not config_path.exists():
                logger.error("Config file not found: %s", config_path)
                raise FileNotFoundError(
                    f"Model directory not found: {config_path}"
                )
            self.config = AutoConfig.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            logger.info("Loaded config from: %s", config_path)

            # Load tokenizer
            tokenizer_files = ["tokenizer.json", "vocab.txt", "tokenizer_config.json"]
            if not any(
                (self.model_path / f).exists() for f in tokenizer_files
            ):
                logger.error("Tokenizer file(s) not found: %s", tokenizer_files)
                raise FileNotFoundError(
                    f"Tokenizer files not found: {tokenizer_files}"
                )
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            logger.info("Loaded tokenizer from: %s", self.model_path)

            # Load model
            model_files = ["pytorch_model.bin", "model.safetensors"]
            if not any(
                (self.model_path / f).exists() for f in model_files
            ):
                logger.error("Model file(s) not found: %s", model_files)
                raise FileNotFoundError(
                    f"Model file(s) not found: {model_files}"
                )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path),
                config=self.config,
                local_files_only=True,
            )
            logger.info("Loaded model from: %s", self.model_path)

            # Move model to device
            self.model.to(self.device)
            logger.info("Model loaded on device: %s", self.device)

            # Load label metadata if present
            label_path = self.model_path / "label_to_id.json"
            if label_path.exists():
                self.label_to_id = load_json(label_path)
                logger.info("Loaded label_to_id from: %s", label_path)

        except Exception as e:
            logger.error("Error loading model: %s", e)
            raise

    def encode_text(
        self,
        text: str,
        max_length: int = 512,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Encode input text for model inference.

        Args:
            text: Input text to encode.
            max_length: Maximum sequence length.
            return_tensors: Format of return tensors.

        Returns:
            Dictionary with encoded inputs.
        """
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors=return_tensors,
        )

        # Move tensors to model device
        if return_tensors == "pt":
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

        return encoded

    def predict(self, text: str) -> Dict[str, any]:
        """Make prediction on input text.

        Args:
            text: Input text for classification.

        Returns:
            Dictionary with predictions and probabilities.
        """
        # Encode text
        inputs = self.encode_text(text)

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)

        # Build id_to_label mapping
        id_to_label = None
        if self.label_to_id is not None:
            id_to_label = {int(v): k for k, v in self.label_to_id.items()}

        predicted_label = None
        if id_to_label is not None:
            predicted_label = id_to_label.get(predicted_class.item())

        return {
            "text": text,
            "predicted_class": predicted_class.cpu().item(),
            "predicted_label": predicted_label,
            "probabilities": probabilities.cpu().numpy().tolist()[0],
            "logits": logits.cpu().numpy().tolist()[0],
        }

    def batch_predict(
        self, texts: List[str], batch_size: int = 8
    ) -> List[Dict]:
        """Make predictions on batch of texts.

        Args:
            texts: List of input texts.
            batch_size: Size of processing batches.

        Returns:
            List of prediction dictionaries.
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Encode batch
            inputs = self.tokenizer(
                batch_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1)

            # Build id_to_label mapping
            id_to_label = None
            if self.label_to_id is not None:
                id_to_label = {int(v): k for k, v in self.label_to_id.items()}

            # Process batch results
            for j in range(len(batch_texts)):
                predicted_label = None
                if id_to_label is not None:
                    predicted_label = id_to_label.get(
                        predicted_classes[j].cpu().item()
                    )

                results.append(
                    {
                        "text": batch_texts[j],
                        "predicted_class": predicted_classes[j].cpu().item(),
                        "predicted_label": predicted_label,
                        "probabilities": probabilities[j].cpu().numpy().tolist(),
                        "logits": logits[j].cpu().numpy().tolist(),
                    }
                )

        return results

    def _reconfigure_model_for_labels(self, num_labels: int):
        """Reconfigure model for different number of labels."""
        logger.info("Reconfiguring model: %s labels", num_labels)

        # Update configuration
        self.config.num_labels = num_labels

        # Create new model with updated config
        self.model = DistilBertForSequenceClassification(config=self.config)
        self.model.to(self.device)

        logger.info("Model reconfigured for %s labels", num_labels)

    def train_model(
        self,
        train_texts: List[str],
        train_labels: List,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        max_length: int = 512,
        save_path: Optional[str] = None,
        save_best_only: bool = True,
        monitor_metric: str = "val_accuracy",
    ) -> Dict[str, any]:
        """Train the DistilBERT model for classification.

        Args:
            train_texts: List of training texts.
            train_labels: List of training labels.
            val_texts: List of validation texts (optional).
            val_labels: List of validation labels (optional).
            epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate for optimizer.
            warmup_steps: Number of warmup steps for scheduler.
            max_length: Maximum sequence length.
            save_path: Path to save the trained model.
            save_best_only: If True, save only when model improves.
            monitor_metric: Metric to monitor for best model.

        Returns:
            Dictionary with training history and metrics.
        """
        logger.info("Starting model training...")

        # Create datasets
        train_dataset = TextClassificationDataset(
            train_texts, train_labels, self.tokenizer, max_length
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        self.label_to_id = train_dataset.label_to_id

        num_labels = len(
            set(train_dataset.label_to_id.values())
        ) if train_dataset.label_to_id is not None else len(set(train_labels))

        self._reconfigure_model_for_labels(num_labels)

        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_dataset = TextClassificationDataset(
                val_texts,
                val_labels,
                self.tokenizer,
                max_length,
                train_dataset.label_to_id,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )

        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate
        )
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training history
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "best_epoch": 0,
            "best_metric_value": None,
            "label_to_id": train_dataset.label_to_id,
        }

        # Initialize best metric tracking
        if monitor_metric.endswith("_loss"):
            best_metric = float("inf")
            is_better = lambda new, best: new < best
        else:
            best_metric = 0.0
            is_better = lambda new, best: new > best

        # Training loop
        for epoch in range(epochs):
            logger.info("Epoch %d/%d", epoch + 1, epochs)

            # Training phase
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, scheduler
            )
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)

            logger.info(
                "Train loss: %.4f, Train Accuracy: %.4f",
                train_loss,
                train_acc,
            )

            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(val_loader)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)

                logger.info(
                    "Val loss: %.4f, Val Accuracy: %.4f",
                    val_loss,
                    val_acc,
                )

            # Determine current metric value for comparison
            current_metrics = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss if val_loader is not None else None,
                "val_accuracy": val_acc if val_loader is not None else None,
            }

            current_metric_value = current_metrics.get(monitor_metric)

            # Check if current metric is available
            if current_metric_value is None:
                logger.warning(
                    "Monitor metric '%s' not available, using train_accuracy instead.",
                    monitor_metric,
                )
                monitor_metric = "train_accuracy"
                current_metric_value = train_acc

            if epoch == 0:
                best_metric = 0.0
                is_better = lambda new, best: new > best

            # Check if this is the best model so far
            model_improved = is_better(current_metric_value, best_metric)

            if model_improved:
                best_metric = current_metric_value
                history["best_epoch"] = epoch + 1
                history["best_metric_value"] = current_metric_value

                logger.info(
                    "*** New best model (%s): %.4f ***",
                    monitor_metric,
                    current_metric_value,
                )

                # Save best model checkpoint
                if save_path is not None:
                    if save_best_only:
                        best_model_path = Path(save_path) / "best_model"
                        self.save_model(str(best_model_path))
                        logger.info(
                            "Best model saved to: %s", best_model_path
                        )
                    else:
                        # Save both best and epoch checkpoint
                        best_model_path = Path(save_path) / "best_model"
                        checkpoint_path = (
                            Path(save_path) / f"checkpoint_epoch_{epoch}"
                        )
                        self.save_model(str(best_model_path))
                        self.save_model(str(checkpoint_path))
                        logger.info(
                            "Best model saved to: %s", best_model_path
                        )
                        logger.info(
                            "Checkpoint saved to: %s", checkpoint_path
                        )
            else:
                logger.info(
                    "No improvement. Current (%s): %.4f, Best: %.4f",
                    monitor_metric,
                    current_metric_value,
                    best_metric,
                )
                # Save checkpoint even if not best (optional)
                if save_path is not None and not save_best_only:
                    checkpoint_path = (
                        Path(save_path) / f"checkpoint_epoch_{epoch}"
                    )
                    self.save_model(str(checkpoint_path))
                    logger.info(
                        "Checkpoint saved to: %s", checkpoint_path
                    )

        # Save final model (always save the last epoch)
        if save_path is not None:
            final_path = Path(save_path) / "final_model"
            self.save_model(str(final_path))
            logger.info("Final model saved to: %s", final_path)

            # Log best model info
            logger.info("Training completed!")
            logger.info(
                "Best model was at epoch %d with (%s): %.4f",
                history["best_epoch"],
                monitor_metric,
                history["best_metric_value"],
            )

        return history

    def _train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(train_loader, desc="Training")

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(all_labels, all_predictions)

        return avg_loss, accuracy

    def _validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")

            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                logits = outputs.logits

                # Track metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(val_loader)
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(all_labels, all_predictions)

        return avg_loss, accuracy

    def save_model(self, save_path: str) -> None:
        """Save model, tokenizer, and label metadata to local directory.

        Args:
            save_path: Directory to save model artifacts.
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        if self.label_to_id is not None:
            with open(
                save_path / "label_to_id.json", "w", encoding="utf-8"
            ) as f:
                json.dump(self.label_to_id, f, indent=2, sort_keys=True)

            id_to_label = {
                int(v): k for k, v in self.label_to_id.items()
            }

            with open(
                save_path / "id_to_label.json", "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {str(k): v for k, v in id_to_label.items()},
                    f,
                    indent=2,
                    sort_keys=True,
                )

            self.model.config.label2id = dict(self.label_to_id)
            self.model.config.id2label = {
                int(k): v for k, v in id_to_label.items()
            }
            self.model.config.save_pretrained(save_path)

        logger.info("Model saved to: %s", save_path)
