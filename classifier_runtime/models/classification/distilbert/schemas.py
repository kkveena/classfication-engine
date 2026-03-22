"""Pydantic schemas for DistilBERT classification requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Single text classification request."""

    text: str = Field(..., description="Input text to classify")


class PredictBatchRequest(BaseModel):
    """Batch text classification request."""

    texts: list[str] = Field(..., description="List of input texts to classify")


class PredictResponse(BaseModel):
    """Single text classification response."""

    text: str
    predicted_class: int
    predicted_label: str | None = None
    probabilities: list[float]
    logits: list[float]


class PredictBatchResponse(BaseModel):
    """Batch text classification response."""

    predictions: list[PredictResponse]


class ExportRequest(BaseModel):
    """ONNX export request settings."""

    model_dir: str = Field(..., description="Path to local PyTorch model directory")
    output_dir: str = Field(..., description="Path to write ONNX artifacts")
    task: str = Field(default="text-classification", description="Export task type")
    opset: int = Field(default=18, description="ONNX opset version")
    device: str = Field(default="cuda", description="Device for export (cuda or cpu)")


class ParityCheckResult(BaseModel):
    """Result of a parity comparison between PyTorch and ONNX outputs."""

    text: str
    pytorch_predicted_class: int
    onnx_predicted_class: int
    class_match: bool
    label_match: bool | None = None
    logits_close: bool
    probabilities_close: bool
    max_logit_diff: float
    max_probability_diff: float
    passed: bool
