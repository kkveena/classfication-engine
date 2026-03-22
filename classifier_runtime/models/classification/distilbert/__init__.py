"""DistilBERT classification model package.

Provides PyTorch reference wrapper, ONNX Runtime wrapper,
export utilities, parity verification, and FastAPI inference service.
"""

from classifier_runtime.models.classification.distilbert.pytorch_model import (
    LocalDistilBertModel,
    TextClassificationDataset,
)
from classifier_runtime.models.classification.distilbert.onnx_model import (
    LocalDistilBertOnnxModel,
)
from classifier_runtime.models.classification.distilbert.schemas import (
    PredictRequest,
    PredictBatchRequest,
    PredictResponse,
    PredictBatchResponse,
    ExportRequest,
    ParityCheckResult,
)

__all__ = [
    "LocalDistilBertModel",
    "TextClassificationDataset",
    "LocalDistilBertOnnxModel",
    "PredictRequest",
    "PredictBatchRequest",
    "PredictResponse",
    "PredictBatchResponse",
    "ExportRequest",
    "ParityCheckResult",
]
