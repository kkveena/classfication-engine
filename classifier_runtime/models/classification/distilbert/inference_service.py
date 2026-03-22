"""FastAPI inference service for DistilBERT ONNX text classification.

Exposes /health, /predict, and /predict-batch endpoints.
Loads the ONNX model once at startup for efficient serving.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException

from classifier_runtime.models.classification.distilbert.onnx_model import (
    LocalDistilBertOnnxModel,
)
from classifier_runtime.models.classification.distilbert.schemas import (
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

# Global model reference
_model: Optional[LocalDistilBertOnnxModel] = None


def load_config(config_path: str = "config/onnx_inference_config.yaml") -> dict:
    """Load service configuration from YAML file.

    Args:
        config_path: Path to config file.

    Returns:
        Configuration dictionary.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning("Config file not found: %s, using defaults", config_path)
        return {
            "runtime": {
                "onnx_output_dir": "./artifacts/distilbert/final_model_onnx",
                "max_length": 512,
                "batch_size": 8,
                "provider": "CPUExecutionProvider",
            },
            "service": {
                "host": "0.0.0.0",
                "port": 8080,
            },
        }

    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ONNX model at startup."""
    global _model

    config = load_config()
    runtime_cfg = config.get("runtime", {})

    model_dir = runtime_cfg.get(
        "onnx_output_dir", "./artifacts/distilbert/final_model_onnx"
    )
    provider = runtime_cfg.get("provider", "CPUExecutionProvider")
    max_length = runtime_cfg.get("max_length", 512)

    try:
        _model = LocalDistilBertOnnxModel(
            model_dir=model_dir,
            provider=provider,
            max_length=max_length,
        )
        logger.info("ONNX model loaded successfully from: %s", model_dir)
    except Exception as e:
        logger.error("Failed to load ONNX model: %s", e)
        _model = None

    yield

    _model = None
    logger.info("Service shutdown, model released")


app = FastAPI(
    title="DistilBERT Classification Service",
    description="ONNX-based text classification inference service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint.

    Returns 200 if model is loaded, 503 otherwise.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Single text classification endpoint.

    Args:
        request: PredictRequest with text field.

    Returns:
        PredictResponse with classification results.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    try:
        result = _model.predict(request.text)
        return PredictResponse(**result)
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-batch", response_model=PredictBatchResponse)
async def predict_batch(request: PredictBatchRequest):
    """Batch text classification endpoint.

    Args:
        request: PredictBatchRequest with texts field.

    Returns:
        PredictBatchResponse with list of classification results.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    try:
        config = load_config()
        batch_size = config.get("runtime", {}).get("batch_size", 8)
        results = _model.batch_predict(request.texts, batch_size=batch_size)
        predictions = [PredictResponse(**r) for r in results]
        return PredictBatchResponse(predictions=predictions)
    except Exception as e:
        logger.error("Batch prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


def run_server():
    """Run the inference service with uvicorn."""
    import uvicorn

    config = load_config()
    service_cfg = config.get("service", {})

    host = service_cfg.get("host", "0.0.0.0")
    port = service_cfg.get("port", 8080)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    run_server()
