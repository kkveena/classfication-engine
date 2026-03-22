# Classification Engine - Project Wiki

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [System Design](#system-design)
  - [Data Flow](#data-flow)
  - [Component Diagram](#component-diagram)
- [Repository Structure](#repository-structure)
- [Codebase Reference](#codebase-reference)
  - [classifier_runtime Package](#classifier_runtime-package)
  - [Configuration](#configuration)
  - [Schemas](#schemas)
  - [PyTorch Model Wrapper](#pytorch-model-wrapper)
  - [ONNX Model Wrapper](#onnx-model-wrapper)
  - [Export Utility](#export-utility)
  - [Parity Verification](#parity-verification)
  - [Inference Service](#inference-service)
  - [Utilities](#utilities)
- [Test Suite](#test-suite)
  - [Test Architecture](#test-architecture)
  - [Fixtures (conftest.py)](#fixtures-conftestpy)
  - [Test Modules](#test-modules)
- [Usage Guide](#usage-guide)
  - [Installation](#installation)
  - [Training a Model](#training-a-model)
  - [Exporting to ONNX](#exporting-to-onnx)
  - [Verifying Parity](#verifying-parity)
  - [Running the Inference Service](#running-the-inference-service)
  - [Running Tests](#running-tests)
- [API Reference](#api-reference)
- [Design Principles](#design-principles)

---

## Overview

The **Classification Engine** (`classifier-runtime`) is a production-grade text classification system built on **DistilBERT**. It provides a complete pipeline from model training through ONNX export to CPU-based inference serving via a FastAPI REST API.

**Key capabilities:**
- Train DistilBERT text classification models with PyTorch
- Export trained models to ONNX format for optimized CPU inference
- Validate PyTorch-to-ONNX parity within configurable tolerances
- Serve predictions through a FastAPI REST service
- Operate entirely offline with local model artifacts (no internet required)

**Tech stack:** Python 3.10+, PyTorch, Hugging Face Transformers, ONNX Runtime, FastAPI, Pydantic, NumPy

---

## Architecture

### System Design

The system follows a **dual-runtime architecture** with strict separation between training-time (PyTorch) and serving-time (ONNX) code paths:

```
                    +-----------------------+
                    |   Training Pipeline   |
                    |   (PyTorch Runtime)    |
                    +-----------+-----------+
                                |
                          save_model()
                                |
                                v
                    +-----------+-----------+
                    |  Local Model Artifacts |
                    |  (config, weights,     |
                    |   tokenizer, labels)   |
                    +-----------+-----------+
                                |
                        export_model.py
                                |
                                v
                    +-----------+-----------+
                    |  ONNX Model Artifacts  |
                    |  (model.onnx, config,  |
                    |   tokenizer, labels)   |
                    +-----------+-----------+
                                |
                       inference_service.py
                                |
                                v
                    +-----------+-----------+
                    |   FastAPI REST API     |
                    |  (ONNX Runtime CPU)    |
                    +-----------------------+
```

### Data Flow

#### Training Flow
```
Raw Text + Labels
    --> TextClassificationDataset (tokenization + label encoding)
    --> DataLoader (batching)
    --> DistilBertForSequenceClassification (forward + backward pass)
    --> save_model() (weights + tokenizer + label metadata)
```

#### Inference Flow
```
HTTP JSON Request {"text": "..."}
    --> FastAPI endpoint (/predict)
    --> AutoTokenizer (text --> numpy arrays)
    --> ort.InferenceSession (ONNX Runtime, CPU)
    --> softmax(logits) --> predicted_class + probabilities
    --> JSON Response
```

#### Export Flow
```
Local PyTorch artifacts
    --> validate_source_directory()
    --> optimum-cli export (or torch.onnx.export fallback)
    --> copy_required_artifacts() (tokenizer, labels, config)
    --> validate_output_directory()
```

### Component Diagram

```
classifier_runtime/
└── models/
    └── classification/
        └── distilbert/
            ├── schemas.py .............. Pydantic data contracts
            ├── pytorch_model.py ....... Training + PyTorch inference (source of truth)
            ├── onnx_model.py .......... ONNX Runtime CPU inference
            ├── export_model.py ........ PyTorch --> ONNX conversion
            ├── verify_parity.py ....... Cross-runtime validation
            ├── inference_service.py ... FastAPI REST endpoints
            └── utils.py ............... Shared helpers (softmax, JSON I/O, file validation)
```

**Dependency graph between modules:**

```
inference_service.py --> onnx_model.py --> utils.py
                     --> schemas.py

verify_parity.py --> pytorch_model.py --> utils.py
                 --> onnx_model.py
                 --> schemas.py

export_model.py --> utils.py
```

---

## Repository Structure

```
classfication-engine/
├── config/
│   └── onnx_inference_config.yaml   # Runtime + service configuration
├── classifier_runtime/              # Main Python package
│   ├── __init__.py
│   └── models/
│       ├── __init__.py
│       └── classification/
│           ├── __init__.py
│           └── distilbert/
│               ├── __init__.py
│               ├── schemas.py           # Pydantic request/response models
│               ├── pytorch_model.py     # PyTorch training + inference wrapper
│               ├── onnx_model.py        # ONNX Runtime inference wrapper
│               ├── export_model.py      # ONNX export utility
│               ├── verify_parity.py     # PyTorch vs ONNX validation
│               ├── inference_service.py # FastAPI service
│               └── utils.py            # Shared helpers
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Shared pytest fixtures
│   ├── test_pytorch_model.py        # PyTorch wrapper tests
│   ├── test_onnx_model.py           # ONNX wrapper tests
│   ├── test_export.py               # Export validation tests
│   ├── test_parity.py               # Parity check tests
│   └── test_inference_service.py    # FastAPI endpoint tests
├── docs/
│   └── wiki.md                      # This file
├── .gitignore
├── pyproject.toml                   # Build config + dependencies + CLI entry points
├── requirements.txt                 # Flat dependency list
└── README.md                        # Experiment guide
```

---

## Codebase Reference

### classifier_runtime Package

The package uses a hierarchical namespace: `classifier_runtime.models.classification.distilbert`. Each level has an `__init__.py` for proper Python package resolution. This structure is designed to accommodate additional model types (e.g., `bert`, `roberta`) or task types (e.g., `ner`, `qa`) in the future.

---

### Configuration

**File:** `config/onnx_inference_config.yaml`

```yaml
runtime:
  model_dir: ./artifacts/distilbert/final_model          # PyTorch model path
  onnx_output_dir: ./artifacts/distilbert/final_model_onnx  # ONNX output path
  max_length: 512                                        # Max token sequence length
  batch_size: 8                                          # Batch processing size
  provider: CPUExecutionProvider                         # ONNX Runtime provider
  parity_atol: 1.0e-4                                    # Absolute tolerance
  parity_rtol: 1.0e-4                                    # Relative tolerance
service:
  host: 0.0.0.0
  port: 8080
```

The config is loaded by `inference_service.py` at startup via `load_config()`. If the file is missing, sensible defaults are used.

---

### Schemas

**File:** `classifier_runtime/models/classification/distilbert/schemas.py`

Defines 6 Pydantic models that enforce type safety at API boundaries:

| Schema | Purpose | Key Fields |
|--------|---------|------------|
| `PredictRequest` | Single classification request | `text: str` |
| `PredictBatchRequest` | Batch classification request | `texts: list[str]` |
| `PredictResponse` | Single classification result | `text`, `predicted_class`, `predicted_label`, `probabilities`, `logits` |
| `PredictBatchResponse` | Batch classification result | `predictions: list[PredictResponse]` |
| `ExportRequest` | ONNX export parameters | `model_dir`, `output_dir`, `task`, `opset`, `device` |
| `ParityCheckResult` | Parity comparison output | `class_match`, `label_match`, `logits_close`, `probabilities_close`, `passed` |

---

### PyTorch Model Wrapper

**File:** `classifier_runtime/models/classification/distilbert/pytorch_model.py`

This is the **source of truth** for training-time behavior.

#### Key Classes

**`TextClassificationDataset(Dataset)`** - Custom PyTorch dataset that:
- Accepts raw text + labels (string or integer)
- Auto-converts string labels to integers with a `label_to_id` mapping
- Tokenizes text with padding and truncation
- Returns tensors ready for DataLoader

**`LocalDistilBertModel`** - Full model lifecycle wrapper:

| Method | Description |
|--------|-------------|
| `__init__(model_path, device)` | Load model from local directory, validate all required files exist |
| `_load_model()` | Load config, tokenizer, weights, and label metadata with `local_files_only=True` |
| `encode_text(text, max_length)` | Tokenize and move to device |
| `predict(text)` | Single inference: returns dict with `predicted_class`, `predicted_label`, `probabilities`, `logits` |
| `batch_predict(texts, batch_size)` | Batched inference with configurable batch size |
| `train_model(...)` | Full training loop with AdamW, linear warmup scheduler, validation, and best-model checkpointing |
| `save_model(save_path)` | Persist weights, tokenizer, `label_to_id.json`, `id_to_label.json`, and updated `config.json` |

#### Device Resolution

`resolve_device(device_override)` follows this priority:
1. Explicit `device_override` parameter ("cpu" or "cuda")
2. `DEVICE_TYPE` environment variable
3. Auto-detect: CUDA if available, else CPU

---

### ONNX Model Wrapper

**File:** `classifier_runtime/models/classification/distilbert/onnx_model.py`

A **separate runtime path** from PyTorch, designed for CPU serving.

**`LocalDistilBertOnnxModel`**:

| Method | Description |
|--------|-------------|
| `__init__(model_dir, provider, max_length)` | Load ONNX session, tokenizer, and label metadata |
| `_tokenize(texts)` | Convert text to numpy int64 arrays for ONNX Runtime |
| `predict(text)` | Single inference via `session.run()` |
| `batch_predict(texts, batch_size)` | Batched inference with chunking |

**Key differences from PyTorch wrapper:**
- Uses `onnxruntime.InferenceSession` instead of PyTorch model
- All tensor operations use NumPy (not torch)
- Softmax is computed via `utils.softmax()` (NumPy-based)
- Automatically discovers `.onnx` files via glob
- Handles optional `token_type_ids` based on model input schema

---

### Export Utility

**File:** `classifier_runtime/models/classification/distilbert/export_model.py`

Converts a locally saved PyTorch model to ONNX format.

#### Export Strategies

1. **Primary: CLI-based** (`export_via_cli`) - Uses `optimum.exporters.onnx` subprocess with offline environment variables
2. **Fallback: Programmatic** (`export_programmatic`) - Uses `torch.onnx.export` directly with dynamic axes

#### Validation Functions

| Function | Purpose |
|----------|---------|
| `validate_source_directory(model_dir)` | Ensures `config.json`, model weights, and tokenizer files exist |
| `validate_output_directory(output_dir)` | Checks for `.onnx` file, `config.json`, and tokenizer files |

#### CLI Entry Point

```bash
export-model --model-dir ./artifacts/distilbert/final_model \
             --output-dir ./artifacts/distilbert/final_model_onnx \
             --task text-classification \
             --opset 18 \
             --device cuda
```

---

### Parity Verification

**File:** `classifier_runtime/models/classification/distilbert/verify_parity.py`

Validates that ONNX outputs match PyTorch outputs within tolerance.

**`check_single_parity(pytorch_result, onnx_result, atol, rtol)`**

Checks for each text sample:
- Same `predicted_class` (argmax match)
- Same `predicted_label` (when labels exist)
- Logits within tolerance (`np.allclose`)
- Probabilities within tolerance (`np.allclose`)

A check **passes** only if all conditions are met.

**`run_parity_checks(...)`** - End-to-end parity test:
1. Loads both PyTorch and ONNX models
2. Runs single predictions on 4 default test samples (short, medium, long, multi-sentence)
3. Runs batch predictions on the same samples
4. Returns summary with pass/fail counts

#### CLI Entry Point

```bash
verify-parity --pytorch-model-dir ./artifacts/distilbert/final_model \
              --onnx-model-dir ./artifacts/distilbert/final_model_onnx \
              --atol 1e-4 --rtol 1e-4 \
              --output parity_results.json
```

---

### Inference Service

**File:** `classifier_runtime/models/classification/distilbert/inference_service.py`

FastAPI application with ONNX model loaded at startup via `lifespan` context manager.

#### Endpoints

| Method | Path | Description | Request Body | Response |
|--------|------|-------------|--------------|----------|
| GET | `/health` | Health check | - | `{"status": "healthy", "model_loaded": true}` |
| POST | `/predict` | Single classification | `{"text": "..."}` | `PredictResponse` |
| POST | `/predict-batch` | Batch classification | `{"texts": ["...", "..."]}` | `PredictBatchResponse` |

**Error responses:**
- `503 Service Unavailable` - Model not loaded
- `422 Unprocessable Entity` - Invalid request body
- `500 Internal Server Error` - Prediction failure

#### CLI Entry Point

```bash
serve   # starts uvicorn on configured host:port
```

---

### Utilities

**File:** `classifier_runtime/models/classification/distilbert/utils.py`

| Function | Description |
|----------|-------------|
| `softmax(logits)` | NumPy softmax with numerical stability (max subtraction). Handles 1D and 2D arrays. |
| `load_json(path)` | Load and parse a JSON file |
| `save_json(path, payload)` | Save dict as JSON with `indent=2`, `sort_keys=True`. Auto-creates parent dirs. |
| `ensure_local_model_dir(path)` | Validate model directory has `config.json`, weight files, and tokenizer files |
| `copy_required_artifacts(src, dst)` | Copy tokenizer, label metadata, and config from source to ONNX output directory |

---

## Test Suite

### Test Architecture

The test suite uses **pytest** with `unittest.mock` for isolating components from heavy ML dependencies. Tests validate logic, contracts, and error handling without requiring actual model weights or GPU resources.

```
tests/
├── conftest.py ..................... Shared fixtures (7 fixtures)
├── test_pytorch_model.py .......... 8 tests across 3 test classes
├── test_onnx_model.py ............. 4 tests across 1 test class
├── test_export.py ................. 7 tests across 2 test classes
├── test_parity.py ................. 4 tests across 1 test class
└── test_inference_service.py ...... 8 tests across 3 test classes
                                    ─────
                                    31 tests total
```

### Fixtures (conftest.py)

| Fixture | Scope | Description |
|---------|-------|-------------|
| `sample_texts` | function | 3 sample classification texts |
| `sample_label_to_id` | function | Label mapping: `{"billing": 0, "general_inquiry": 1, "order_status": 2}` |
| `sample_id_to_label` | function | Reverse mapping derived from `sample_label_to_id` |
| `tmp_model_dir` | function | Temporary directory with minimal PyTorch model files (config, dummy weights, tokenizer, labels) |
| `tmp_onnx_dir` | function | Temporary directory with minimal ONNX model files (dummy `.onnx`, config, tokenizer, labels) |
| `mock_pytorch_predict` | function | Sample PyTorch prediction output dict |
| `mock_onnx_predict` | function | Sample ONNX prediction output dict (slightly offset values for parity testing) |

### Test Modules

#### test_pytorch_model.py (8 tests)

| Class | Test | What It Validates |
|-------|------|-------------------|
| `TestResolveDevice` | `test_explicit_cpu` | `resolve_device("cpu")` returns `"cpu"` |
| | `test_explicit_cuda` | `resolve_device("cuda")` returns `"cuda"` |
| | `test_auto_no_cuda` | Auto-detection falls back to CPU when CUDA unavailable |
| `TestTextClassificationDataset` | `test_string_labels_converted` | String labels get auto-mapped to integers with `label_to_id` dict |
| | `test_int_labels_preserved` | Integer labels pass through unchanged |
| `TestLocalDistilBertModel` | `test_missing_directory_raises` | `FileNotFoundError` on nonexistent path |
| | `test_missing_config_raises` | `FileNotFoundError` when `config.json` is missing |
| | `test_label_metadata_roundtrip` | Label JSON files can be written and read back identically |

#### test_onnx_model.py (4 tests)

| Class | Test | What It Validates |
|-------|------|-------------------|
| `TestLocalDistilBertOnnxModel` | `test_missing_directory_raises` | `FileNotFoundError` on nonexistent path |
| | `test_missing_onnx_file_raises` | `FileNotFoundError` when no `.onnx` file exists |
| | `test_predict_returns_expected_keys` | Mocked predict returns all required fields; argmax is correct |
| | `test_batch_predict_length` | Batch predict returns correct count; per-item argmax is correct |

#### test_export.py (7 tests)

| Class | Test | What It Validates |
|-------|------|-------------------|
| `TestValidateSourceDirectory` | `test_valid_source_directory` | Valid directory passes without error |
| | `test_missing_directory_raises` | Nonexistent path raises `FileNotFoundError` |
| | `test_missing_config_raises` | Missing `config.json` raises error |
| | `test_missing_model_weights_raises` | Missing weight files raises error with descriptive message |
| `TestValidateOutputDirectory` | `test_valid_output_directory` | Complete ONNX directory returns `True` |
| | `test_missing_onnx_file` | Missing `.onnx` file returns `False` |
| | `test_missing_config` | Missing `config.json` returns `False` |

#### test_parity.py (4 tests)

| Class | Test | What It Validates |
|-------|------|-------------------|
| `TestCheckSingleParity` | `test_matching_outputs_pass` | Close outputs within tolerance pass |
| | `test_mismatched_class_fails` | Different predicted classes cause failure |
| | `test_large_logit_diff_fails` | Logit difference exceeding tolerance fails |
| | `test_no_labels_skips_label_check` | `None` labels result in `label_match=None` (skipped) |

#### test_inference_service.py (8 tests)

| Class | Test | What It Validates |
|-------|------|-------------------|
| `TestHealthEndpoint` | `test_health_with_model` | 200 with `{"status": "healthy"}` when model loaded |
| | `test_health_without_model` | 503 when model is `None` |
| `TestPredictEndpoint` | `test_predict_success` | 200 with correct fields from mocked model |
| | `test_predict_missing_text` | 422 on empty request body |
| | `test_predict_no_model` | 503 when model not loaded |
| `TestPredictBatchEndpoint` | `test_batch_predict_success` | 200 with correct prediction count |
| | `test_batch_predict_empty_list` | 200 with empty predictions for empty input |
| | `test_batch_predict_no_model` | 503 when model not loaded |

---

## Usage Guide

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd classfication-engine

# Install with pip (production)
pip install .

# Install with dev dependencies (for testing)
pip install -e ".[dev]"
```

**Python requirement:** 3.10+

### Training a Model

```python
from classifier_runtime.models.classification.distilbert.pytorch_model import (
    LocalDistilBertModel,
)

# Load a pretrained DistilBERT model from a local directory
model = LocalDistilBertModel(
    model_path="./artifacts/distilbert/base_model",
    device="cuda",  # or "cpu", or None for auto-detect
)

# Train on your data
history = model.train_model(
    train_texts=["I need a refund", "Where is my order?", ...],
    train_labels=["billing", "order_status", ...],
    val_texts=["Help with payment", ...],
    val_labels=["billing", ...],
    epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    save_path="./artifacts/distilbert",
    save_best_only=True,
    monitor_metric="val_accuracy",
)

# The trained model is saved to:
#   ./artifacts/distilbert/best_model/
#   ./artifacts/distilbert/final_model/
```

### Exporting to ONNX

**Option 1: CLI command** (installed via `pyproject.toml`)

```bash
# Set offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Export
export-model \
    --model-dir ./artifacts/distilbert/final_model \
    --output-dir ./artifacts/distilbert/final_model_onnx \
    --task text-classification \
    --opset 18 \
    --device cuda
```

**Option 2: Python API**

```python
from classifier_runtime.models.classification.distilbert.export_model import (
    export_via_cli,
)

success = export_via_cli(
    model_dir="./artifacts/distilbert/final_model",
    output_dir="./artifacts/distilbert/final_model_onnx",
    task="text-classification",
    opset=18,
    device="cuda",
)
```

### Verifying Parity

**CLI:**

```bash
verify-parity \
    --pytorch-model-dir ./artifacts/distilbert/final_model \
    --onnx-model-dir ./artifacts/distilbert/final_model_onnx \
    --atol 1e-4 --rtol 1e-4 \
    --output parity_results.json
```

**Python API:**

```python
from classifier_runtime.models.classification.distilbert.verify_parity import (
    run_parity_checks,
)

summary = run_parity_checks(
    pytorch_model_dir="./artifacts/distilbert/final_model",
    onnx_model_dir="./artifacts/distilbert/final_model_onnx",
    test_texts=["Custom test sentence 1", "Custom test sentence 2"],
    atol=1e-4,
    rtol=1e-4,
)

print(f"Passed: {summary['passed']}/{summary['total_checks']}")
```

### Running the Inference Service

**CLI:**

```bash
# Uses config/onnx_inference_config.yaml for model path and port
serve
```

**Python:**

```bash
python -m classifier_runtime.models.classification.distilbert.inference_service
```

**Direct uvicorn:**

```bash
uvicorn classifier_runtime.models.classification.distilbert.inference_service:app \
    --host 0.0.0.0 --port 8080
```

**Making requests:**

```bash
# Health check
curl http://localhost:8080/health

# Single prediction
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "I need help with my billing"}'

# Batch prediction
curl -X POST http://localhost:8080/predict-batch \
    -H "Content-Type: application/json" \
    -d '{"texts": ["billing question", "order status check"]}'
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test module
python -m pytest tests/test_inference_service.py -v

# Run a specific test class
python -m pytest tests/test_parity.py::TestCheckSingleParity -v

# Run with coverage (requires pytest-cov)
python -m pytest tests/ --cov=classifier_runtime --cov-report=term-missing
```

**Expected output:**

```
tests/test_export.py ............. 7 passed
tests/test_inference_service.py .. 8 passed
tests/test_onnx_model.py ......... 4 passed
tests/test_parity.py ............. 4 passed
tests/test_pytorch_model.py ...... 8 passed
============================== 31 passed ==============================
```

---

## API Reference

### POST /predict

Classify a single text input.

**Request:**
```json
{
  "text": "I need help with my account billing"
}
```

**Response (200):**
```json
{
  "text": "I need help with my account billing",
  "predicted_class": 0,
  "predicted_label": "billing",
  "probabilities": [0.92, 0.05, 0.03],
  "logits": [3.8, -1.2, -2.1]
}
```

### POST /predict-batch

Classify multiple texts in one request.

**Request:**
```json
{
  "texts": [
    "I need help with billing",
    "Where is my order?"
  ]
}
```

**Response (200):**
```json
{
  "predictions": [
    {
      "text": "I need help with billing",
      "predicted_class": 0,
      "predicted_label": "billing",
      "probabilities": [0.92, 0.05, 0.03],
      "logits": [3.8, -1.2, -2.1]
    },
    {
      "text": "Where is my order?",
      "predicted_class": 2,
      "predicted_label": "order_status",
      "probabilities": [0.02, 0.08, 0.90],
      "logits": [-2.5, -0.8, 3.5]
    }
  ]
}
```

### GET /health

**Response (200):** `{"status": "healthy", "model_loaded": true}`

**Response (503):** `{"detail": "Model not loaded"}`

---

## Design Principles

1. **Offline-first** - All model loading uses `local_files_only=True`. Environment variables `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are enforced during export. No remote model registry access at any point.

2. **Separate runtime paths** - PyTorch and ONNX code are completely independent modules. The PyTorch wrapper is the source of truth for training behavior. The ONNX wrapper is the serving path. They are never mixed.

3. **JSON at the boundary, tensors inside** - External API uses only JSON (via Pydantic schemas). Internal processing uses NumPy arrays (ONNX path) or PyTorch tensors (training path). This keeps serialization concerns at the edges.

4. **Validate before proceeding** - `ensure_local_model_dir()` checks for required files before loading. `validate_source_directory()` and `validate_output_directory()` gate export operations. Errors are surfaced early with descriptive messages.

5. **Config-driven** - Runtime parameters (model paths, batch sizes, tolerances, service port) are centralized in `config/onnx_inference_config.yaml` rather than scattered as hardcoded values.

6. **Parity as a first-class concern** - The `verify_parity.py` module and `test_parity.py` tests ensure that the ONNX export faithfully reproduces PyTorch behavior. Tolerances are explicit and configurable.

7. **Testable without GPUs or real models** - All tests use mocks and temporary directories. No actual model weights, CUDA, or network access required. The full suite runs in ~3 seconds.
