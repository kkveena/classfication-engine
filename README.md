# ONNX GPU Export Experiment Guide

## Purpose

This guide is for developers who will validate ONNX export on a GPU-enabled workstation and prepare a DistilBERT text-classification model for later CPU serving.

This version assumes:

- the base model artifacts already exist locally,
- internet access to model hosting is not available,
- all model loading, export, validation, and serving must use local files only,
- the implementation should follow the same **high-level repository organization style** as the referenced sample repository: a top-level `config/` folder, one main Python package, a `tests/` folder, and root build files such as `pyproject.toml` and `requirements.txt`.

## Recommended repository layout

Use a neutral package layout like the following:

```text
project_root/
├── config/
│   └── onnx_inference_config.yaml
├── classifier_runtime/
│   └── models/
│       └── classification/
│           └── distilbert/
│               ├── __init__.py
│               ├── schemas.py
│               ├── pytorch_model.py
│               ├── onnx_model.py
│               ├── export_model.py
│               ├── verify_parity.py
│               ├── inference_service.py
│               └── utils.py
├── tests/
│   ├── conftest.py
│   ├── test_pytorch_model.py
│   ├── test_onnx_model.py
│   ├── test_parity.py
│   └── test_inference_service.py
├── README.md
├── pyproject.toml
└── requirements.txt
```

## File responsibilities

### `config/onnx_inference_config.yaml`
Central configuration for:

- local model directory
- local ONNX output directory
- max sequence length
- batch size
- provider selection
- probability tolerance for parity tests
- service port and health settings

### `classifier_runtime/models/classification/distilbert/schemas.py`
Pydantic models for:

- single inference request
- batch inference request
- single inference response
- batch inference response
- export settings
- validation summary

### `pytorch_model.py`
Reference PyTorch wrapper used for:

- loading the saved local model
- tokenizer loading
- single inference
- batch inference
- parity reference behavior

This file remains the source of truth for training-time behavior.

### `onnx_model.py`
ONNX Runtime wrapper used for:

- loading tokenizer from the local model directory
- loading label metadata from local JSON
- creating `onnxruntime.InferenceSession`
- performing tokenization and numpy conversion
- computing logits, probabilities, predicted index, and predicted label

### `export_model.py`
Offline-safe export script that:

- reads a saved local model directory
- exports to ONNX from local files only
- copies or preserves tokenizer and label metadata in the ONNX directory
- validates that the output directory is complete

### `verify_parity.py`
Compares PyTorch and ONNX outputs on representative samples.

Validation should check:

- same predicted index
- same predicted label when metadata exists
- logits close within tolerance
- probabilities close within tolerance

### `inference_service.py`
FastAPI service exposing:

- `/health`
- `/predict`
- `/predict-batch`

### `utils.py`
Small shared helpers for:

- softmax
- local JSON loading
- local JSON saving
- environment checks
- normalization helpers

## Local-only operating model

All model usage must be local-only.

Required runtime settings:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Required loading behavior:

```python
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

config = AutoConfig.from_pretrained(local_model_dir, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    local_model_dir,
    config=config,
    local_files_only=True,
)
```

Do not use remote model identifiers in export jobs, tests, or serving code.

## Required saved artifacts

Before export, the saved local model directory must include at least:

- `config.json`
- `pytorch_model.bin` or `model.safetensors`
- tokenizer artifacts
- `label_to_id.json`
- `id_to_label.json`

If label metadata is not currently persisted, update the save path first.

## Required save behavior

When persisting the trained model, also persist:

- `label_to_id.json`
- `id_to_label.json`
- `config.label2id`
- `config.id2label`

Example logic:

```python
import json
from pathlib import Path

save_path = Path(save_path)
save_path.mkdir(parents=True, exist_ok=True)

self.model.save_pretrained(save_path)
self.tokenizer.save_pretrained(save_path)

if self.label_to_id is not None:
    with open(save_path / "label_to_id.json", "w", encoding="utf-8") as f:
        json.dump(self.label_to_id, f, indent=2, sort_keys=True)

    id_to_label = {int(v): k for k, v in self.label_to_id.items()}

    with open(save_path / "id_to_label.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id_to_label.items()}, f, indent=2, sort_keys=True)

    self.model.config.label2id = dict(self.label_to_id)
    self.model.config.id2label = {int(k): v for k, v in id_to_label.items()}
    self.model.config.save_pretrained(save_path)
```

## Export workflow

### Step 1: save the final local model

Example local source directory:

```text
artifacts/distilbert/final_model/
```

### Step 2: export on a GPU-enabled workstation

Use the local source directory as input.

Example command:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
optimum-cli export onnx \
  --model ./artifacts/distilbert/final_model \
  --task text-classification \
  --opset 18 \
  --device cuda \
  ./artifacts/distilbert/final_model_onnx
```

Notes:

- export from a local directory only
- keep the task explicit as `text-classification`
- keep tokenizer and label metadata present in the ONNX output directory
- the exported artifact will later run on CPU

### Step 3: validate parity

Run representative checks using:

- short text
- long text near sequence limit
- batch inputs
- class-imbalanced examples

Suggested tolerances:

- `atol=1e-4`
- `rtol=1e-4`

### Step 4: prepare CPU serving

For serving:

- use `onnxruntime.InferenceSession`
- use `providers=["CPUExecutionProvider"]`
- keep JSON at the service boundary
- keep tensors and numpy arrays internal

## Service contract

### Single request

```json
{
  "text": "sample text to classify"
}
```

### Single response

```json
{
  "text": "sample text to classify",
  "predicted_class": 2,
  "predicted_label": "invoice_query",
  "probabilities": [0.01, 0.03, 0.96],
  "logits": [-3.2, -2.1, 4.9]
}
```

### Batch request

```json
{
  "texts": ["text A", "text B"]
}
```

## Acceptance criteria

The experiment is successful when all of the following are true:

1. the local saved model reloads with no network access,
2. ONNX export completes from a local directory,
3. ONNX output includes model, tokenizer, config, and label metadata,
4. PyTorch and ONNX predictions match within tolerance,
5. CPU inference works through the ONNX wrapper,
6. FastAPI endpoints respond correctly,
7. test coverage exists for export, parity, and service behavior.

## Risks and caveats

- Package installation is separate from model availability. Even with local model files, developers still need Python packages available from an internal package source or prebuilt image.
- Dynamic sequence behavior should be validated carefully during export.
- Label metadata must stay version-aligned with the exported model.
- Do not merge ONNX loading into the PyTorch wrapper. Keep them as separate runtime paths.

## Summary

Recommended split:

- PyTorch wrapper for training and reference behavior
- ONNX export script for artifact generation
- ONNX Runtime wrapper for CPU inference
- FastAPI service for deployment
- config-driven layout with a top-level `config/`, a primary Python package, and a dedicated `tests/` folder
# classfication-engine
