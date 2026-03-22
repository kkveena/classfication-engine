# Claude Code Implementation Brief — DistilBERT ONNX Runtime, Offline, Matching-Style Layout

## 1) Objective

Implement a local-only ONNX export and CPU inference path for a DistilBERT text-classification model.

The implementation must:

- work without internet access to model hosting,
- use only local model directories,
- preserve the current PyTorch reference path for training and validation,
- add a separate ONNX Runtime path for CPU inference,
- follow the same **high-level repository organization pattern** as the referenced sample repository:
  - top-level `config/`
  - one main Python package
  - `tests/`
  - root `README.md`, `pyproject.toml`, and `requirements.txt`
- avoid restricted package or module names.

## 2) Target repository layout

Create or refactor into a neutral layout like this:

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
│   ├── test_export.py
│   ├── test_parity.py
│   └── test_inference_service.py
├── README.md
├── pyproject.toml
└── requirements.txt
```

## 3) Non-negotiable operating constraints

### 3.1 Local-only model access
Do not use remote model identifiers anywhere in code paths intended for enterprise execution.

Always load from a local path and force local-only behavior.

Required environment variables:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Required load pattern:

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

### 3.2 Separate runtime paths
Keep PyTorch and ONNX runtime code separate.

Do not make the PyTorch loader try to read ONNX files.
Do not make the ONNX wrapper try to load PyTorch weight files.

### 3.3 Offline package availability
If package installation is restricted, assume dependencies must come from:

- internal package mirror,
- approved wheelhouse,
- prebuilt image,
- or enterprise base container.

## 4) Required modules

### `config/onnx_inference_config.yaml`
Define:

- `model_dir`
- `onnx_output_dir`
- `max_length`
- `batch_size`
- `provider`
- `parity_atol`
- `parity_rtol`
- `service_host`
- `service_port`

Example:

```yaml
runtime:
  model_dir: ./artifacts/distilbert/final_model
  onnx_output_dir: ./artifacts/distilbert/final_model_onnx
  max_length: 512
  batch_size: 8
  provider: CPUExecutionProvider
  parity_atol: 1.0e-4
  parity_rtol: 1.0e-4
service:
  host: 0.0.0.0
  port: 8080
```

### `classifier_runtime/models/classification/distilbert/schemas.py`
Create Pydantic models for:

- `PredictRequest`
- `PredictBatchRequest`
- `PredictResponse`
- `PredictBatchResponse`
- `ExportRequest`
- `ParityCheckResult`

Suggested response model fields:

```python
text: str
predicted_class: int
predicted_label: str | None = None
probabilities: list[float]
logits: list[float]
```

### `pytorch_model.py`
This is the reference implementation.

Responsibilities:

- load local saved model
- load tokenizer from local saved model directory
- load optional label metadata JSON
- run single prediction
- run batch prediction
- provide output shape used for parity checks

Required functions or methods:

- `load_model()` or equivalent constructor-based load
- `predict(text: str)`
- `batch_predict(texts: list[str], batch_size: int = 8)`
- `save_model(save_path: str)`

### `onnx_model.py`
Create a dedicated ONNX Runtime wrapper.

Suggested class:

```python
class LocalDistilBertOnnxModel:
    ...
```

Responsibilities:

1. load tokenizer from local ONNX directory using `local_files_only=True`
2. load `label_to_id.json` if present
3. create `id_to_label`
4. create `onnxruntime.InferenceSession`
5. use `providers=["CPUExecutionProvider"]` by default
6. tokenize raw text internally
7. convert tokenizer outputs to numpy `int64`
8. run ONNX inference
9. compute softmax probabilities in numpy
10. return JSON-friendly output

Required methods:

- `predict(text: str) -> dict`
- `batch_predict(texts: list[str], batch_size: int = 8) -> list[dict]`

### `export_model.py`
Implement an export utility for a local saved model directory.

Responsibilities:

- validate that the source directory is complete
- run local-only ONNX export
- copy or retain tokenizer files
- copy or retain label metadata JSON
- validate the destination directory

Support CLI args:

- `--model-dir`
- `--output-dir`
- `--task`
- `--opset`
- `--device`

Preferred command:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
optimum-cli export onnx \
  --model ./artifacts/distilbert/final_model \
  --task text-classification \
  --opset 18 \
  --device cuda \
  ./artifacts/distilbert/final_model_onnx
```

If CLI packaging is unavailable but the package exists, provide a programmatic fallback.

### `verify_parity.py`
Implement parity checks between PyTorch and ONNX.

Required checks:

- same predicted class
- same predicted label when metadata exists
- logits within tolerance
- probabilities within tolerance
- batch inference parity

Suggested tolerances:

- `atol=1e-4`
- `rtol=1e-4`

Output should be machine-readable and human-readable.

### `inference_service.py`
Implement a FastAPI service.

Routes:

- `GET /health`
- `POST /predict`
- `POST /predict-batch`

Rules:

- load the ONNX model once at startup
- validate requests with Pydantic
- return JSON-friendly outputs only
- do not expose tensors or numpy arrays directly

### `utils.py`
Provide shared helpers for:

- `load_json(path)`
- `save_json(path, payload)`
- `softmax(logits)`
- `ensure_local_model_dir(path)`
- `copy_required_artifacts(src, dst)`

## 5) Required save-path patch in the PyTorch reference wrapper

Wherever the training/reference wrapper persists the final model, also persist label metadata.

Required behavior:

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

Also update the local loader so it reads `label_to_id.json` when present.

## 6) Export flow requirements

### 6.1 Validate the local source directory
Before export, ensure the source directory contains at minimum:

- `config.json`
- `pytorch_model.bin` or `model.safetensors`
- tokenizer artifacts
- `label_to_id.json`
- `id_to_label.json`

### 6.2 Export from local directory only
No remote lookups are allowed.

### 6.3 Preserve metadata in the ONNX output directory
The ONNX output directory must contain:

- ONNX model artifact
- tokenizer artifacts
- `config.json`
- `label_to_id.json`
- `id_to_label.json`

## 7) ONNX Runtime implementation details

Use raw ONNX Runtime for serving.

Recommended pattern:

```python
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
```

Tokenization:

```python
enc = tokenizer(
    text,
    truncation=True,
    padding="max_length",
    max_length=max_length,
    return_tensors="np",
)
```

Input conversion:

```python
ort_inputs = {
    "input_ids": enc["input_ids"].astype(np.int64),
    "attention_mask": enc["attention_mask"].astype(np.int64),
}
```

If token type ids are present, pass them too.

Post-processing:

- logits = first output tensor
- compute softmax in numpy
- select predicted class via `argmax`
- map class index to label if metadata exists

## 8) Testing requirements

Create tests for:

- local model directory validation
- label metadata round-trip
- ONNX export artifact completeness
- single prediction parity
- batch prediction parity
- FastAPI route behavior
- startup loading failure behavior

Use `pytest` and keep fixtures in `tests/conftest.py`.

## 9) Packaging guidance

`requirements.txt` should include at least:

```text
transformers
optimum[onnx]
onnx
onnxruntime
fastapi
uvicorn
pydantic
numpy
pytest
```

If extras syntax is not supported by the enterprise package source, split the requirement into the exact approved packages.

## 10) Docker guidance

Provide a production image that:

- copies the ONNX artifact directory into the image or mounts it at runtime,
- sets offline environment variables,
- starts the FastAPI service with Uvicorn,
- does not attempt any network fetch during startup.

Example environment section:

```dockerfile
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV PYTHONUNBUFFERED=1
```

## 11) OpenShift guidance

Provide manifests or templates for:

- Deployment
- Service
- Route
- ConfigMap for non-secret runtime config
- Secret only if required for unrelated platform integration
- HorizontalPodAutoscaler if the platform team wants autoscaling

Service runtime behavior:

- one ONNX model load per pod at startup
- readiness endpoint must fail until the model session is ready
- liveness endpoint can be simple process health

## 12) Acceptance criteria

This work is complete when:

1. the local saved model reloads without network access,
2. ONNX export works from a local directory,
3. ONNX output contains model, tokenizer, config, and label metadata,
4. PyTorch and ONNX predictions match within tolerance,
5. the FastAPI service returns stable JSON responses,
6. the package layout follows the required top-level organization pattern,
7. restricted names are not used in package paths, modules, class names, or docs.

## 13) Implementation order

1. create `schemas.py`
2. patch the PyTorch reference wrapper save/load behavior
3. build `onnx_model.py`
4. build `export_model.py`
5. build `verify_parity.py`
6. build `inference_service.py`
7. add tests
8. add Docker and OpenShift assets
9. run final offline validation

## 14) Final rule

Keep the external contract simple:

- request: plain JSON
- internal processing: tokenizer + numpy + ONNX Runtime
- response: plain JSON

Do not require callers to send tensors.
