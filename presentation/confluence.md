# DistilBERT to ONNX Runtime: Reference Implementation and Team Rollout Guide

## Executive summary

We validated that our fine-tuned DistilBERT classifier can be exported from a **local-only PyTorch model directory** to **ONNX Runtime** and then served efficiently on **CPU** behind a FastAPI service and OpenShift deployment topology.

This is important because it gives us a practical path to:
- keep the existing **GPU-based training** workflow intact,
- avoid dependence on remote model hosting,
- produce a **portable serving artifact** for CPU environments,
- and standardize a repeatable deployment approach for the team.

The implementation branch has already demonstrated feasibility through working code, passing tests, export logic, parity validation utilities, and deployment assets.

---

## Why we explored this approach

Our target operating environment has a few constraints that strongly influence design:

- **Offline / enterprise-safe environment**: the runtime cannot assume access to external model hubs.
- **GPU for model work, CPU for serving**: the model may be trained or exported on a GPU workstation, but production-style inference is expected to run on CPU infrastructure.
- **Artifact portability matters**: config, weights, tokenizer, and label metadata must stay together.
- **OpenShift is the target deployment pattern**: pods sit behind a Service and Route and can scale horizontally.

Given these constraints, the question was not whether DistilBERT works in general; the question was whether we can create a disciplined, local-only, enterprise-safe export and serving path that the team can own and extend.

The answer is **yes**.

---

## What the experiment proved

The implementation proved the following:

1. A fine-tuned DistilBERT classifier can be exported from a **local saved model directory** into an **ONNX serving artifact**.
2. The serving path can remain **separate from the training-time PyTorch path**, which is operationally cleaner.
3. The ONNX Runtime path is suitable for **CPU-serving scenarios**.
4. The service contract can remain **plain JSON at the boundary**, while tokenization, NumPy conversion, and ONNX inference stay internal.
5. The solution can be packaged in a way that aligns with **FastAPI + OpenShift + HPA** deployment patterns.
6. Validation can be made explicit through **export checks, runtime checks, API tests, and optional parity validation**.

---

## Decision and recommendation

### Recommended architecture decision

Use the following split:

- **PyTorch path** for training, evaluation, and reference behavior
- **ONNX Runtime path** for CPU inference and service deployment

### Why this is the preferred choice

| Option | Offline / local-only fit | Reuse current trained artifact | CPU serving fit | Operational complexity | Recommendation |
|---|---:|---:|---:|---:|---|
| Stay in PyTorch only | Medium | High | Medium | Medium | Useful for training/reference only |
| Convert to TensorFlow | Medium | Low | Medium | High | Not preferred |
| Export to ONNX Runtime | High | High | High | Low-Medium | **Preferred** |

### Key takeaway

ONNX should be treated as the **serving artifact**, not as a replacement for the team’s training-time PyTorch implementation.

---

## High-level architecture

## 1. Training-time path
- Fine-tune DistilBERT on GPU
- Evaluate with the existing PyTorch-based implementation
- Save a complete local model directory

## 2. Export path
- Export from the saved local directory to ONNX
- Preserve tokenizer and label metadata beside the ONNX artifact
- Keep export logic reproducible and CLI-first where possible

## 3. Serving path
- Load ONNX artifact once at service startup
- Accept JSON requests
- Perform tokenization and NumPy conversion inside the service
- Run inference through ONNX Runtime using CPU execution provider
- Return JSON response with class, label, probabilities, and logits

## 4. Deployment path
- Package the service in a container
- Deploy behind OpenShift Route → Service → Deployment/Pods
- Use HPA for horizontal scaling where needed

---

## System architecture details

### Runtime split

This split is intentional and should be preserved:

| Layer | Responsibility |
|---|---|
| Training-time wrapper | Fine-tuning, validation, checkpoint saving, reference behavior |
| Export utility | Converts local saved model directory into ONNX artifact |
| ONNX wrapper | Loads tokenizer + ONNX Runtime session, performs CPU inference |
| FastAPI service | Exposes `/health`, `/predict`, `/predict-batch` |
| OpenShift manifests | Deployment, Service, Route, HPA, ConfigMap |

### Design principle

**Do not merge ONNX loading into the PyTorch model wrapper.**

Keeping the runtime paths separate makes troubleshooting, ownership, rollback, and validation easier.

---

## Artifact lifecycle

The following artifact lifecycle should be treated as the standard:

1. **Base model** available locally
2. **Fine-tuned model** saved as a local directory
3. **ONNX export** produced from that local directory
4. **CPU-serving artifact** deployed into runtime environment

### Artifacts that must stay version-aligned

- `config.json`
- model weights or saved model directory
- tokenizer artifacts
- `label_to_id.json`
- `id_to_label.json`
- `model.onnx`

### Export rule

Export from a **local model directory only**.

### Serve rule

Keep JSON at the service boundary; keep tensors/arrays internal.

---

## Suggested repository structure

```text
classification-engine/
├── config/
├── classifier_runtime/
│   └── models/classification/distilbert/
│       ├── schemas.py
│       ├── pytorch_model.py
│       ├── onnx_model.py
│       ├── export_model.py
│       ├── verify_parity.py
│       ├── inference_service.py
│       └── utils.py
├── tests/
├── docs/
├── openshift/
└── root build files
```

### Why this structure works

- clean separation of concerns
- explicit place for export, parity, and serving logic
- easy to test
- easy to extend for future models or tasks
- friendly for both developers and deployment workflows

---

## API contract and data flow

### Endpoints

| Endpoint | Purpose | Contract |
|---|---|---|
| `GET /health` | Readiness / model loaded check | returns status + model load indicator |
| `POST /predict` | Single-text classification | request → single prediction response |
| `POST /predict-batch` | Batch classification | request → list of prediction responses |

### Data flow

1. Caller sends plain JSON request
2. Service tokenizes input text
3. Service converts tokenizer output to NumPy int64 arrays
4. Arrays are passed into ONNX Runtime session
5. Model returns logits
6. Service computes probabilities and predicted class
7. Service resolves label if metadata is available
8. Service returns JSON response

### Example response fields

- `text`
- `predicted_class`
- `predicted_label`
- `probabilities`
- `logits`

### Design principle

**Keep JSON at the boundary; keep tensors internal.**

---

## Validation strategy

Validation should happen before any production-style rollout.

### Validation gates

| Gate | What is checked | Expected outcome |
|---|---|---|
| Source directory checks | config, weights, tokenizer, label metadata | all required files present |
| Export checks | export completes successfully | ONNX artifact created |
| Output directory checks | ONNX + metadata copied correctly | artifact complete |
| Runtime checks | model loads and predicts | service path healthy |
| API checks | endpoints behave as expected | contracts stable |
| Optional parity checks | PyTorch vs ONNX outputs align within tolerance | confidence in equivalence |

### Recommended parity checks

Use representative inputs such as:
- short text
- medium-length text
- long text near sequence limit
- multi-sentence text
- batch predictions on the same sample set

### Example pass conditions for optional parity

- predicted class matches
- predicted label matches when metadata exists
- logits are close within agreed tolerance
- probabilities are close within agreed tolerance

**Note:** parity is valuable, but it does not need to be the only blocker for rollout. For the current team plan, parity should remain **optional** and can run in parallel.

---

## Test and quality strategy

The implementation already demonstrated a strong validation posture with tests across:

- PyTorch wrapper behavior
- ONNX wrapper behavior
- export workflow
- parity verification
- inference service endpoints

### Quality gates to preserve

- directory validation
- metadata round-trip checks
- export completeness
- single and batch inference tests
- service startup checks
- endpoint contract checks

### Practical takeaway

This is not just a prototype that returns one successful prediction. The branch shows a path for **correctness plus operational hardening**.

---

## OpenShift deployment model

### Target topology

- **Route** for external entry
- **Service** for stable internal access
- **Deployment** to manage pod replicas
- **Inference pods** that load the ONNX model once at startup
- **ConfigMap** for runtime configuration and paths
- **HPA** for horizontal scaling

### Runtime guidance

- readiness should depend on successful model/session load
- liveness can remain simple
- each pod should maintain one loaded ONNX session per model instance
- horizontal scale should be the default CPU-serving approach

### Deployment assets to standardize

- `configmap.yaml`
- `deployment.yaml`
- `service.yaml`
- `route.yaml`
- `hpa.yaml`

---

## Offline / local-only operating model

This is critical for enterprise usage.

### Model and artifact rules

- keep the model as a complete local directory
- load config/tokenizer/model from local files only
- preserve label metadata beside the model artifact
- do not rely on remote model identifiers in runtime code paths

### Environment rules

Use offline-safe settings where relevant:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

### Package delivery rule

The model may be local-only, but package delivery still requires an enterprise-approved path, such as:
- internal package mirror
- wheelhouse
- approved base image

---

## Risks and mitigations

| ID | Risk | Impact | Mitigation |
|---|---|---|---|
| A | Missing label metadata | wrong or missing label resolution | persist and validate label JSON during save/export |
| B | Package availability inside enterprise network | build/runtime delays | use internal mirror, wheelhouse, or base image |
| C | Shape mismatch after export | inference/runtime failures | test long inputs and batched samples early |
| D | Runtime divergence vs PyTorch | trust gap during rollout | use explicit optional parity checks and tolerances |
| E | Pod startup failures | deployment instability | gate readiness on model/session load |

### Summary of risk posture

Most of the real risk is **operational**, not conceptual.

The architecture itself is straightforward. The team’s focus should be on repeatable artifact handling, disciplined validation, and deployment hygiene.

---

## Recommended 3-week rollout plan

This is the proposed team-friendly rollout structure.

### Week 1: export and artifact hardening
- finalize local model save/export rules
- ensure tokenizer and label metadata are preserved
- validate ONNX output completeness
- define artifact directory conventions

### Week 2: CPU runtime and API readiness
- harden ONNX Runtime wrapper
- finalize `/health`, `/predict`, `/predict-batch`
- package service for container usage
- run dev smoke tests in containerized environment

### Week 3: OpenShift deployment and team handoff
- deploy to dev namespace
- validate startup, readiness, and service access
- run basic latency / throughput / memory checks
- document standard operating procedure for future models

### Optional parallel track: parity
- run PyTorch vs ONNX parity checks in parallel when time permits
- use parity as confidence-building evidence, not necessarily as the only blocker

### Decision gates

| Gate | Decision |
|---|---|
| Gate 1 | export is reproducible and artifact is complete |
| Gate 2 | CPU service is healthy in dev |
| Gate 3 | team handoff package is ready |

---

## Recommended operating model for the team

### What should become standard

1. Treat the implementation branch as a **reference implementation**
2. Standardize the artifact contract for all future exported models
3. Keep training-time and serving-time wrappers separate
4. Benchmark CPU behavior explicitly rather than assuming it
5. Roll out in dev/test first, then production once operational criteria are met

### Key decisions the team should align on

- Who owns the model artifact lifecycle?
- What benchmark dataset or replay traffic should represent production load?
- Will ONNX artifacts be baked into the image or mounted at runtime?
- Which metrics and thresholds determine readiness for production?

---

## Conclusion

The primary technical risk has already been retired: **the approach is feasible and implemented**.

The remaining work is not about proving that ONNX Runtime can work. The remaining work is about:
- operational hardening,
- packaging discipline,
- deployment repeatability,
- benchmarking,
- and team standardization.

That is exactly the right next phase for the team.

---

# JIRA / task list for team execution

## Suggested epic

**Epic:** DistilBERT ONNX Runtime CPU Serving Rollout

**Epic goal:** Standardize export, validation, CPU inference service, and OpenShift deployment for the DistilBERT classifier using a local-only, enterprise-safe ONNX Runtime path.

---

## Suggested stories

### Story 1: Persist model metadata for serving artifacts
**Summary:** Ensure saved model directories consistently include tokenizer, config, `label_to_id.json`, and `id_to_label.json`.

**Description:**
Update the training-time save path so every saved model directory contains all metadata required for ONNX export and serving.

**Acceptance criteria:**
- saved model directory includes config and tokenizer artifacts
- `label_to_id.json` is persisted
- `id_to_label.json` is persisted or derivable
- save/load tests cover metadata round-trip

**Suggested tasks:**
- patch save logic
- patch load logic
- add metadata validation test

---

### Story 2: Harden ONNX export utility
**Summary:** Create a repeatable local-only export path from saved model directory to ONNX artifact.

**Description:**
Implement or refine the export utility so it works from local files only and preserves all serving-time metadata in the ONNX output directory.

**Acceptance criteria:**
- export works from local saved model path
- output directory contains `model.onnx`
- tokenizer and metadata are copied to output directory
- export fails with clear message when source directory is incomplete

**Suggested tasks:**
- finalize CLI flow
- add programmatic fallback if needed
- add export completeness checks

---

### Story 3: Implement ONNX Runtime inference wrapper
**Summary:** Build the CPU-serving wrapper around ONNX Runtime.

**Description:**
Implement the ONNX model loader and inference wrapper that tokenizes requests, converts inputs to NumPy arrays, runs ORT inference, and returns structured outputs.

**Acceptance criteria:**
- wrapper loads model once at startup
- uses CPU execution provider by default
- supports single-text and batch inference
- returns class, label, probabilities, and logits
- handles missing label metadata gracefully

**Suggested tasks:**
- implement load/session logic
- implement predict path
- implement batch predict path
- add unit tests

---

### Story 4: Finalize FastAPI service contract
**Summary:** Expose `/health`, `/predict`, and `/predict-batch` endpoints.

**Description:**
Build the service layer using explicit request/response schemas so callers use plain JSON while all tensor logic remains internal.

**Acceptance criteria:**
- `/health` reports readiness correctly
- `/predict` accepts single text and returns valid response
- `/predict-batch` accepts list input and returns valid response
- invalid inputs are handled cleanly
- endpoint tests pass

**Suggested tasks:**
- define schemas
- implement endpoints
- add startup load handling
- add API tests

---

### Story 5: Add optional parity validation workflow
**Summary:** Provide a parity utility to compare PyTorch and ONNX outputs.

**Description:**
Implement an optional validation workflow to compare outputs on representative samples and confirm acceptable closeness.

**Acceptance criteria:**
- representative sample set can be executed through both paths
- predicted class comparison is reported
- logits/probabilities tolerances are configurable
- parity report is saved or printed clearly

**Suggested tasks:**
- build parity script
- define default tolerances
- add representative sample fixture
- document how to run parity

---

### Story 6: Package service for container deployment
**Summary:** Build a container-ready packaging path for the ONNX inference service.

**Description:**
Prepare Docker/build assets so the service can be deployed consistently into OpenShift.

**Acceptance criteria:**
- container image builds successfully
- runtime configuration is externalized where appropriate
- startup command is documented
- image can run locally for smoke testing

**Suggested tasks:**
- finalize Dockerfile
- finalize requirements/build files
- add local container run instructions

---

### Story 7: Create OpenShift deployment assets
**Summary:** Standardize manifests for dev deployment.

**Description:**
Create or harden OpenShift manifests required to deploy the CPU-serving service in a dev namespace.

**Acceptance criteria:**
- deployment manifest exists
- service manifest exists
- route manifest exists
- config map exists
- HPA exists or scaling guidance is documented
- readiness probes reflect model load state

**Suggested tasks:**
- finalize manifests
- externalize config paths
- test deployment in dev namespace

---

### Story 8: Run dev smoke and basic performance checks
**Summary:** Validate service behavior in the target-like environment.

**Description:**
Run a lightweight benchmark and smoke check in dev to confirm service startup, endpoint behavior, and basic CPU performance.

**Acceptance criteria:**
- service starts successfully in dev
- endpoints respond as expected
- basic latency numbers are captured
- basic memory usage is captured
- observations are documented

**Suggested tasks:**
- define smoke dataset
- run functional checks
- capture latency/startup/memory notes
- publish dev validation summary

---

### Story 9: Publish team standard and handoff documentation
**Summary:** Create a standard operating guide for future model export and serving.

**Description:**
Document the reference architecture, artifact contract, validation steps, deployment pattern, and troubleshooting notes so the team can reuse the same pattern.

**Acceptance criteria:**
- Confluence page is published
- export steps are documented
- deployment steps are documented
- troubleshooting section exists
- ownership model is stated

**Suggested tasks:**
- publish Confluence page
- attach deck link and repo link
- add runbook section
- review with team

---

## Suggested execution order

1. Story 1 — metadata persistence
2. Story 2 — export utility
3. Story 3 — ONNX wrapper
4. Story 4 — FastAPI service
5. Story 6 — container packaging
6. Story 7 — OpenShift deployment
7. Story 8 — dev smoke / perf
8. Story 9 — documentation / handoff
9. Story 5 — optional parity in parallel or afterward

---

## Optional sub-task checklist

- [ ] confirm model artifact directory contract
- [ ] confirm offline-only runtime assumptions
- [ ] confirm label metadata persistence
- [ ] confirm ONNX export reproducibility
- [ ] confirm service startup and health behavior
- [ ] confirm OpenShift deployment path in dev
- [ ] capture latency / startup / memory notes
- [ ] publish handoff documentation

