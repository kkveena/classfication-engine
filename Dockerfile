FROM python:3.11-slim

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY classifier_runtime/ classifier_runtime/
COPY config/ config/
COPY pyproject.toml .

# Install the package
RUN pip install --no-cache-dir -e .

# Copy ONNX model artifacts (mount or bake in at build time)
# COPY artifacts/distilbert/final_model_onnx/ /app/artifacts/distilbert/final_model_onnx/

EXPOSE 8080

CMD ["uvicorn", "classifier_runtime.models.classification.distilbert.inference_service:app", "--host", "0.0.0.0", "--port", "8080"]
