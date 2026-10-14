FROM python:3.11-slim

LABEL org.opencontainers.image.title="Explanations Evaluator" \
      org.opencontainers.image.description="Scores explanation faithfulness as mean SHAP Drop@10" \
      org.opencontainers.image.vendor="Landseer Project" \
      org.opencontainers.image.source="https://github.com/landseer-project/landseer" \
      org.opencontainers.image.base.name="python:3.11-slim" \
      org.landseer.schema="1" \
      org.landseer.type="evaluator" \
      org.landseer.name="explanations" \
      org.landseer.datasets="cifar10,celeba" \
      org.landseer.modality="image" \
      org.landseer.framework="numpy==1.24.0" \
      org.landseer.python="3.11" \
      org.landseer.accelerator="cpu" \
      org.landseer.metrics="drop10_score" \
      org.landseer.required_artifacts="" \
      org.landseer.defense_types="explainability" \
      org.landseer.dependencies="numpy==1.24.0"

WORKDIR /app

RUN pip install --no-cache-dir numpy==1.24.0

COPY explanations/evaluate.py ./evaluate.py

ENTRYPOINT ["python", "evaluate.py"]

# FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# LABEL org.opencontainers.image.title="Explanations Evaluator"
# LABEL org.landseer.type="evaluator"
# LABEL org.landseer.metrics="drop10_score"

# WORKDIR /app
# RUN pip install --no-cache-dir numpy==1.24.0
# COPY common/model_loader.py ./model_loader.py
# COPY explanations/evaluate.py ./evaluate.py
# ENTRYPOINT ["python", "evaluate.py"]
