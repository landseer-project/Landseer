FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.title="Fingerprinting Evaluator" \
      org.opencontainers.image.description="Evaluates model fingerprinting using MINGD attack" \
      org.opencontainers.image.vendor="Landseer Project" \
      org.opencontainers.image.source="https://github.com/landseer-project/landseer" \
      org.opencontainers.image.base.name="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime" \
      org.landseer.schema="1" \
      org.landseer.type="evaluator" \
      org.landseer.name="fingerprinting" \
      org.landseer.datasets="cifar10,celeba" \
      org.landseer.modality="image" \
      org.landseer.framework="pytorch==2.1.0" \
      org.landseer.python="3.10" \
      org.landseer.accelerator="cuda" \
      org.landseer.cuda="11.8" \
      org.landseer.metrics="mingd_score,fingerprint_accuracy" \
      org.landseer.required_artifacts="" \
      org.landseer.defense_types="fingerprinting" \
      org.landseer.dependencies="numpy==1.24.0,scikit-learn==1.3.0"

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy==1.24.0 \
    scikit-learn==1.3.0

COPY common/model_loader.py ./model_loader.py
COPY fingerprinting/evaluate.py ./evaluate.py

ENTRYPOINT ["python", "evaluate.py"]
