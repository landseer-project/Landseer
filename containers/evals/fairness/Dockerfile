FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.title="Fairness Evaluator" \
      org.opencontainers.image.description="Evaluates model fairness using demographic parity and equalized odds" \
      org.opencontainers.image.vendor="Landseer Project" \
      org.opencontainers.image.source="https://github.com/landseer-project/landseer" \
      org.opencontainers.image.base.name="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime" \
      org.landseer.schema="1" \
      org.landseer.type="evaluator" \
      org.landseer.name="fairness" \
      org.landseer.datasets="celeba" \
      org.landseer.modality="image" \
      org.landseer.framework="pytorch==2.1.0" \
      org.landseer.python="3.10" \
      org.landseer.accelerator="cuda" \
      org.landseer.cuda="11.8" \
      org.landseer.metrics="demographic_parity,equalized_odds_diff" \
      org.landseer.required_artifacts="test_sensitive.npy" \
      org.landseer.defense_types="fairness" \
      org.landseer.dependencies="fairlearn==0.9.0,numpy==1.24.0,scikit-learn==1.3.0"

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fairlearn==0.9.0 \
    numpy==1.24.0 \
    scikit-learn==1.3.0

COPY common/model_loader.py ./model_loader.py
COPY fairness/evaluate.py ./evaluate.py

ENTRYPOINT ["python", "evaluate.py"]
