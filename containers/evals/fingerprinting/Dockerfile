FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.title="Fingerprinting Evaluator"
LABEL org.opencontainers.image.description="Evaluates model fingerprinting using MINGD attack"
LABEL org.landseer.type="evaluator"
LABEL org.landseer.category="evals"
LABEL org.landseer.metrics="mingd_score,fingerprint_accuracy"
LABEL org.landseer.required_artifacts=""
LABEL org.landseer.defense_types="fingerprinting"

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy==1.24.0 \
    scikit-learn==1.3.0

COPY evaluate.py .

ENTRYPOINT ["python", "evaluate.py"]
