FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.title="Fairness Evaluator"
LABEL org.opencontainers.image.description="Evaluates model fairness using demographic parity and equalized odds"
LABEL org.landseer.type="evaluator"
LABEL org.landseer.category="evals"
LABEL org.landseer.metrics="demographic_parity,equalized_odds_diff"
LABEL org.landseer.required_artifacts="sensitive_attributes.npy"
LABEL org.landseer.defense_types="fairness"

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fairlearn==0.9.0 \
    numpy==1.24.0 \
    scikit-learn==1.3.0

COPY evaluate.py .

ENTRYPOINT ["python", "evaluate.py"]
