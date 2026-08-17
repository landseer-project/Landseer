FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.title="LiRA MIA Evaluator"
LABEL org.opencontainers.image.description="Evaluates membership inference risk using saved shadow-model scores and Online LiRA AUC"
LABEL org.landseer.type="evaluator"
LABEL org.landseer.category="evals"
LABEL org.landseer.metrics="mia_auc"
LABEL org.landseer.required_artifacts=""
LABEL org.landseer.defense_types="differential_privacy"

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    numpy==1.24.0 \
    scipy==1.10.1 \
    scikit-learn==1.3.0

COPY common/model_loader.py ./model_loader.py
COPY privacy/evaluate.py ./evaluate.py

ENTRYPOINT ["python", "evaluate.py"]