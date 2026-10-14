FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.title="LiRA MIA Evaluator" \
      org.opencontainers.image.description="Evaluates membership inference risk using saved shadow-model scores and Online LiRA AUC" \
      org.opencontainers.image.vendor="Landseer Project" \
      org.opencontainers.image.source="https://github.com/landseer-project/landseer" \
      org.opencontainers.image.base.name="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime" \
      org.landseer.schema="1" \
      org.landseer.type="evaluator" \
      org.landseer.name="privacy" \
      org.landseer.datasets="celeba" \
      org.landseer.modality="image" \
      org.landseer.framework="pytorch==2.1.0" \
      org.landseer.python="3.10" \
      org.landseer.accelerator="cuda" \
      org.landseer.cuda="11.8" \
      org.landseer.metrics="mia_auc" \
      org.landseer.required_artifacts="" \
      org.landseer.defense_types="differential_privacy" \
      org.landseer.dependencies="numpy==1.24.0,scipy==1.10.1,scikit-learn==1.3.0"

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    numpy==1.24.0 \
    scipy==1.10.1 \
    scikit-learn==1.3.0

COPY common/model_loader.py ./model_loader.py
COPY privacy/evaluate.py ./evaluate.py
COPY privacy/celeba_shadows ./celeba_shadows

ENTRYPOINT ["python", "evaluate.py"]