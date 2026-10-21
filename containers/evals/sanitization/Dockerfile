FROM python:3.10-slim

LABEL org.opencontainers.image.title="Sanitization Evaluator" \
      org.opencontainers.image.description="ROC-AUC and Precision@n of pre-training detector scores on the training set the model was trained on" \
      org.opencontainers.image.vendor="Landseer Project" \
      org.opencontainers.image.source="https://github.com/landseer-project/landseer" \
      org.opencontainers.image.base.name="python:3.10-slim" \
      org.landseer.schema="1" \
      org.landseer.type="evaluator" \
      org.landseer.name="sanitization" \
      org.landseer.datasets="cifar10,celeba" \
      org.landseer.modality="image" \
      org.landseer.framework="numpy==2.2.6" \
      org.landseer.python="3.10" \
      org.landseer.accelerator="cpu" \
      org.landseer.metrics="xgbod_train_roc_auc,xgbod_train_precision_at_n" \
      org.landseer.required_artifacts="" \
      org.landseer.defense_types="outlier_removal" \
      org.landseer.dependencies="numpy==2.2.6"

WORKDIR /app

RUN pip install --no-cache-dir numpy==2.2.6

COPY sanitization/evaluate.py ./evaluate.py

ENTRYPOINT ["python", "evaluate.py"]
