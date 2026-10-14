FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.title="OOD Evaluator" \
      org.opencontainers.image.description="Evaluates out-of-distribution detection using AUC and FPR metrics" \
      org.opencontainers.image.vendor="Landseer Project" \
      org.opencontainers.image.source="https://github.com/landseer-project/landseer" \
      org.opencontainers.image.base.name="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime" \
      org.landseer.schema="1" \
      org.landseer.type="evaluator" \
      org.landseer.name="ood" \
      org.landseer.datasets="cifar10,celeba" \
      org.landseer.modality="image" \
      org.landseer.framework="pytorch==2.1.0" \
      org.landseer.python="3.10" \
      org.landseer.accelerator="cuda" \
      org.landseer.cuda="11.8" \
      org.landseer.network="required" \
      org.landseer.metrics="ood_auc,fpr_at_95_tpr" \
      org.landseer.required_artifacts="" \
      org.landseer.defense_types="outlier_removal" \
      org.landseer.dependencies="numpy==1.24.0,scikit-learn==1.3.0,torchvision==0.16.0"

WORKDIR /app

# Install dependencies including torchvision for CIFAR-100 loading
RUN pip install --no-cache-dir \
    numpy==1.24.0 \
    scikit-learn==1.3.0 \
    torchvision==0.16.0

COPY common/model_loader.py ./model_loader.py
COPY ood/evaluate.py ./evaluate.py

ENTRYPOINT ["python", "evaluate.py"]
