FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.title="OOD Evaluator"
LABEL org.opencontainers.image.description="Evaluates out-of-distribution detection using AUC and FPR metrics"
LABEL org.landseer.type="evaluator"
LABEL org.landseer.category="evals"
LABEL org.landseer.metrics="ood_auc,fpr_at_95_tpr"
LABEL org.landseer.required_artifacts=""
LABEL org.landseer.defense_types="outlier_removal"

WORKDIR /app

# Install dependencies including torchvision for CIFAR-100 loading
RUN pip install --no-cache-dir \
    numpy==1.24.0 \
    scikit-learn==1.3.0 \
    torchvision==0.16.0

COPY evaluate.py .

ENTRYPOINT ["python", "evaluate.py"]
