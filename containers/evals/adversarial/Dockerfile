FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.title="Adversarial Evaluator" \
      org.opencontainers.image.description="Evaluates adversarial robustness using PGD, FGSM, and Carlini L2 attacks" \
      org.opencontainers.image.vendor="Landseer Project" \
      org.opencontainers.image.source="https://github.com/landseer-project/landseer" \
      org.opencontainers.image.base.name="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime" \
      org.landseer.schema="1" \
      org.landseer.type="evaluator" \
      org.landseer.name="adversarial" \
      org.landseer.datasets="cifar10,celeba" \
      org.landseer.modality="image" \
      org.landseer.framework="pytorch==2.1.0" \
      org.landseer.python="3.10" \
      org.landseer.accelerator="cuda" \
      org.landseer.cuda="11.8" \
      org.landseer.metrics="clean_accuracy,pgd_accuracy,fgsm_accuracy,carlini_l2_accuracy" \
      org.landseer.required_artifacts="" \
      org.landseer.defense_types="adversarial" \
      org.landseer.dependencies="torchattacks==3.5.1,numpy==1.24.0,scikit-learn==1.3.0"

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    torchattacks==3.5.1 \
    numpy==1.24.0 \
    scikit-learn==1.3.0

COPY common/model_loader.py ./model_loader.py
COPY adversarial/evaluate.py ./evaluate.py
COPY adversarial/dds_runtime ./dds_runtime

ENTRYPOINT ["python", "evaluate.py"]
