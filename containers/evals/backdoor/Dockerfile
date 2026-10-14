FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.title="Backdoor Evaluator" \
      org.opencontainers.image.description="Evaluates backdoor attack success rate and clean accuracy" \
      org.opencontainers.image.vendor="Landseer Project" \
      org.opencontainers.image.source="https://github.com/landseer-project/landseer" \
      org.opencontainers.image.base.name="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime" \
      org.landseer.schema="1" \
      org.landseer.type="evaluator" \
      org.landseer.name="backdoor" \
      org.landseer.datasets="cifar10,celeba" \
      org.landseer.modality="image" \
      org.landseer.framework="pytorch==2.1.0" \
      org.landseer.python="3.10" \
      org.landseer.accelerator="cuda" \
      org.landseer.cuda="11.8" \
      org.landseer.metrics="attack_success_rate,clean_accuracy_post_attack,backdoor_robustness" \
      org.landseer.required_artifacts="poisoning_metadata.json" \
      org.landseer.defense_types="backdoor" \
      org.landseer.dependencies="numpy==1.24.0,scikit-learn==1.3.0"

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy==1.24.0 \
    scikit-learn==1.3.0

COPY common/model_loader.py ./model_loader.py
COPY backdoor/evaluate.py ./evaluate.py

ENTRYPOINT ["python", "evaluate.py"]
