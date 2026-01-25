FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.title="Backdoor Evaluator"
LABEL org.opencontainers.image.description="Evaluates backdoor attack success rate and clean accuracy"
LABEL org.landseer.type="evaluator"
LABEL org.landseer.category="evals"
LABEL org.landseer.metrics="attack_success_rate,clean_accuracy_post_attack,backdoor_robustness"
LABEL org.landseer.required_artifacts="poisoning_metadata.json"
LABEL org.landseer.defense_types="backdoor"

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy==1.24.0 \
    scikit-learn==1.3.0

COPY evaluate.py .

ENTRYPOINT ["python", "evaluate.py"]
