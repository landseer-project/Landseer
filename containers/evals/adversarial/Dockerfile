FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.title="Adversarial Evaluator"
LABEL org.opencontainers.image.description="Evaluates adversarial robustness using PGD, FGSM, and Carlini L2 attacks"
LABEL org.landseer.type="evaluator"
LABEL org.landseer.category="evals"
LABEL org.landseer.metrics="clean_accuracy,pgd_accuracy,fgsm_accuracy,carlini_l2_accuracy"
LABEL org.landseer.required_artifacts=""
LABEL org.landseer.defense_types="adversarial"

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    torchattacks==3.5.1 \
    numpy==1.24.0 \
    scikit-learn==1.3.0

COPY evaluate.py .

ENTRYPOINT ["python", "evaluate.py"]
