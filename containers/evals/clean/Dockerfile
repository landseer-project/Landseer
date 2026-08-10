# Build with context directory containers/evals:
#   docker build -f clean/Dockerfile .
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.title="Clean Evaluator"
LABEL org.opencontainers.image.description="Evaluates clean top-1 train and test accuracy"
LABEL org.landseer.type="evaluator"
LABEL org.landseer.category="evals"
LABEL org.landseer.metrics="clean_accuracy,clean_train_accuracy,poisoned_train_accuracy,poisoned_test_accuracy,mia_auc"
LABEL org.landseer.required_artifacts=""
LABEL org.landseer.defense_types=""

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy==1.24.0 \
    scikit-learn==1.3.0

COPY common/model_loader.py ./model_loader.py
COPY clean/evaluate.py ./evaluate.py

ENTRYPOINT ["python", "evaluate.py"]
