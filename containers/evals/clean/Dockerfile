FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.title="Clean Evaluator"
LABEL org.opencontainers.image.description="Evaluates clean top-1 test accuracy"
LABEL org.landseer.type="evaluator"
LABEL org.landseer.category="evals"
LABEL org.landseer.metrics="clean_accuracy"
LABEL org.landseer.required_artifacts=""
LABEL org.landseer.defense_types=""

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy==1.24.0

COPY evaluate.py .

ENTRYPOINT ["python", "evaluate.py"]
