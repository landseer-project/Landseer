FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.title="Explanations Evaluator"
LABEL org.landseer.type="evaluator"
LABEL org.landseer.metrics="drop10_score"

WORKDIR /app
RUN pip install --no-cache-dir numpy==1.24.0
COPY common/model_loader.py ./model_loader.py
COPY explanations/evaluate.py ./evaluate.py
ENTRYPOINT ["python", "evaluate.py"]
