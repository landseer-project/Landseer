# Build from Landseer repo root:
#
# docker build \
#   -t ghcr.io/landseer-project/dataset_celeba_poisoned:v1 \
#   -f containers/datasets/celeba_poisoned/Dockerfile \
#   .

FROM python:3.11-slim

LABEL org.opencontainers.image.title="CelebA BadNets Dataset Prep"
LABEL org.opencontainers.image.description="Creates a BadNets-poisoned CelebA Landseer dataset"
LABEL org.landseer.type="dataset"
LABEL org.landseer.category="datasets"
LABEL org.landseer.dataset="celeba"
LABEL org.landseer.variant="badnets"

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    pillow

# Poisoned CelebA preparation script
COPY containers/datasets/celeba_poisoned/main.py /app/main.py

# Reuse the existing Landseer poisoning implementation.
# This includes badnets.py plus its base.py/registry.py dependencies.
COPY src/data/poisoning /app/data/poisoning

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "/app/main.py"]