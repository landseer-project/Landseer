# Build from Landseer repo root:
#
# docker build \
#   -t ghcr.io/landseer-project/dataset_celeba_poisoned:v1 \
#   -f containers/datasets/celeba_poisoned/Dockerfile \
#   .

FROM python:3.11-slim

LABEL org.opencontainers.image.title="CelebA BadNets Dataset Prep" \
      org.opencontainers.image.description="Creates a BadNets-poisoned CelebA Landseer dataset" \
      org.opencontainers.image.vendor="Landseer Project" \
      org.opencontainers.image.source="https://github.com/landseer-project/landseer" \
      org.opencontainers.image.base.name="python:3.11-slim" \
      org.landseer.schema="1" \
      org.landseer.type="dataset" \
      org.landseer.name="celeba_poisoned" \
      org.landseer.dataset="celeba" \
      org.landseer.variant="poisoned" \
      org.landseer.attack="badnets" \
      org.landseer.datasets="celeba" \
      org.landseer.modality="image" \
      org.landseer.framework="numpy" \
      org.landseer.python="3.11" \
      org.landseer.accelerator="cpu" \
      org.landseer.inputs="/source_artifacts:list_attr_celeba.csv,img_align_celeba" \
      org.landseer.outputs="data.npy,labels.npy,test_data.npy,test_labels.npy,filenames.npy,test_filenames.npy,poisoning_metadata.json" \
      org.landseer.dependencies="numpy,pandas,scikit-learn,pillow"

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