# Build from repo root:
#   docker build -t ghcr.io/landseer-project/dataset_cifar10_poisoned:v1 \
#     -f containers/datasets/cifar10_poisoned/Dockerfile .

FROM python:3.11-slim

LABEL org.opencontainers.image.title="CIFAR-10 BadNets Poisoned Dataset Prep" \
      org.opencontainers.image.description="Downloads CIFAR-10, applies BadNets poisoning, and writes Landseer .npy artifacts" \
      org.opencontainers.image.vendor="Landseer Project" \
      org.opencontainers.image.source="https://github.com/landseer-project/landseer" \
      org.opencontainers.image.base.name="python:3.11-slim" \
      org.landseer.schema="1" \
      org.landseer.type="dataset" \
      org.landseer.name="cifar10_poisoned" \
      org.landseer.dataset="cifar10" \
      org.landseer.variant="poisoned" \
      org.landseer.datasets="cifar10" \
      org.landseer.modality="image" \
      org.landseer.framework="torchvision" \
      org.landseer.python="3.11" \
      org.landseer.accelerator="cpu" \
      org.landseer.network="required" \
      org.landseer.outputs="data.npy,labels.npy,test_data.npy,test_labels.npy,poisoning_metadata.json" \
      org.landseer.dependencies="numpy,torch,torchvision"

WORKDIR /app

RUN pip install --no-cache-dir numpy torchvision torch

COPY containers/datasets/cifar10_poisoned/main.py /app/main.py
COPY src/data/poisoning /app/data/poisoning

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "/app/main.py"]