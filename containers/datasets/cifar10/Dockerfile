# Build from repo root:
#   docker build -t ghcr.io/landseer-project/dataset_cifar10:v1 \
#     -f containers/datasets/cifar10/Dockerfile containers/datasets/cifar10
FROM python:3.11-slim

LABEL org.opencontainers.image.title="CIFAR-10 Dataset Prep"
LABEL org.opencontainers.image.description="Downloads CIFAR-10 and writes Landseer .npy artifacts"
LABEL org.landseer.type="dataset"
LABEL org.landseer.category="datasets"
LABEL org.landseer.dataset="cifar10"

WORKDIR /app

RUN pip install --no-cache-dir numpy torchvision torch

COPY main.py /app/main.py

ENTRYPOINT ["python", "/app/main.py"]
