# Build from repo root:
#   docker build -t ghcr.io/landseer-project/dataset_mnist:v1 \
#     -f containers/datasets/mnist/Dockerfile containers/datasets/mnist
FROM python:3.11-slim

LABEL org.opencontainers.image.title="MNIST Dataset Prep"
LABEL org.opencontainers.image.description="Downloads MNIST and writes Landseer .npy artifacts"
LABEL org.landseer.type="dataset"
LABEL org.landseer.category="datasets"
LABEL org.landseer.dataset="mnist"

WORKDIR /app

RUN pip install --no-cache-dir numpy torchvision torch

COPY main.py /app/main.py

ENTRYPOINT ["python", "/app/main.py"]
