# Build from repo root:
#   docker build -t ghcr.io/landseer-project/dataset_celeba:v1 \
#     -f containers/datasets/celeba/Dockerfile containers/datasets/celeba
FROM python:3.11-slim

LABEL org.opencontainers.image.title="CelebA Dataset Prep"
LABEL org.opencontainers.image.description="Converts a local CelebA download tree into Landseer .npy artifacts"
LABEL org.landseer.type="dataset"
LABEL org.landseer.category="datasets"
LABEL org.landseer.dataset="celeba"

WORKDIR /app

RUN pip install --no-cache-dir numpy pandas scikit-learn pillow

COPY main.py /app/main.py

ENTRYPOINT ["python", "/app/main.py"]
