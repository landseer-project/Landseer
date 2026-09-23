# Differential privacy (DP-SGD) during-training defense, CPU only.
#
# The ghcr in_dp image is cifar10-specific (3-channel Normalize) and is a cuda
# build, so it neither runs on mnist nor natively on arm. This one works on any
# input shape and any architecture opacus accepts.
#
# Build from repo root:
#   docker build -t landseer/in_dp_lenet:cpu \
#     -f containers/tools/in_dp_lenet/Dockerfile containers/tools/in_dp_lenet
FROM python:3.11-slim

LABEL org.opencontainers.image.title="DP-SGD defense (CPU)"
LABEL org.opencontainers.image.description="Differential privacy via Opacus DP-SGD, during-training stage"
LABEL org.opencontainers.image.defense_stage="during_training"
LABEL org.opencontainers.image.defense_type="differential_privacy"
LABEL org.opencontainers.image.dataset="MNIST"
LABEL org.opencontainers.image.framework="pytorch"
LABEL org.landseer.type="tool"

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir numpy opacus

COPY main.py /app/main.py

ENTRYPOINT ["python", "/app/main.py"]
