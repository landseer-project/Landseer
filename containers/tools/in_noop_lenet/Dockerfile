# CPU-only during-training baseline (noop) that trains LeNet-5.
# Build from repo root:
#   docker build -t landseer/in_noop_lenet:cpu \
#     -f containers/tools/in_noop_lenet/Dockerfile containers/tools/in_noop_lenet
FROM python:3.11-slim

LABEL org.opencontainers.image.title="in_noop LeNet-5 (CPU)"
LABEL org.opencontainers.image.description="During-training baseline: trains LeNet-5 with no defense, CPU only"
LABEL org.landseer.type="tool"
LABEL org.landseer.category="tools"
LABEL org.opencontainers.image.stage="during_training"
LABEL org.landseer.defense_type="none"
LABEL org.landseer.dataset="mnist"

# so per-epoch logs appear while training instead of buffering
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# CPU-only torch wheels keep the image small and multi-arch (no CUDA).
RUN pip install --no-cache-dir numpy \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY main.py /app/main.py

ENTRYPOINT ["python", "/app/main.py"]
