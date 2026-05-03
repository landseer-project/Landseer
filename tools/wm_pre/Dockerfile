# Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

LABEL org.opencontainers.image.dataset="CIFAR-10, CelebA"
LABEL org.opencontainers.image.defense_stage="pre_training"
LABEL org.opencontainers.image.defense_type="watermarking"
LABEL org.opencontainers.image.framework="pytorch"


WORKDIR /app

# copy the preprocessing script and the default trigger
COPY main.py /app/main.py
COPY trigger_white.png /app/trigger_white.png


# only need lightweight deps (no torch required for this script, but base image already has it)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      pillow \
      numpy

# Default command (Landseer will override this)
CMD ["python", "main.py", "--input-dir", "/data", "--output", "/output"]
