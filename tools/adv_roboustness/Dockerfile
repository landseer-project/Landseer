FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

LABEL org.opencontainers.image.dataset="CIFAR-10, CelebA"
LABEL org.opencontainers.image.defense_stage="during_training"
LABEL org.opencontainers.image.defense_type="adversarial_training"
LABEL org.opencontainers.image.framework="pytorch"


WORKDIR /app
COPY main.py /app/
COPY trades.py /app/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      torchvision \
      torchattacks \
      pillow \
      numpy
