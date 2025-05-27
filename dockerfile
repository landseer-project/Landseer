FROM python:3.9-slim


LABEL org.opencontainers.image.dataset="CIFAR-10","CELEBA"
LABEL org.opencontainers.image.defense_stage="pre"
LABEL org.opencontainers.image.defense_type="dp"

WORKDIR /app
COPY defense.py /app/

RUN pip install diffprivlib tensorflow numpy
