#Docker for In DP stage
FROM python:3.10-slim

LABEL org.opencontainers.image.dataset="CIFAR-10","CELEBA"
LABEL org.opencontainers.image.defense_stage="in"
LABEL org.opencontainers.image.defense_type="dp"

WORKDIR /app
COPY defense.py /app/
RUN pip install torch torchvision opacus pandas numpy tqdm scikit-learn


#Dockerfile for DP Pre stage
# FROM python:3.9-slim

# LABEL org.opencontainers.image.dataset="CIFAR-10","CELEBA"
# LABEL org.opencontainers.image.defense_stage="pre"
# LABEL org.opencontainers.image.defense_type="dp"

# WORKDIR /app
# COPY defense.py /app/

# RUN pip install diffprivlib tensorflow numpy
