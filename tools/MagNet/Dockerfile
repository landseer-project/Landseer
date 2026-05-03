# NVIDIA CUDA base — matches existing Landseer image stack
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

LABEL org.opencontainers.image.dataset="CIFAR-10, CelebA"
LABEL org.opencontainers.image.defense_stage="post_training"
LABEL org.opencontainers.image.defense_type="adversarial training"
LABEL org.opencontainers.image.framework="pytorch"

WORKDIR /app

ARG UPSTREAM_REPO=https://github.com/Trevillie/MagNet.git
ARG UPSTREAM_SHA=b115cf8798dc0c304ca0da34b2bdf1485da6f54f

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip git ca-certificates python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/local/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip3

RUN git clone "${UPSTREAM_REPO}" /app \
    && cd /app \
    && git checkout "${UPSTREAM_SHA}"

COPY patches/ /tmp/patches/
RUN cd /app \
    && git config user.email "artifact@local" \
    && git config user.name "artifact" \
    && git am $(ls -1 /tmp/patches/*.patch | sort) \
    && rm -rf /tmp/patches

# Ubuntu 20.04 system Python 3.8: PyTorch index may serve typing-extensions wheels
# that need a 3.8–compatible version.
RUN printf '%s\n' "typing-extensions==4.8.0" > /tmp/pip-constraints.txt && \
    pip3 install --no-cache-dir \
    -c /tmp/pip-constraints.txt \
    "torch==1.13.1+cu117" "torchvision==0.14.1+cu117" \
    --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip3 install --no-cache-dir numpy scipy matplotlib h5py "protobuf==3.20.*"

ENV TF_CPP_MIN_LOG_LEVEL=3

CMD ["python3", "main.py"]
