# Landseer adapter: PyTorch (standard form) <-> TensorFlow/Keras.
# Build with context directory containers/adapters/pytorch_tensorflow:
#   docker build -t <image> containers/adapters/pytorch_tensorflow
# Used as both the input adapter (--direction input) and the output adapter
# (--direction output) of TensorFlow tools; see docs/how-to/adapter-layers.md.
FROM python:3.10-slim

LABEL org.opencontainers.image.title="PyTorch-TensorFlow Adapter" \
      org.opencontainers.image.description="Converts config_model.py/model.pt to a trainable Keras model and trained Keras models back to model.pt" \
      org.opencontainers.image.vendor="Landseer Project" \
      org.opencontainers.image.source="https://github.com/landseer-project/landseer" \
      org.opencontainers.image.base.name="python:3.10-slim" \
      org.landseer.schema="1" \
      org.landseer.type="adapter" \
      org.landseer.name="pytorch_tensorflow" \
      org.landseer.datasets="*" \
      org.landseer.modality="image" \
      org.landseer.framework="pytorch==2.1.2,tensorflow==2.15.1" \
      org.landseer.python="3.10" \
      org.landseer.accelerator="cpu" \
      org.landseer.source_framework="pytorch,tensorflow" \
      org.landseer.target_framework="tensorflow,pytorch" \
      org.landseer.dependencies="nobuco==0.17.0,numpy==1.26.4"

ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

RUN pip install --no-cache-dir tensorflow-cpu==2.15.1 numpy==1.26.4 && \
    pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir nobuco==0.17.0

WORKDIR /app
COPY main.py ./main.py

CMD ["python", "main.py"]
