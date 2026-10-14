# Landseer in-training tool: Free Adversarial Training (TensorFlow).
# Build with context directory containers/tools/in_freeat:
#   docker build -t <image> containers/tools/in_freeat
# Runs between the pytorch_to_tensorflow input and output adapters
# (configs/tools.yaml); see docs/how-to/adapter-layers.md.
FROM tensorflow/tensorflow:2.15.0-gpu

LABEL org.opencontainers.image.title="Free Adversarial Training" \
      org.opencontainers.image.description="Adversarial training for free (Shafahi et al. 2019) in TensorFlow on the pipeline's model" \
      org.opencontainers.image.vendor="Landseer Project" \
      org.opencontainers.image.source="https://github.com/landseer-project/landseer" \
      org.opencontainers.image.base.name="tensorflow/tensorflow:2.15.0-gpu" \
      org.landseer.schema="1" \
      org.landseer.type="tool" \
      org.landseer.name="in_freeat" \
      org.landseer.datasets="cifar10" \
      org.landseer.modality="image" \
      org.landseer.framework="tensorflow==2.15.0" \
      org.landseer.python="3.11" \
      org.landseer.accelerator="cuda" \
      org.landseer.cuda="12.3" \
      org.landseer.stage="during_training" \
      org.landseer.defense_type="adversarial"

ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app
COPY main.py ./main.py

CMD ["python", "main.py"]
