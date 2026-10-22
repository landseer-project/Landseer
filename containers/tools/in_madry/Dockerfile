# Landseer in-training tool: PGD adversarial training (Madry et al. 2018) in TensorFlow.
# Build with context directory containers/tools/in_madry:
#   docker build -t <image> containers/tools/in_madry
# Its model_* labels make Landseer run it between the PyTorch<->TensorFlow
# adapters (configs/tools.yaml); see docs/how-to/adapter-layers.md.
FROM tensorflow/tensorflow:2.15.0-gpu

LABEL org.opencontainers.image.title="PGD Adversarial Training" \
      org.opencontainers.image.description="PGD adversarial training (Madry et al. 2018) in TensorFlow on the pipeline's model" \
      org.opencontainers.image.vendor="Landseer Project" \
      org.opencontainers.image.source="https://github.com/landseer-project/landseer" \
      org.opencontainers.image.base.name="tensorflow/tensorflow:2.15.0-gpu" \
      org.opencontainers.image.licenses="MIT" \
      org.landseer.schema="1" \
      org.landseer.type="tool" \
      org.landseer.name="in_madry" \
      org.landseer.datasets="cifar10,celeba" \
      org.landseer.modality="image" \
      org.landseer.framework="tensorflow==2.15.0" \
      org.landseer.python="3.11" \
      org.landseer.accelerator="cuda" \
      org.landseer.cuda="12.3" \
      org.landseer.stage="during_training" \
      org.landseer.defense_type="adversarial" \
      org.landseer.model_input="architecture" \
      org.landseer.model_output="model" \
      org.landseer.model_path="model_tf/model.keras"

# The base image lists the CUDA stub libcuda.so.1 in the linker cache ahead of the
# driver's, so TensorFlow loads the stub (cuInit error 34) and silently runs on the CPU.
RUN rm /etc/ld.so.conf.d/z-cuda-stubs.conf && ldconfig

# Allocate GPU memory as needed: by default TensorFlow reserves almost all of it,
# starving other tasks that share the GPU.
ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_FORCE_GPU_ALLOW_GROWTH=true

WORKDIR /app
COPY main.py ./main.py

CMD ["python", "main.py"]
