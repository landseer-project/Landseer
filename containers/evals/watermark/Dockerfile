FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.title="Watermark Evaluator" \
      org.opencontainers.image.description="Evaluates watermark detection and extraction accuracy" \
      org.opencontainers.image.vendor="Landseer Project" \
      org.opencontainers.image.source="https://github.com/landseer-project/landseer" \
      org.opencontainers.image.base.name="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime" \
      org.landseer.schema="1" \
      org.landseer.type="evaluator" \
      org.landseer.name="watermark" \
      org.landseer.datasets="cifar10,celeba" \
      org.landseer.modality="image" \
      org.landseer.framework="pytorch==2.1.0" \
      org.landseer.python="3.10" \
      org.landseer.accelerator="cuda" \
      org.landseer.cuda="11.8" \
      org.landseer.metrics="watermark_accuracy,bit_accuracy,detection_rate,wmacc_badnets" \
      org.landseer.required_artifacts="" \
      org.landseer.defense_types="watermarking" \
      org.landseer.dependencies="numpy==1.24.0,scikit-learn==1.3.0,torchvision==0.16.0,Pillow==10.0.0"

WORKDIR /app

# Install dependencies including torchvision and Pillow for trigger loading
RUN pip install --no-cache-dir \
    numpy==1.24.0 \
    scikit-learn==1.3.0 \
    torchvision==0.16.0 \
    Pillow==10.0.0

COPY common/model_loader.py ./model_loader.py
COPY watermark/evaluate.py ./evaluate.py

# Include all watermark trigger files in the evaluator image
COPY watermark/watermark_triggers/ ./watermark_triggers/

# Baked-in Uchida weight watermark reference
COPY watermark/watermark_weights/ ./watermark_weights/


ENTRYPOINT ["python", "evaluate.py"]
