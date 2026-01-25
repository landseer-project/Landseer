FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.title="Watermark Evaluator"
LABEL org.opencontainers.image.description="Evaluates watermark detection and extraction accuracy"
LABEL org.landseer.type="evaluator"
LABEL org.landseer.category="evals"
LABEL org.landseer.metrics="watermark_accuracy,bit_accuracy,detection_rate"
LABEL org.landseer.required_artifacts="watermark_key.json"
LABEL org.landseer.defense_types="watermarking"

WORKDIR /app

# Install dependencies including torchvision and Pillow for trigger loading
RUN pip install --no-cache-dir \
    numpy==1.24.0 \
    scikit-learn==1.3.0 \
    torchvision==0.16.0 \
    Pillow==10.0.0

COPY evaluate.py .

ENTRYPOINT ["python", "evaluate.py"]
