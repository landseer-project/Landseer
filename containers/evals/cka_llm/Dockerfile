FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.title="REEF CKA Evaluator" \
      org.opencontainers.image.description="Computes REEF CKA similarity from saved model activations" \
      org.opencontainers.image.vendor="Landseer Project" \
      org.opencontainers.image.source="https://github.com/landseer-project/landseer" \
      org.opencontainers.image.base.name="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime" \
      org.landseer.schema="1" \
      org.landseer.type="evaluator" \
      org.landseer.name="cka_llm" \
      org.landseer.datasets="llm_data" \
      org.landseer.modality="text" \
      org.landseer.framework="pytorch==2.1.0" \
      org.landseer.python="3.10" \
      org.landseer.accelerator="cuda" \
      org.landseer.cuda="11.8" \
      org.landseer.metrics="cka_sim" \
      org.landseer.required_artifacts="" \
      org.landseer.defense_types="reef" \
      org.landseer.dependencies="numpy>=1.24"

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY evaluate.py ./evaluate.py

ENTRYPOINT ["python", "evaluate.py"]