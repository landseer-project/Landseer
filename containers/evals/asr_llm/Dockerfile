FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

LABEL org.opencontainers.image.title="DOOR Prefill ASR Evaluator" \
      org.opencontainers.image.description="Evaluates LLM safety under harmful prefilling" \
      org.opencontainers.image.vendor="Landseer Project" \
      org.opencontainers.image.source="https://github.com/landseer-project/landseer" \
      org.opencontainers.image.base.name="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime" \
      org.landseer.schema="1" \
      org.landseer.type="evaluator" \
      org.landseer.name="asr_llm" \
      org.landseer.datasets="llm_data" \
      org.landseer.modality="text" \
      org.landseer.framework="pytorch==2.6.0" \
      org.landseer.python="3.11" \
      org.landseer.accelerator="cuda" \
      org.landseer.cuda="12.4" \
      org.landseer.metrics="llm_asr" \
      org.landseer.required_artifacts="" \
      org.landseer.defense_types="door" \
      org.landseer.dependencies="transformers==4.45.2,accelerate==1.3.0,tokenizers==0.20.3,safetensors==0.5.2,sentencepiece==0.2.0,openai==1.61.1"

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY evaluate.py ./evaluate.py
COPY utils ./utils

CMD ["python", "evaluate.py"]