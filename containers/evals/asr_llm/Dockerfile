FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

LABEL org.opencontainers.image.title="DOOR Prefill ASR Evaluator"
LABEL org.opencontainers.image.description="Evaluates LLM safety under harmful prefilling"
LABEL org.landseer.type="evaluator"
LABEL org.landseer.category="evals"
LABEL org.landseer.metrics="llm_asr"
LABEL org.landseer.required_artifacts=""
LABEL org.landseer.defense_types="door"

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY evaluate.py ./evaluate.py
COPY utils ./utils

CMD ["python", "evaluate.py"]