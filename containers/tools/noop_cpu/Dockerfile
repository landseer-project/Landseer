# Baseline (noop) tool for the pre / post / deployment stages, CPU only.
# One image serves all three stages because stage_validation reads
# defense_stage out of configs/tools.yaml and only falls back to image
# labels when that field is missing.
#
# Build from repo root:
#   docker build -t landseer/noop_cpu:cpu \
#     -f containers/tools/noop_cpu/Dockerfile containers/tools/noop_cpu
FROM python:3.11-slim

LABEL org.opencontainers.image.title="noop passthrough (CPU)"
LABEL org.opencontainers.image.description="Baseline stage that copies input artifacts to output unchanged"
LABEL org.landseer.type="tool"
LABEL org.landseer.defense_type="none"

# so logs show up while it runs instead of sitting in a buffer
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY main.py /app/main.py
ENTRYPOINT ["python", "/app/main.py"]
