FROM python:3.10-slim
WORKDIR /app
COPY defense.py /app/
COPY normal.py /app/
RUN pip install numpy torch torchvision
# ENTRYPOINT ["python", "feature_squeeze.py"]
