FROM python:3.10-slim
WORKDIR /app
COPY defense.py /app/
#COPY feature_squeeze_celebA.py /app/
COPY normal.py /app/
RUN pip install numpy torch torchvision pandas scikit-learn
# ENTRYPOINT ["python", "feature_squeeze.py"]S
