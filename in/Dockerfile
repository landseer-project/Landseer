FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
WORKDIR /app
COPY defense.py trades.py /app/
COPY normal.py /app/
RUN pip install torchvision torchattacks
