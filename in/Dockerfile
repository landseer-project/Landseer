FROM python:3.10-slim
WORKDIR /app
COPY . /app/
COPY defense.py /app/
RUN pip install torch torchvision opacus pandas numpy tqdm scikit-learn