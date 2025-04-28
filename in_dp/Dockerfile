FROM python:3.10-slim
WORKDIR /app
COPY . /app/
COPY defense.py /app/
# COPY privacy_tracker.py /app/
# COPY normal.py /app/
RUN pip install torch torchvision opacus pandas numpy tqdm scikit-learn
#CMD ["python", "defense.py", "--img_dir", "/app/data/img_align_celeba", "--csv_path", "/app/data/list_attr_celeba.csv", "--enable_dp"]