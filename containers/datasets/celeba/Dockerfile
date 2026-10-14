# Build from repo root:
#   docker build -t ghcr.io/landseer-project/dataset_celeba:v1 \
#     -f containers/datasets/celeba/Dockerfile containers/datasets/celeba
FROM python:3.11-slim

LABEL org.opencontainers.image.title="CelebA Dataset Prep" \
      org.opencontainers.image.description="Converts a local CelebA download tree into Landseer .npy artifacts" \
      org.opencontainers.image.vendor="Landseer Project" \
      org.opencontainers.image.source="https://github.com/landseer-project/landseer" \
      org.opencontainers.image.base.name="python:3.11-slim" \
      org.landseer.schema="1" \
      org.landseer.type="dataset" \
      org.landseer.name="celeba" \
      org.landseer.dataset="celeba" \
      org.landseer.variant="clean" \
      org.landseer.datasets="celeba" \
      org.landseer.modality="image" \
      org.landseer.framework="numpy" \
      org.landseer.python="3.11" \
      org.landseer.accelerator="cpu" \
      org.landseer.inputs="/source_artifacts:list_attr_celeba.csv,img_align_celeba" \
      org.landseer.outputs="data.npy,labels.npy,test_data.npy,test_labels.npy,filenames.npy,test_filenames.npy,img_align_celeba" \
      org.landseer.dependencies="numpy,pandas,scikit-learn,pillow"

WORKDIR /app

RUN pip install --no-cache-dir numpy pandas scikit-learn pillow

COPY main.py /app/main.py

ENTRYPOINT ["python", "/app/main.py"]
