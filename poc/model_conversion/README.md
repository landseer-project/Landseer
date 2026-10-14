# in-noop CIFAR-10 checkpoint

Proof-of-concept run of `ghcr.io/landseer-project/in_noop:v7` (`python main.py`).

Every training run in this POC uses **200 epochs**. Do not shorten the run.

- Dataset: `/data/landseer/landseer_old_data/data/cifar10/clean` (`data.npy`, `labels.npy`, `test_data.npy`, `test_labels.npy`). Shape `(50000, 3, 32, 32)` / `(10000, 3, 32, 32)`, float32 in `[0, 1]`, labels 0–9.
- Epochs: **200**. Train accuracy 99.68%, test accuracy 0.9237.
- Checkpoint: `model.pt` in this directory. ResNet-20 state dict with a 10-way linear layer.

`configs/model/config_model_resnet20.py` is copied here unchanged as `config_model_resnet20.py`. That file’s `config()` builds ResNet-20 with the default `num_classes=2`, so `config_model.py` builds the same network with `num_classes=10`.

```bash
docker run --rm --name landseer-poc-in-noop-cifar10 \
  --gpus '"device=0"' \
  --shm-size=8g \
  -v /data/landseer/landseer_old_data/data/cifar10/clean:/data:ro \
  -v /share/landseer/workspace-ayushi/Landseer/poc/model_conversion:/opt/model:ro \
  -v /share/landseer/workspace-ayushi/Landseer/poc/model_conversion/train_output:/output \
  -e PYTHONPATH=/opt/model \
  ghcr.io/landseer-project/in_noop:v7 \
  python main.py --output /output --input_dir /data --epochs 200
```

`--shm-size=8g` is required. The image DataLoader uses `num_workers=2`, and the default Docker `/dev/shm` raises a bus error. The image also writes the input arrays into `/output`; those copies were removed, and `model.pt` was copied to this directory. The log is `train.log`.

## PyTorch to TensorFlow

`ghcr.io/landseer-project/model_converter_pytorch_to_other:v1` converts `model.pt` to a TensorFlow SavedModel at `tensorflow/model`. onnx-tf writes a SavedModel directory, so there is no `model.h5` file. `conversion_metadata.json` is beside it.

The training forward uses `F.avg_pool2d(out, out.size()[3])`. `torch.onnx.export` rejects a tensor kernel size, so `config_model_export.py` uses adaptive average pooling to 1×1 for export. The weights are unchanged.

Test accuracy on the CIFAR-10 test set, with the in-noop image normalization `(0.4914, 0.4822, 0.4465)` / `(0.2023, 0.1994, 0.2010)`:

- PyTorch: 0.9236
- TensorFlow: 0.9236

The training log reported 0.9237.
