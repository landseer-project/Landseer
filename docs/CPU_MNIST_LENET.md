# Running Landseer on CPU with MNIST and LeNet-5

This adds a CPU-only pipeline so Landseer can be run end to end on a laptop, with
no GPU, no MySQL, no MinIO and no GHCR access. It also adds MNIST support, which
`configs/datasets.yaml` advertised but nothing implemented.

Everything here runs through the normal Landseer path: backend, scheduler and a
worker executing each stage as a container. Nothing bypasses the pipeline.

## Quick start

Docker needs to be running. Then, from the repo root:

```bash
# build the images (once)
docker build -t landseer/dataset_mnist:cpu -f containers/datasets/mnist/Dockerfile      containers/datasets/mnist
docker build -t landseer/in_noop_lenet:cpu -f containers/tools/in_noop_lenet/Dockerfile containers/tools/in_noop_lenet
docker build -t landseer/in_dp_lenet:cpu   -f containers/tools/in_dp_lenet/Dockerfile   containers/tools/in_dp_lenet
docker build -t landseer/noop_cpu:cpu      -f containers/tools/noop_cpu/Dockerfile      containers/tools/noop_cpu
for e in clean adversarial ood fingerprinting privacy explanations cka_llm; do
  tag=$e; [ "$e" = cka_llm ] && tag=cka
  docker build -t landseer/eval_${tag}:cpu -f containers/evals/$e/Dockerfile.cpu containers/evals
done

# run it
./helper_scripts/run_mnist_lenet_landseer.sh
```

Or drive the backend and worker by hand, in two terminals:

```bash
export PYTHONPATH=./src:.
export LANDSEER_DB_TYPE=sqlite LANDSEER_DB_PATH=$PWD/landseer.db
export LANDSEER_USE_MINIO=false LANDSEER_HARDLINK_MACRO=true

# terminal 1
landseer-backend --host 127.0.0.1 --port 8000 \
  --config configs/pipeline/mnist_lenet.yaml --data-dir $PWD/data

# terminal 2 (omitting --gpu is what makes it CPU)
landseer-worker --backend-url http://127.0.0.1:8000 \
  --workspace $PWD/ws --no-minio --no-cache
```

Results land in `results/config_mnist_lenet/pipeline_1/metrics_summary.csv`.

A full run is 29 tasks in about 2m40s on an M-series laptop, with two workflows:
the DP-defended arm and the undefended baseline.

## What each file does

### New dataset container

| File | What it does |
|---|---|
| `containers/datasets/mnist/main.py` | Downloads MNIST via torchvision and writes `data.npy`, `labels.npy`, `test_data.npy`, `test_labels.npy` in Landseer's format — CHW float32 scaled to `[0,1]`, so `(60000,1,28,28)`. Mirrors `containers/datasets/cifar10`. Skips the download if the four files already exist. |
| `containers/datasets/mnist/Dockerfile` | `python:3.11-slim` plus torch/torchvision. Multi-arch, no CUDA. |

`configs/datasets.yaml` previously had `mnist` with `default_image: ""` and a
`# TODO: add dataset_mnist container image when available`. This is that image.

### New tool containers

| File | What it does |
|---|---|
| `containers/tools/in_noop_lenet/main.py` | During-training baseline: loads the `.npy` files, builds the model from `config_model.py`, trains normally (no defense), writes `model.pt`. Passes the dataset artifacts through to `/output` so the next stage can consume the directory directly. |
| `containers/tools/in_dp_lenet/main.py` | Differential privacy via Opacus DP-SGD. Same contract as above, plus it writes `privacy_metrics.txt` (`epsilon=`, `dp_accuracy=`), which is how the `clean` evaluator populates `privacy_epsilon` and `dp_accuracy` in the results CSV. |
| `containers/tools/noop_cpu/main.py` | Passthrough baseline for the pre-training, post-training and deployment stages: copies input artifacts to output unchanged. Needed because the registered `pre_noop` / `post_noop` / `deploy_noop` images are private on GHCR. |
| `containers/tools/*/Dockerfile` | CPU-only builds, `ENV PYTHONUNBUFFERED=1` so logs stream instead of buffering. |

Two things `in_dp_lenet` handles that are worth knowing:

- **In-place activations break Opacus.** It attaches backward hooks for per-sample
  gradients, and an in-place op modifies a view those hooks created
  (`Output 0 of BackwardHookFunction is a view and is being modified inplace`).
  The tool walks the model and sets `inplace=False` before wrapping, rather than
  editing the shared model config.
- **Opacus wraps the module.** `GradSampleModule` prefixes every `state_dict` key
  with `_module.`, which the evaluators' loader can't read. The tool unwraps
  before saving.

### New CPU evaluator builds

| File | What it does |
|---|---|
| `containers/evals/{clean,adversarial,ood,fingerprinting,privacy,explanations,cka_llm}/Dockerfile.cpu` | CPU builds of the existing evaluators. |

**No evaluator code is changed.** These copy the same `evaluate.py` and
`common/model_loader.py` as the default Dockerfiles; the only difference is the
base image — `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` (linux/amd64 only)
becomes `python:3.11-slim` plus CPU torch, which is multi-arch and runs natively
on arm64 instead of under emulation. torch goes in its own layer before the
per-evaluator dependencies so all seven images share it.

### New model config

| File | What it does |
|---|---|
| `configs/model/config_model_lenet.py` | LeNet-5 for 1×28×28 input, 61,706 params. Exposes `config()` per the evaluator contract. |
| `configs/model/config_model_mnist.py` | A larger plain CNN (~1.2M params) for the same input, kept so a bigger model can be compared against LeNet on the same data. Select it with `MODEL_CONFIG=`. |

Every pre-existing config in `configs/model/` starts with `nn.Conv2d(3, ...)` at
32×32, so none of them accept MNIST. `padding=2` on LeNet's first convolution
makes the 28×28 input behave like the 32×32 the original paper assumed.

### New pipeline config

| File | What it does |
|---|---|
| `configs/pipeline/mnist_lenet.yaml` | Binds `dataset: mnist` to `model: config_model_lenet.py`, with `in_dp_lenet` and `in_noop_lenet` both listed under during-training so Landseer generates two workflows — defended and control — and the metrics can be compared side by side. Other stages are noop. |

### Changed config files

| File | Change |
|---|---|
| `configs/datasets.yaml` | `mnist.default_image` and `variants.clean.image` set to the new container. Resolves the `TODO`. |
| `configs/tools.yaml` | Registers `pre_noop_cpu`, `in_noop_lenet`, `in_dp_lenet`, `post_noop_cpu`, `deploy_noop_cpu`. Each sets `defense_stage`, which is what `stage_validation` reads — when present it skips fetching image labels from a registry, so local-only images work. |
| `configs/evaluators.yaml` | Points the seven evaluators at the CPU images. Adds `required_artifacts: [gpt2]` to `fingerprinting_llm` and `reef_cka`, and adds a `reef_cka` entry (it had none, so it fell back to the GHCR default hardcoded in `src/pipeline/config_loader.py`). Both are LLM evaluators; declaring the artifact makes them **skip** cleanly on image pipelines instead of running and failing. Also removes a duplicate `required_artifacts` key on `fingerprinting_llm` — YAML kept the second, silently discarding the first. |

### Test fixes

| File | Change |
|---|---|
| `tests/pipeline/test_stage_validation.py` | Adds `from pathlib import Path`. Four tests annotate `tmp_path: Path` without importing it, raising `NameError` **during collection** — which aborted the entire 1301-test suite, not just this file. |
| `tests/evaluators/conftest.py` | Puts `containers/evals/common` and the evaluator's own directory on `sys.path`. The evaluators do `from model_loader import ...`, which resolves inside the image because `common/` is copied next to `evaluate.py`, but nothing reproduced that locally — so all 38 evaluator tests errored at setup. They now pass. |
| `tests/test_artifact_cache_basic.py` | `pytest.importorskip` guard. It imports `landseer_pipeline`, removed by the restructure. Guarding makes it skip visibly instead of aborting collection; the test body is left intact. |
| `tests/data/test_celeba_compatibility.py` | Same, for `src.data.loaders.celeba`, removed in `84f5297`. |

### .gitignore

The `tools/` rule was unanchored, so besides the root `tools/` submodule
directory it also matched **`containers/tools/`** — meaning any defense tool
container added there was silently invisible to git. Now `/tools/`.

Also ignores local run outputs (`cpu_landseer_run/`, `dp_run*/`, `ws/`), which
otherwise show up as hundreds of megabytes of `.npy` files and downloaded MNIST.

### Helper scripts

| File | What it does |
|---|---|
| `helper_scripts/run_mnist_lenet_landseer.sh` | Starts the backend and one CPU worker, waits for every task, prints the task table and metrics, then shuts both down. Refuses to start if something already holds the port or a worker is already running — otherwise the health check silently passes against a pre-existing backend and the run dir gets wiped underneath it. |
| `helper_scripts/run_mnist_lenet_cpu.sh` | Runs the three container stages directly without the backend or scheduler. Useful for debugging a container in isolation; **not** a Landseer run. |

## Known limitations

- **MNIST is a poor dataset for evaluating defenses.** The undefended baseline's
  train/test gap is about 0.002, so membership inference has almost nothing to
  detect and `mia_auc` sits at chance regardless of the defense. Good for
  validating the pipeline; not for producing privacy results.
- **The adversarial evaluator hardcodes `eps=8/255`**, the CIFAR-10 convention.
  MNIST's is roughly `0.3`, so the attack is far too weak here and the robustness
  numbers should only be compared between arms, not read as absolute.
- **The `privacy` evaluator fails on grayscale.** It hardcodes `IMAGE_SIZE = 32`
  and expands 1 channel to 3, then feeds that to a 1-channel model. Its task is
  still recorded `COMPLETED` because the container exits 0, so
  `combination_success` reads `failure` while every task shows as completed.
- **No random seed is set anywhere**, so metrics move between identical runs.
  `drop10_score` has been observed anywhere from 0.20 to 0.45. Differences
  smaller than that are not meaningful.
- **The images built here are for the host architecture.** Building on an arm64
  Mac produces arm64 images, which will not run on x86 nodes. Use
  `docker buildx --platform linux/amd64,linux/arm64` before pushing anywhere shared.
