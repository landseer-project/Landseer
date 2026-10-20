# Adapter Layers (tools in other frameworks)

Landseer's standard intermediate form is PyTorch: the model architecture is
`config_model.py` (`config()` returns an `nn.Module`), trained weights are a
`state_dict` saved as `model.pt`, and datasets are `data.npy`, `labels.npy`,
`test_data.npy`, `test_labels.npy` (images NCHW, float32 in `[0, 1]`). Every
evaluator and most tools read exactly that.

A tool written in another framework (for example TensorFlow) does not have to
be ported. It declares **adapters**: optional containers that run in the same
task as the tool,

- an **input adapter** before the tool, converting standard inputs into what
  the tool needs (e.g. `config_model.py` -> a trainable Keras model), and
- an **output adapter** after the tool, converting its outputs back into the
  standard form (e.g. a trained Keras model -> `model.pt`).

The first tool using this is `in_freeat` (Free Adversarial Training in
TensorFlow).

## Declaring adapters

Adapters are defined once under a top-level `adapters:` key in
`configs/tools.yaml`. A tool either gets them
[automatically from its image labels](#automatic-selection-from-image-labels)
or names them explicitly by key, which overrides the automatic choice:

```yaml
adapters:
  config_model_to_tensorflow:
    name: config-model-to-tensorflow
    source_framework: pytorch
    target_framework: tensorflow
    container:
      image: ghcr.io/landseer-project/adapter_pytorch_tensorflow:v1
      command: python main.py --direction input --weights none
      runtime: null
    # env: {KEY: value}    # optional, added to the adapter container only

  tensorflow_to_pytorch:
    name: tensorflow-to-pytorch
    source_framework: tensorflow
    target_framework: pytorch
    container:
      image: ghcr.io/landseer-project/adapter_pytorch_tensorflow:v1
      command: python main.py --direction output
      runtime: null

tools:
  in_freeat:
    name: in-freeat
    defense_stage: during_training
    framework: tensorflow
    adapters:                             # explicit; omit to select from the image labels
      input: config_model_to_tensorflow   # optional
      output: tensorflow_to_pytorch       # optional
    container:
      image: ghcr.io/landseer-project/in_freeat:v1
      command: python main.py
```

Either position may be omitted. Unknown adapter names or positions fail when
`tools.yaml` is loaded. Pipeline configs need nothing extra: list the tool as
usual, and stage-level `command`/`args` overrides apply to the tool only
(`configs/pipeline/freeat_cifar10.yaml`).

## Automatic selection from image labels

A tool image says which framework it uses and which model it reads and writes
([image labels](../reference/image-labels.md)):

| Label | Values | Meaning |
|---|---|---|
| `org.landseer.framework` | `tensorflow==2.15.0` | The tool's framework (first entry, version ignored) |
| `org.landseer.model_input` | `none` / `architecture` / `weights` | No model, the untrained `config_model.py` network, or the upstream `model.pt` |
| `org.landseer.model_output` | `none` / `model` | Whether the tool writes a trained model |
| `org.landseer.model_path` | `model_tf/model.keras` | Where the tool reads (`/input/...`) and saves (`/output/...`) its model; default for TensorFlow |

When a tool's `tools.yaml` entry names no adapters, `TaskFactory.create_task`
reads these labels (`docker image inspect`, else the registry, cached per
image) and picks adapters from the `adapters:` section by
`source_framework`/`target_framework`:

- an **input adapter** `pytorch -> <framework>` unless `model_input` is `none`;
- an **output adapter** `<framework> -> pytorch` when `model_output` is `model`.

So the stages come out as:

| Tool | model_input | model_output | Adapters |
|---|---|---|---|
| in-training, trains from scratch (`in_freeat`) | `architecture` | `model` | input + output |
| post-training, changes a trained model | `weights` | `model` | input + output |
| deployment, explains or measures a model | `weights` | `none` | input only; downstream keeps the upstream `model.pt` |
| reads data only, writes a model | `none` | `model` | output only |

The chosen adapters get `LANDSEER_MODEL_INPUT` and `LANDSEER_MODEL_PATH` in
their environment; the PyTorch-TensorFlow adapter turns them into
`--weights none` (architecture) or `--weights required` (weights) and
`--model-path`. Flags in an adapter's `command` still win.

Rules:

- `framework`, `model_input`, `model_output` and `model_path` in a tool's
  `tools.yaml` entry override its labels; labels are not fetched when
  `tools.yaml` already says everything (or says `framework: pytorch`).
- An adapter with `auto_select: false` is only used when named explicitly
  (`config_model_to_tensorflow`). Two auto-selectable adapters for the same
  conversion are an error.
- Frameworks no adapter converts (e.g. `numpy` data tools) without
  `model_input`/`model_output` get no adapters, as before.
- A TensorFlow tool without `model_input`/`model_output` (or whose labels
  cannot be fetched, with `framework: tensorflow` in `tools.yaml`), or with no
  matching adapter, fails pipeline construction with a message saying what to
  declare.
- `LANDSEER_AUTO_ADAPTERS=0` turns label lookups off.

The selected adapters are written into the task config, so they are part of
the task hash and cache key exactly like explicit ones; relabelling an image
makes new tasks.

## How they run

The worker runs **input adapter -> tool -> output adapter inside one task**
(`src/worker/adapters.py`, called from `TaskRunner.run_task`):

| Step | `/input` | `/output` | Afterwards |
|---|---|---|---|
| input adapter | task input (dataset, ancestor outputs, `config_model.py`, `config.json`) | `<task>/adapters/input` | its files are overlaid onto the task input |
| tool | task input (now including the input adapter's files) | task output | unchanged tool contract |
| output adapter | task input with the tool's outputs overlaid | `<task>/adapters/output` | its files are overlaid onto the task output, replacing tool files with the same name |

All three containers are started by the same runner (Docker or Apptainer) with
the same mounts (`/input`, `/output`, `/data`, `config_model.py` in `/app`),
environment, GPU and timeout as a plain tool. Adapters additionally get
`LANDSEER_ADAPTER=input|output`, `LANDSEER_TOOL_IMAGE=<tool image>` and their
own `env`. On Docker the worker pulls adapter images like tool images.

The task fails at the first step that fails, and its error names the step
(`Input adapter 'config-model-to-tensorflow' exited with code 1`). The task log
holds the logs and commands of every step that ran.

**The tool must save its model at `/output/<model_path>`.** The input
adapter's untrained model is also under `/input/<model_path>`, so the output
adapter alone cannot tell a trained model from a tool that saved elsewhere, in
another format, or not at all. Before running the output adapter the worker
checks that the tool wrote `/output/<model_path>` and otherwise fails the task
(`Tool did not write its model to /output/model_tf/model.keras; ...`).
Adapters in task configs written before this check have no `model_path` and
run unchecked, with a warning.

### Identity, caching and lineage

`TaskFactory.create_task` copies the tool's adapters (name, image, command,
env) into the task config as `config["adapters"]`. Everything that identifies a
task already includes its config, so adapters are part of the task hash and
deduplication, the run-scoped lineage hash used for task IDs, and the worker's
cache key. Changing an adapter image or command makes a new task; the stored
config also tells the worker what to run after a backend restart.

### Why inside the task, not as separate DAG tasks

- **Downstream only ever sees standard outputs.** A task's output (and its
  cache entry) is already PyTorch. Separate converter tasks would put
  non-standard intermediates into the cache/MinIO and into the ancestor merge
  of later tasks.
- **No scheduler or workflow changes.** Stages, workflow generation,
  priorities, stage validation and the UI keep working with one task per tool;
  adapters are an execution detail of that task.
- **One identity and one failure.** The adapters are covered by the task's
  hash/lineage/cache key without new task types, and a conversion failure fails
  the tool's task with a message saying which step failed.
- **Runtime parity for free.** The chain reuses the existing Docker/Apptainer
  runner, so mounts, GPU flags and environment behave the same.

The cost: adapters re-run when the task re-runs, and one conversion cannot be
shared by several tools. Conversion takes seconds next to training, so this
was accepted. (`tests/backend/scheduler/test_converter_invocation.py` describes
an older planner design with separate converter tasks; it is skipped because
that module does not exist.)

## Adapter contract

An adapter is a normal Landseer container: it reads `/input`, writes
`/output`, exits non-zero on failure, and must be deterministic for a given
input. It should verify its own conversion (compare source and converted
models on a batch of the task's data) and fail rather than emit a wrong model.

- **Input adapters** add files for the tool; they never need to copy the
  standard inputs, which stay in `/input`.
- **Output adapters** must leave the task output in standard form, i.e. write
  `model.pt` as a `state_dict` for `config_model.config()`, which is what
  PyTorch tools write and what `containers/evals/common/model_loader.py` loads.
  Other tool outputs (metrics, the tool's native model) stay in the task output.

## The PyTorch <-> TensorFlow adapter

`containers/adapters/pytorch_tensorflow` (one image, two directions):

**`--direction input`**: builds `config_model.config()` (and loads `/input/model.pt`
with `--weights auto|required`; `--weights none` keeps the fresh
initialisation, used by in-training tools that train from scratch like TRADES
does) and converts it with [nobuco](https://github.com/AlexanderLutsenko/nobuco)
into a trainable Keras functional model made of standard Keras layers. Writes:

- `/output/model_tf/model.keras`: input NCHW float32 `[0, 1]` (the dataset
  layout), output logits; batch size is dynamic.
- `/output/model_tf/weight_map.json`: which Keras weight holds which PyTorch
  tensor and with which axis permutation. It is found by converting once with
  unique random marker values in every tensor and matching them exactly, so it
  works for any architecture nobuco converts, without name heuristics.

Both models are compared on samples of `test_data.npy`; the adapter fails if
logits differ by more than `atol + rtol * max|logit|` (default `1e-3`).

**`--direction output`**: loads `/input/model_tf/model.keras` (the tool's
trained model, which replaced the input adapter's file) and `weight_map.json`
(from the input adapter), copies every weight back into `config_model.config()`
with the inverse permutation, loads it with `strict=True`, compares logits again
and writes `/output/model.pt` plus `/output/conversion_report.json` (parity and
accuracy of both models on the sample).

Limitations: only operations nobuco can convert are supported; `view`/`reshape`
calls whose leading dimension equals the trace batch size (11, `--trace-batch`)
are treated as batch-dynamic; `num_batches_tracked` buffers are not trained in
TensorFlow and keep their initial value.

### Contract for TensorFlow tools

- Label the image: `org.landseer.framework`, `org.landseer.model_input`,
  `org.landseer.model_output` and, if not `model_tf/model.keras`,
  `org.landseer.model_path`.
- Read the model from `/input/<model_path>`
  (`tf.keras.models.load_model(..., compile=False)`); it takes NCHW `[0, 1]`
  input like the PyTorch model.
- Train it in place (keep its layers; do not rebuild or rename them).
- Save it to `/output/<model_path>`; the task fails if it is missing.
- Read data from the standard `.npy` files in `/input`.

### Native TensorFlow model definitions

`configs/model/config_model_resnet20_tf.py` and
`configs/model/config_model_resnet20_celeba_tf.py` are hand-written Keras
equivalents of `config_model_resnet20.py` / `config_model_resnet20_celeba.py`
(10 and 2 classes). `config()` returns a Keras model with the same
architecture, NCHW input, padding, BatchNorm eps/momentum, initialisation and
final pooling, layers named after the PyTorch modules, plus
`load_torch_state_dict` / `to_torch_state_dict` / `weight_map`.
`tests/models/test_resnet20_tf_parity.py` checks them against the PyTorch
models: in float64 logits, loss, every gradient and an SGD step agree to about
1e-12, so they are the same function. The one known difference: during
training PyTorch updates `running_var` with the unbiased batch variance, Keras
with the biased one (relative 1e-5 to 1e-4 per step). The test needs torch and
tensorflow; run it in the adapter image:

```bash
docker run --rm -v "$PWD":/repo:ro -w /repo landseer-local/adapter_pytorch_tensorflow:dev     sh -c "pip install -q pytest && python -m pytest tests/models -q -p no:cacheprovider"
```

## `in_freeat`

`containers/tools/in_freeat` runs Free Adversarial Training
([paper](https://arxiv.org/abs/1904.12843),
[code](https://github.com/ais2397/free_adv_train_fork)) in TensorFlow 2.15. It
keeps the original algorithm and arguments of `free_train.py` (`--replay_m`,
`--epsilon` on the 0-255 scale, `--train_batch_size`, `--weight_decay`,
`--momentum`, `--step_size_schedule`, `--train_steps`, seeds) but trains the
pipeline's `config_model.py` network provided by the input adapter instead of
the repository's fixed WRN-28-10, because the rest of the pipeline loads
`model.pt` into `config_model.py`. The original TF1 graph code (`tf.contrib`,
sessions) does not run on TF2, so the loop is written with `tf.GradientTape`.

Extra arguments: `--epochs` (passes over the data, each minibatch replayed
`m` times; default 26 is about the original 80000 steps), `--no_augment`,
`--max_train_examples`. Outputs: `model_tf/model.keras`, `freeat_metrics.json`,
and (from the output adapter) `model.pt` and `conversion_report.json`.

## `in_madry`

`containers/tools/in_madry` runs PGD adversarial training
([paper](https://arxiv.org/abs/1706.06083),
[code](https://github.com/MadryLab/cifar10_challenge)) in TensorFlow 2.15, ported
from the official TF1 `train.py` and `pgd_attack.LinfPGDAttack` for the same
reason as `in_freeat`: it trains the pipeline's `config_model.py` network. The
port keeps the original's semantics: `tf.compat.v1.train.MomentumOptimizer`,
`tf.train.piecewise_constant` boundaries, weight decay on conv/dense kernels,
pad-2 crop + flip augmentation, `DataSubset` batching, and the attack running on
the training-mode model (batch norm updates its moving averages during attack
steps, as `tf.contrib.layers.batch_norm(updates_collections=None)` does). Its
arguments are the original `config.json` keys (`--max_num_training_steps`,
`--epsilon` and `--step_size` on the 0-255 scale, `--num_steps`,
`--step_size_schedule`, ...); the default `--num_steps` is the paper's 7 (the
repository ships 10). `containers/tools/in_madry/test_main.py` checks these
against TF1 behaviour inside the image.

Extra arguments: `--epochs`, `--eval_num_steps` (PGD steps of the final robust
accuracy, default 20), `--no_augment`, `--max_train_examples`. Outputs:
`model_tf/model.keras`, `madry_metrics.json`, and (from the output adapter)
`model.pt` and `conversion_report.json`.

The Dockerfile deletes the base image's `/etc/ld.so.conf.d/z-cuda-stubs.conf`:
with it, the linker cache resolves `libcuda.so.1` to the CUDA stub before the
host driver, `cuInit` fails with error 34 and TensorFlow silently runs on the
CPU (about 8x slower here). Images built on `tensorflow/tensorflow:2.15.0-gpu`
without that line, such as `in_freeat`'s, have the same problem; check with
`python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.

## Building and testing locally

```bash
docker build -t landseer-local/adapter_pytorch_tensorflow:dev containers/adapters/pytorch_tensorflow
docker build -t landseer-local/in_freeat:dev containers/tools/in_freeat
docker build -t landseer-local/in_madry:dev containers/tools/in_madry
```

`configs/tools.yaml` references `ghcr.io/landseer-project/adapter_pytorch_tensorflow:v1`
and `ghcr.io/landseer-project/in_freeat:v1`; publish those tags (or point the
entries at local tags) before running `configs/pipeline/freeat_cifar10.yaml`.
`in_freeat` names no adapters in `tools.yaml`, so its image must carry the
`model_*` labels from its Dockerfile (rebuild images built before them).
Unit tests: `tests/pipeline/test_tool_adapters_unit.py`,
`tests/pipeline/test_auto_adapters_unit.py`, `tests/worker/test_adapter_chain.py`.
