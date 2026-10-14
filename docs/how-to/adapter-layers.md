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
`configs/tools.yaml` and referenced by key from a tool:

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
    framework: tensorflow          # informational; warns if non-pytorch without an output adapter
    adapters:
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

- Read the model from `/input/model_tf/model.keras`
  (`tf.keras.models.load_model(..., compile=False)`); it takes NCHW `[0, 1]`
  input like the PyTorch model.
- Train it in place (keep its layers; do not rebuild or rename them).
- Save it to `/output/model_tf/model.keras`.
- Read data from the standard `.npy` files in `/input`.

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

## Building and testing locally

```bash
docker build -t landseer-local/adapter_pytorch_tensorflow:dev containers/adapters/pytorch_tensorflow
docker build -t landseer-local/in_freeat:dev containers/tools/in_freeat
```

`configs/tools.yaml` references `ghcr.io/landseer-project/adapter_pytorch_tensorflow:v1`
and `ghcr.io/landseer-project/in_freeat:v1`; publish those tags (or point the
entries at local tags) before running `configs/pipeline/freeat_cifar10.yaml`.
Unit tests: `tests/pipeline/test_tool_adapters_unit.py`,
`tests/worker/test_adapter_chain.py`.
