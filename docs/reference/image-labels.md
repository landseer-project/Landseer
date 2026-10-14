# Image labels

Every Landseer image (defense tools, dataset loaders and evaluators) carries a fixed
set of labels so it can be identified, and so the pipeline can tell what it needs
and what it supports without pulling or running it. The schema is defined in
`src/pipeline/image_labels.py` (schema version `1`).

Two namespaces are used:

* `org.opencontainers.image.*`: the [OCI standard annotations][oci] for identity
  and provenance. Only keys defined by the OCI spec go here.
* `org.landseer.*`: Landseer-specific metadata.

List values are comma-separated with no spaces. `*` means "any".

## Labels every image declares

| Label | Example | Meaning |
|---|---|---|
| `org.opencontainers.image.title` | `Clean Evaluator` | Human-readable name |
| `org.opencontainers.image.description` | `Evaluates clean top-1 accuracy` | One line on what it does |
| `org.opencontainers.image.vendor` | `Landseer Project` | Always this value |
| `org.opencontainers.image.source` | `https://github.com/landseer-project/landseer` | Repo the image is built from |
| `org.opencontainers.image.base.name` | `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` | Must match the final `FROM` |
| `org.landseer.schema` | `1` | Label schema version |
| `org.landseer.type` | `tool` / `dataset` / `evaluator` | Kind of image |
| `org.landseer.name` | `clean`, `cifar10`, `in_trades` | Canonical id (directory or tool id) |
| `org.landseer.datasets` | `cifar10,celeba` or `*` | Datasets the image supports; read by the dataset compatibility check |
| `org.landseer.modality` | `image` / `text` / `tabular` | Data modality |
| `org.landseer.framework` | `pytorch==2.1.0` | Primary ML framework and version |
| `org.landseer.python` | `3.10` | Python version in the image |
| `org.landseer.accelerator` | `cuda` / `cpu` | Hardware needed |
| `org.landseer.cuda` | `11.8` | CUDA version; required when accelerator is `cuda` |

Optional on any image:

| Label | Example | Meaning |
|---|---|---|
| `org.landseer.dependencies` | `numpy==1.24.0,scikit-learn==1.3.0` | Key pinned packages installed on top of the base |
| `org.landseer.network` | `required` | Needs internet at run time (downloads data); matters on air-gapped clusters |
| `org.landseer.inputs` | `/source_artifacts:list_attr_celeba.csv,img_align_celeba` | Host data that must be mounted, as `mount:entries` |
| `org.opencontainers.image.licenses` | `MIT` | SPDX license expression |

Stamped at build time, never written in the Dockerfile (CI's
`docker/metadata-action` adds them; for manual builds pass `--label`):

* `org.opencontainers.image.version`
* `org.opencontainers.image.revision`
* `org.opencontainers.image.created`

## Labels by image type

**Tool** (`org.landseer.type=tool`)

| Label | Example | Meaning |
|---|---|---|
| `org.landseer.stage` | `pre_training` / `during_training` / `post_training` / `deployment` | Pipeline stage; read by stage validation when `tools.yaml` has no `defense_stage` |
| `org.landseer.defense_type` | `adversarial` | Defense family; matches evaluator `defense_types` |

**Dataset loader** (`org.landseer.type=dataset`)

| Label | Example | Meaning |
|---|---|---|
| `org.landseer.dataset` | `celeba` | Dataset key in `configs/datasets.yaml`; `org.landseer.datasets` must equal it |
| `org.landseer.variant` | `clean` / `poisoned` | Variant key in `configs/datasets.yaml` |
| `org.landseer.outputs` | `data.npy,labels.npy,test_data.npy,test_labels.npy` | Files written to `/output` |
| `org.landseer.attack` | `badnets` | Optional: poisoning attack for poisoned variants |

**Evaluator** (`org.landseer.type=evaluator`)

| Label | Example | Meaning |
|---|---|---|
| `org.landseer.metrics` | `clean_accuracy,clean_train_accuracy` | Metric keys the evaluator emits |
| `org.landseer.required_artifacts` | `poisoning_metadata.json` | Files it needs from upstream; may be empty |
| `org.landseer.defense_types` | `backdoor` | Defense families it applies to; may be empty |

The lint warns when an evaluator's `metrics`, `required_artifacts` or
`defense_types` labels disagree with its entry in `configs/evaluators.yaml`.

## Legacy keys

Older images use keys that put Landseer data in the reserved OCI namespace. They
are still read as fallbacks and the checks report them as warnings:

| Legacy | Use instead |
|---|---|
| `org.opencontainers.image.dataset` | `org.landseer.datasets` |
| `org.opencontainers.image.stage`, `org.opencontainers.image.defense_stage` | `org.landseer.stage` |
| `org.landseer.category` | `org.landseer.type` |

## Template for a tool image

Tool images are built in their own repositories. Copy this into the tool's
Dockerfile right after the final `FROM`:

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL org.opencontainers.image.title="TRADES" \
      org.opencontainers.image.description="Adversarial training with the TRADES loss" \
      org.opencontainers.image.vendor="Landseer Project" \
      org.opencontainers.image.source="https://github.com/landseer-project/<tool-repo>" \
      org.opencontainers.image.base.name="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime" \
      org.landseer.schema="1" \
      org.landseer.type="tool" \
      org.landseer.name="in_trades" \
      org.landseer.stage="during_training" \
      org.landseer.defense_type="adversarial" \
      org.landseer.datasets="cifar10,celeba" \
      org.landseer.modality="image" \
      org.landseer.framework="pytorch==2.1.0" \
      org.landseer.python="3.10" \
      org.landseer.accelerator="cuda" \
      org.landseer.cuda="11.8"
```

For a manual build, stamp the provenance labels too:

```bash
docker build \
  --label org.opencontainers.image.version=v4 \
  --label org.opencontainers.image.revision="$(git rev-parse HEAD)" \
  --label org.opencontainers.image.created="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t ghcr.io/landseer-project/in_trades:v4 .
```

## Checking labels

```bash
# Dockerfiles under containers/ (stdlib only; also runs in CI and pytest)
python src/pipeline/image_labels.py lint

# A Dockerfile in another repo
python src/pipeline/image_labels.py lint --type tool path/to/Dockerfile

# Published images referenced by configs/tools.yaml, datasets.yaml, evaluators.yaml
python -m src.pipeline.image_labels audit

# Specific images
python -m src.pipeline.image_labels audit --type tool ghcr.io/landseer-project/in_trades:v4
```

`audit` uses `docker image inspect` when the image is local, otherwise the
registry API, and requires the build-stamped labels as well.

[oci]: https://github.com/opencontainers/image-spec/blob/main/annotations.md
