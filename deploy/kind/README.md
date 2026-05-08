# Local Kubernetes development

This guide walks through running Landseer end-to-end against a local
[kind](https://kind.sigs.k8s.io/) cluster, using the `KubernetesRunner`
worker path. It mirrors what you would deploy on AKS / GKE so you can
develop and test the K8s flow without a managed cluster.

The setup:

```
┌─────────────────────────┐         ┌─────────────────────────────┐
│ HOST (your laptop)      │         │ kind cluster                │
│                         │         │  namespace: landseer-workers│
│  ┌───────────────────┐  │         │                             │
│  │ backend           │  │  HTTP   │  ┌───────────────────────┐  │
│  └─────────┬─────────┘  │◀───────▶│  │ MinIO (PVC-backed)    │  │
│            │            │         │  └───────────────────────┘  │
│  ┌─────────▼─────────┐  │ kubectl │                             │
│  │ worker            │──┼────────▶│  ┌───────────────────────┐  │
│  │ --runtime k8s     │  │         │  │ Job/Pod (per task)    │  │
│  └───────────────────┘  │         │  │  initContainer + tool │  │
│                         │         │  └───────────────────────┘  │
└─────────────────────────┘         └─────────────────────────────┘
            │                                       ▲
            └──── upload/download via MinIO ────────┘
                  (host: localhost:9000, in-cluster: minio.svc:9000)
```

> GPU is not available inside kind on macOS. Use the local cluster for
> functional testing only; run real GPU workloads on a managed cluster
> with the NVIDIA device plugin installed.

---

## 1. Prerequisites (one-time)

Install on the host:

```bash
brew install kind kubectl minio/stable/mc
```

Install Python deps (zaciąga `kubernetes` client):

```bash
poetry install
```

Verify nothing else is using port 9000 (kind maps MinIO there):

```bash
lsof -nP -iTCP:9000 -sTCP:LISTEN          # should print nothing
```

If the port is busy (often a leftover MinIO from earlier work),
stop that container first:

```bash
docker ps | grep minio
docker stop <container-name>
```

---

## 2. Bring up the cluster

From the repo root:

```bash
make kind-up        # 3-node kind cluster (1 control-plane + 2 workers)
make k8s-apply      # namespace, RBAC, MinIO Deployment + PVC + Services
make k8s-status     # MinIO Pod Running, PVC Bound
```

Sanity-check MinIO is reachable from the host:

```bash
mc alias set local http://localhost:9000 minioadmin minioadmin
mc ls local                               # bucket auto-created on first task
```

---

## 3. Run backend, worker, and trigger a pipeline

Open three terminals.

### Terminal 1 — backend

```bash
poetry run python -m src.backend.cli --config configs/pipeline/mini.yaml
```

Wait for `Uvicorn running on http://0.0.0.0:8000`.

### Terminal 2 — worker (Kubernetes runtime)

> On macOS the default workspace path `/data/landseer/...` is on a
> read-only volume. Use `~/.landseer` instead.

```bash
export MINIO_ENDPOINT=localhost:9000
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export LANDSEER_WORKSPACE=~/.landseer/workers
export LANDSEER_CACHE_DIR=~/.landseer/cache

poetry run python -m src.worker.cli \
  --backend-url http://localhost:8000 \
  --runtime kubernetes \
  --no-cache
```

Expected log lines on startup:

- `Loaded local kube config`
- `MinIO store connected: localhost:9000`
- `MinIO lifecycle: k8s-tasks/* expires after 1 day in bucket landseer-artifacts`
- `Registered as worker: worker_xxxxxxxx`

### Terminal 3 — trigger a run

The pipeline configs are auto-discovered by the backend. List them:

```bash
curl -s http://localhost:8000/api/pipeline-configs | jq '.configs[].id'
```

Trigger a run using the **`config_id`** (not the bare name):

```bash
curl -X POST http://localhost:8000/api/pipeline-configs/config_mini/runs \
  -H "Content-Type: application/json" \
  -d '{"use_cache": false}'
```

You should see a JSON response with the new `run_id`.

---

## 4. Verify it works

### Cluster activity

```bash
watch -n 1 'kubectl -n landseer-workers get jobs,pods'
```

Each task spawns a Job that goes through `Init` → `Running` → `Completed`.

### Logs of the most recent task pod

```bash
make k8s-logs
```

Or per-container:

```bash
kubectl -n landseer-workers logs <pod> -c fetch-input    # init: MinIO pull
kubectl -n landseer-workers logs <pod> -c tool           # main: the tool itself
```

### MinIO has the artefacts

```bash
mc ls --recursive local/landseer-artifacts/k8s-tasks/
```

Expect `<task-id>/input/...` (uploaded by worker) and `<task-id>/output/...`
(pushed by the wrapper after the tool finishes).

### Lifecycle policy applied

```bash
mc ilm rule ls local/landseer-artifacts
```

Expect a rule `landseer-k8s-tasks-expire` with prefix `k8s-tasks/` and a
1-day expiration.

### Backend received results

```bash
curl -s http://localhost:8000/api/runs | jq '.[-1]'
```

### Output landed on the worker

```bash
ls -la ~/.landseer/workers/<task-id>/output/
```

### Bonus — orphan-job cleanup

Hit `Ctrl+C` on the worker mid-run; within ~5 s `kubectl get jobs` should
show the active Job has been deleted (the runner's `atexit` handler
cancels in-flight Jobs on shutdown).

---

## 5. Diagnostics

| Symptom | Likely cause | Fix |
|---|---|---|
| Worker logs `KUBERNETES_AVAILABLE: False` | `kubernetes` lib not installed | `poetry install` |
| `OSError: [Errno 30] Read-only file system: '/data'` | macOS default workspace | export `LANDSEER_WORKSPACE` and `LANDSEER_CACHE_DIR` to `~/.landseer/...` |
| `kind create` fails with `Bind for 0.0.0.0:9000 failed: port is already allocated` | another MinIO already listening on 9000 | `lsof -nP -iTCP:9000 -sTCP:LISTEN` then `docker stop <name>` |
| `{"detail": "Pipeline config 'mini' not found"}` | endpoint expects `config_id`, not `name` | use `/api/pipeline-configs/config_mini/runs` |
| PVC `Pending` | StorageClass missing | `kubectl describe pvc minio-data -n landseer-workers` |
| Pod stuck in `Pending` | resource crunch / scheduling | `kubectl describe pod <name>` (check Events) |
| `ImagePullBackOff` on tool container | private GHCR image | `kubectl create secret docker-registry ghcr-pull-secret --docker-server=ghcr.io --docker-username=<user> --docker-password=<pat>` then `export LANDSEER_K8S_IMAGE_PULL_SECRET=ghcr-pull-secret` |
| `Init:Error` | mc init container failed | `kubectl logs <pod> -c fetch-input` |
| Main container `CrashLoopBackOff` | tool errored or `/bin/sh` missing in image | `kubectl logs <pod> -c tool` |
| `OOMKilled` in main container | memory limit too low | `export LANDSEER_K8S_TOOL_MEMORY=8Gi` and restart worker |
| Worker hangs on `_wait_for_job` | pod not starting | `kubectl describe pod <name>` |

---

## 6. Configuration knobs

Set on the **worker** before launch (or via `os.environ`):

| Var | Default | Purpose |
|---|---|---|
| `MINIO_ENDPOINT` | `localhost:9000` | Worker-side MinIO endpoint |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | `minioadmin` / `minioadmin` | MinIO credentials |
| `LANDSEER_WORKSPACE` | `/data/landseer/workers/<worker_id>` | Worker workspace dir (override on macOS) |
| `LANDSEER_CACHE_DIR` | `/data/landseer/cache` | Cache dir (override on macOS) |
| `LANDSEER_K8S_NAMESPACE` | `landseer-workers` | Namespace for Jobs |
| `LANDSEER_K8S_SERVICE_ACCOUNT` | `landseer-worker` | SA Jobs run as |
| `LANDSEER_K8S_MINIO_INCLUSTER_ENDPOINT` | `http://minio.landseer-workers.svc.cluster.local:9000` | MinIO URL injected into pods |
| `LANDSEER_K8S_MINIO_BUCKET` | `landseer-artifacts` | Bucket name |
| `LANDSEER_K8S_MINIO_SECRET_NAME` | `minio-credentials` | K8s Secret with MinIO creds |
| `LANDSEER_K8S_MC_IMAGE` | `minio/mc:latest` | Image for init/upload helpers |
| `LANDSEER_K8S_IMAGE_PULL_SECRET` | (empty) | `imagePullSecrets` for private tool images |
| `LANDSEER_K8S_TOOL_MEMORY` | `4Gi` | Tool container memory request+limit |
| `LANDSEER_K8S_TOOL_CPU` | `1` | Tool container CPU request (no limit, no throttle) |

---

## 7. Tear down

```bash
make k8s-delete    # remove namespace & resources (also deletes the MinIO PVC)
make kind-down     # destroy the cluster
```

> The MinIO data lives on a `PersistentVolumeClaim` (20Gi, kind's built-in
> `standard` StorageClass). Cache survives MinIO Pod restarts but is
> destroyed when the namespace is deleted or `kind-down` removes the
> cluster. To wipe the cache without dropping the cluster:
>
> ```bash
> kubectl -n landseer-workers delete pvc minio-data
> kubectl -n landseer-workers rollout restart deploy/minio
> ```

---

## 8. Credentials — IMPORTANT before going past local dev

`deploy/k8s/minio.yaml` ships hardcoded `minioadmin / minioadmin` in the
`minio-credentials` Secret. **This is intentional for local kind only** —
do NOT apply this manifest to a shared or production cluster as-is.

For AKS / GKE / any non-local cluster:

1. Generate strong credentials and create the Secret out-of-band:
   ```bash
   kubectl -n landseer-workers create secret generic minio-credentials \
     --from-literal=MINIO_ROOT_USER="$(openssl rand -hex 16)" \
     --from-literal=MINIO_ROOT_PASSWORD="$(openssl rand -base64 32)"
   ```
2. Apply the rest of the manifests with the Secret excluded (or use Sealed
   Secrets / External Secrets).
3. Surface the same creds to the worker via `MINIO_ACCESS_KEY` /
   `MINIO_SECRET_KEY`.

For long-term hygiene use [Sealed Secrets](https://sealed-secrets.netlify.app/),
[External Secrets Operator](https://external-secrets.io/) backed by Azure
Key Vault / GCP Secret Manager, or your cluster's native equivalent.

---

## 9. Cloud parity (AKS / GKE)

The same manifests apply: change kubeconfig context, set
`LANDSEER_K8S_IMAGE_PULL_SECRET` to a secret holding your GHCR PAT,
replace the MinIO Secret per the section above, and either keep MinIO
in-cluster or point `LANDSEER_K8S_MINIO_INCLUSTER_ENDPOINT` at a managed
S3 endpoint. The runner does not assume kind anywhere.

> Worker-as-Pod (running the worker itself inside the cluster as a
> `Deployment` with auto-scaling) is intentionally **not** part of this
> setup — it ships in a follow-up PR. Today the worker runs as a host
> process and uses your local kubeconfig; that's enough to validate the
> Job creation flow.
