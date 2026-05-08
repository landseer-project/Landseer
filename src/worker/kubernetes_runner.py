"""
Kubernetes container runner for Landseer worker.

Spawns each task as a Kubernetes Job. Input/output are exchanged with the
task pod via MinIO (which the project already uses for artifact caching),
so worker and task pod do not need to share a filesystem.

Layout of a task Job:

  initContainer "fetch-input" (image: minio/mc):
    - copies the mc client binary into a shared emptyDir at /landseer-bin
    - downloads the per-task input prefix from MinIO into emptyDir at /input

  main container "tool" (image: tool image from pipeline YAML):
    - command is wrapped: original command runs, then mc uploads /output to MinIO
    - sees /input (read by the tool), /output (written by the tool), and the
      mc binary at /landseer-bin/mc

The worker uploads input_dir to MinIO before creating the Job and downloads
the produced /output from MinIO once the Job finishes.

Per-task Job volumes are all emptyDir — no PVC is required for task pods.
(MinIO itself uses a PVC for cache durability; that is a deployment
concern, not a runtime one.) Keeping task pods PVC-free makes the cluster
setup identical between local kind and managed clusters (AKS/GKE), at the
cost of two extra round trips through MinIO per task. For a CPU-cached
pipeline that round trip is negligible; for large model artefacts it
amortises with the existing TwoLevelCache reuse.
"""

import atexit
import hashlib
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..common import get_logger
from ..store.minio_store import MinioConfig, MinioStore

logger = get_logger(__name__)

# Lazy import of the kubernetes client so users without K8s installed are
# not forced to pull the dep at import time.
try:
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config
    from kubernetes import watch as k8s_watch
    from kubernetes.client.exceptions import ApiException
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    k8s_client = None
    k8s_config = None
    k8s_watch = None
    ApiException = Exception


_NAME_SAFE_RE = re.compile(r"[^a-z0-9-]+")


def _k8s_safe_name(*parts: str, max_len: int = 53) -> str:
    """Build a DNS-1123 compliant name (lowercase alnum + hyphens, <=63 chars).

    Leaves 10 chars of headroom under the 63-char K8s limit so callers can
    append a short hash suffix without overflowing.
    """
    raw = "-".join(p for p in parts if p)
    raw = raw.lower()
    raw = _NAME_SAFE_RE.sub("-", raw)
    raw = re.sub(r"-+", "-", raw).strip("-")
    if len(raw) > max_len:
        digest = hashlib.blake2s(raw.encode(), digest_size=4).hexdigest()
        raw = f"{raw[: max_len - len(digest) - 1]}-{digest}"
    return raw or "landseer"


class KubernetesRunner:
    """
    Run Landseer task containers as Kubernetes Jobs.

    Mirrors the public ``run`` interface of ``DockerRunner`` and
    ``ApptainerRunner`` so ``TaskRunner._init_container_runner`` can drop it
    in interchangeably.
    """

    def __init__(
        self,
        workspace_dir: Path,
        gpu_id: Optional[int] = None,
        timeout: int = 7200,
        namespace: Optional[str] = None,
        service_account: Optional[str] = None,
        minio_secret_name: Optional[str] = None,
        minio_incluster_endpoint: Optional[str] = None,
        minio_bucket: Optional[str] = None,
        mc_image: Optional[str] = None,
        image_pull_secret: Optional[str] = None,
    ):
        if not KUBERNETES_AVAILABLE:
            raise RuntimeError(
                "kubernetes Python client is not installed. "
                "Add the 'kubernetes' package to your environment."
            )

        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_id = gpu_id
        self.timeout = timeout

        self.namespace = (
            namespace
            or os.environ.get("LANDSEER_K8S_NAMESPACE")
            or "landseer-workers"
        )
        self.service_account = (
            service_account
            or os.environ.get("LANDSEER_K8S_SERVICE_ACCOUNT")
            or "landseer-worker"
        )
        self.minio_secret_name = (
            minio_secret_name
            or os.environ.get("LANDSEER_K8S_MINIO_SECRET_NAME")
            or "minio-credentials"
        )
        self.minio_incluster_endpoint = (
            minio_incluster_endpoint
            or os.environ.get("LANDSEER_K8S_MINIO_INCLUSTER_ENDPOINT")
            or "http://minio.landseer-workers.svc.cluster.local:9000"
        )
        self.minio_bucket = (
            minio_bucket
            or os.environ.get("LANDSEER_K8S_MINIO_BUCKET")
            or os.environ.get("MINIO_BUCKET")
            or "landseer-artifacts"
        )
        self.mc_image = (
            mc_image
            or os.environ.get("LANDSEER_K8S_MC_IMAGE")
            or "minio/mc:latest"
        )
        self.image_pull_secret = (
            image_pull_secret
            or os.environ.get("LANDSEER_K8S_IMAGE_PULL_SECRET")
            or ""
        )

        # Resource sizing for the tool container. Memory is set as both
        # request and limit so the Pod gets Guaranteed QoS for memory and
        # cannot be evicted by other workloads under pressure. CPU is a
        # request only — ML training workloads are bursty and cgroup CPU
        # throttling under a hard limit hurts throughput far more than it
        # helps fairness on a busy node.
        self.tool_memory = (
            os.environ.get("LANDSEER_K8S_TOOL_MEMORY") or "4Gi"
        )
        self.tool_cpu = (
            os.environ.get("LANDSEER_K8S_TOOL_CPU") or "1"
        )

        # Worker-side MinIO client (uses host-visible endpoint via the
        # existing MinioConfig env-driven defaults).
        self._minio = MinioStore(MinioConfig())
        if not self._minio.is_available:
            raise RuntimeError(
                "KubernetesRunner requires MinIO to be reachable from the worker. "
                "Check MINIO_ENDPOINT / credentials and that MinIO is running."
            )

        # Auto-expire k8s-task scratch objects after 1 day so per-task
        # input/output bundles don't accumulate forever in the bucket.
        # NOTE: minio's set_bucket_lifecycle REPLACES the bucket-wide
        # policy, it does not merge — any other lifecycle rules on the
        # bucket would be overwritten. Failures are non-fatal — we just log.
        self._configure_task_lifecycle()

        # Load kube config: in-cluster first (for worker-as-Pod), then
        # ~/.kube/config (for worker-as-host-process during local dev).
        try:
            k8s_config.load_incluster_config()
            logger.info("Loaded in-cluster kube config")
        except k8s_config.ConfigException:
            k8s_config.load_kube_config()
            logger.info("Loaded local kube config")

        self._batch = k8s_client.BatchV1Api()
        self._core = k8s_client.CoreV1Api()

        # Track Jobs we have submitted but not yet cleaned up. On worker
        # shutdown (Ctrl+C / SIGTERM / unexpected exit) we delete them so
        # the cluster does not keep running orphaned tasks whose output
        # nobody will ever pick up.
        self._active_jobs: Set[str] = set()
        self._active_jobs_lock = threading.Lock()
        atexit.register(self.shutdown)

    def shutdown(self) -> None:
        """Delete every Job this runner submitted that has not been reaped.

        Called automatically via ``atexit`` when the worker process exits
        (graceful Ctrl+C / SIGTERM); safe to call manually too. Idempotent.
        """
        with self._active_jobs_lock:
            jobs = list(self._active_jobs)
            self._active_jobs.clear()
        if not jobs:
            return
        logger.info(f"Cleaning up {len(jobs)} orphan Job(s) on shutdown")
        for name in jobs:
            try:
                self._batch.delete_namespaced_job(
                    name=name,
                    namespace=self.namespace,
                    propagation_policy="Background",
                )
            except ApiException as e:
                # 404 = already gone. Anything else we just log and move on
                # — atexit handlers should never raise.
                if e.status != 404:
                    logger.warning(f"Could not delete orphan Job {name}: {e}")

    def _configure_task_lifecycle(self) -> None:
        """Install a 1-day expiration rule on k8s-tasks/* objects.

        We use the underlying minio client directly because MinioStore does
        not expose lifecycle helpers. ``set_bucket_lifecycle`` REPLACES the
        bucket policy rather than merging — repeated calls are safe for our
        single rule, but any unrelated rule on the same bucket would be
        wiped out. If the bucket needs to host multiple rules in the
        future, switch to a get-merge-set flow.
        """
        from minio.commonconfig import Filter
        from minio.lifecycleconfig import LifecycleConfig, Rule, Expiration

        try:
            client = self._minio._client
            bucket = self._minio.config.bucket
            if client is None:
                return
            policy = LifecycleConfig([
                Rule(
                    rule_id="landseer-k8s-tasks-expire",
                    rule_filter=Filter(prefix="k8s-tasks/"),
                    status="Enabled",
                    expiration=Expiration(days=1),
                ),
            ])
            client.set_bucket_lifecycle(bucket, policy)
            logger.info(
                "MinIO lifecycle: k8s-tasks/* expires after 1 day in bucket %s",
                bucket,
            )
        except Exception as e:
            # Lifecycle is a nice-to-have; failure should not block task
            # execution. Worst case the bucket grows over time.
            logger.warning(f"Failed to set MinIO lifecycle policy: {e}")

    # ------------------------------------------------------------------
    # Public API (matches DockerRunner/ApptainerRunner)
    # ------------------------------------------------------------------

    def run(
        self,
        image: str,
        command: str,
        input_dir: Path,
        output_dir: Path,
        env: Optional[Dict[str, str]] = None,
        extra_mounts: Optional[Dict[str, str]] = None,
        model_script_path: Optional[Path] = None,
    ) -> Tuple[int, str, str]:
        """Run a single task as a Kubernetes Job.

        Returns (exit_code, logs, command_string) to match the other runners.
        ``model_script_path`` is unused: TaskRunner already copies it into
        ``input_dir`` and PYTHONPATH=/input makes it importable.
        ``extra_mounts`` is currently ignored — Landseer's pipeline does not
        rely on it for K8s deployments.
        """
        if extra_mounts:
            logger.debug(
                "KubernetesRunner ignores extra_mounts (received %d entries); "
                "use MinIO-backed inputs instead.",
                len(extra_mounts),
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        task_id = self._derive_task_id(input_dir, image, command)
        job_name = _k8s_safe_name("landseer", "task", task_id)
        input_prefix = f"k8s-tasks/{task_id}/input"
        output_prefix = f"k8s-tasks/{task_id}/output"

        # 1) Upload input bundle to MinIO so the Pod can pull it.
        try:
            uploaded = self._minio.upload_directory(input_dir, input_prefix)
            logger.info(
                "Uploaded %d input files to minio://%s/%s",
                uploaded, self.minio_bucket, input_prefix,
            )
        except Exception as e:
            logger.error(f"Failed to upload input to MinIO: {e}")
            return -1, f"input upload failed: {e}", ""

        # 2) Build and submit Job.
        job = self._build_job(
            job_name=job_name,
            image=image,
            command=command,
            env=env or {},
            input_prefix=input_prefix,
            output_prefix=output_prefix,
        )
        cmd_str = f"kubectl apply -f <Job/{job_name}>"
        logger.info(f"Submitting Job: {job_name} (image={image})")

        try:
            self._batch.create_namespaced_job(self.namespace, job)
        except ApiException as e:
            logger.error(f"Failed to create Job {job_name}: {e}")
            return -1, f"Job submission failed: {e}", cmd_str
        # Track the Job so shutdown() can reap it if the worker exits
        # mid-task. Removed in the finally below once we delete it.
        with self._active_jobs_lock:
            self._active_jobs.add(job_name)

        # 3) Tail logs in a background thread while we wait for completion.
        log_buffer: List[str] = []
        log_lock = threading.Lock()
        log_stop = threading.Event()
        log_thread = threading.Thread(
            target=self._stream_pod_logs,
            args=(job_name, log_buffer, log_lock, log_stop),
            name=f"k8s-logs-{job_name}",
            daemon=True,
        )
        log_thread.start()

        # 4-5) Wait for the Job, then pull output from MinIO. Both happen
        # under a single try/finally so the log thread stop signal AND the
        # Job delete always run, even if either step raises. Without the
        # finally a MinIO download exception would leave the Job around
        # until ttlSecondsAfterFinished kicks in.
        exit_code = -1
        try:
            try:
                exit_code = self._wait_for_job(job_name)
            except Exception as e:
                logger.exception(f"Error while waiting on Job {job_name}: {e}")

            # Best-effort output pull. Partial outputs help debugging, and
            # for evaluators the output file itself communicates failure.
            try:
                downloaded = self._minio.download_directory(output_prefix, output_dir)
                logger.info(
                    "Downloaded %d output files from minio://%s/%s",
                    downloaded, self.minio_bucket, output_prefix,
                )
            except Exception as e:
                logger.warning(f"Failed to download output from MinIO: {e}")
        finally:
            # Stop log tail and reap thread before reading the buffer.
            log_stop.set()
            log_thread.join(timeout=5.0)
            # Delete the Job so resources are released before the TTL
            # window expires. Background propagation lets the API return
            # immediately while kubelet cleans up the Pod.
            try:
                self._batch.delete_namespaced_job(
                    name=job_name,
                    namespace=self.namespace,
                    propagation_policy="Background",
                )
            except ApiException as e:
                logger.debug(f"Job {job_name} delete call failed (already gone?): {e}")
            with self._active_jobs_lock:
                self._active_jobs.discard(job_name)

        with log_lock:
            logs = "".join(log_buffer)

        return exit_code, logs, cmd_str

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_task_id(input_dir: Path, image: str, command: str) -> str:
        """Build a short, K8s-safe id unique to this task invocation.

        The TaskRunner workspace path already contains the Landseer task id;
        we add a short uuid so retries land in distinct MinIO prefixes.
        """
        parent = input_dir.parent.name or "task"
        suffix = uuid.uuid4().hex[:6]
        signature = hashlib.blake2s(
            f"{image}|{command}".encode(), digest_size=3
        ).hexdigest()
        return _k8s_safe_name(parent, signature, suffix, max_len=40)

    def _build_job(
        self,
        job_name: str,
        image: str,
        command: str,
        env: Dict[str, str],
        input_prefix: str,
        output_prefix: str,
    ) -> "k8s_client.V1Job":
        # Env vars surfaced to BOTH the init container (for mc) and the
        # main container (for the wrapper script). Credentials are pulled
        # from the cluster Secret via envFrom on each container that needs
        # them, so they are never serialised into the Job manifest.
        shared_env = [
            k8s_client.V1EnvVar(name="MINIO_URL", value=self.minio_incluster_endpoint),
            k8s_client.V1EnvVar(name="MINIO_BUCKET", value=self.minio_bucket),
            k8s_client.V1EnvVar(name="LANDSEER_INPUT_PREFIX", value=input_prefix),
            k8s_client.V1EnvVar(name="LANDSEER_OUTPUT_PREFIX", value=output_prefix),
        ]
        # Landseer task-config env (passed via TaskRunner.run). These are
        # plain strings so they go straight in.
        task_env = [
            k8s_client.V1EnvVar(name=k, value=str(v))
            for k, v in env.items()
            if v is not None
        ]
        # Standard Landseer container env (parity with DockerRunner).
        landseer_env = [
            k8s_client.V1EnvVar(name="INPUT_DIR", value="/input"),
            k8s_client.V1EnvVar(name="OUTPUT_DIR", value="/output"),
            k8s_client.V1EnvVar(name="WORKSPACE", value="/"),
            k8s_client.V1EnvVar(name="PYTHONPATH", value="/input:/app"),
        ]
        if self.gpu_id is not None:
            landseer_env.extend([
                k8s_client.V1EnvVar(name="NVIDIA_VISIBLE_DEVICES", value="all"),
                k8s_client.V1EnvVar(name="NVIDIA_DRIVER_CAPABILITIES", value="all"),
                k8s_client.V1EnvVar(name="CUDA_VISIBLE_DEVICES", value=str(self.gpu_id)),
            ])

        # Volumes shared between init and main containers.
        volumes = [
            k8s_client.V1Volume(name="input", empty_dir=k8s_client.V1EmptyDirVolumeSource()),
            k8s_client.V1Volume(name="output", empty_dir=k8s_client.V1EmptyDirVolumeSource()),
            k8s_client.V1Volume(
                name="landseer-bin",
                empty_dir=k8s_client.V1EmptyDirVolumeSource(),
            ),
            k8s_client.V1Volume(
                name="dshm",
                empty_dir=k8s_client.V1EmptyDirVolumeSource(
                    medium="Memory", size_limit="8Gi"
                ),
            ),
        ]

        init_mounts = [
            k8s_client.V1VolumeMount(name="input", mount_path="/input"),
            k8s_client.V1VolumeMount(name="landseer-bin", mount_path="/landseer-bin"),
        ]
        main_mounts = [
            k8s_client.V1VolumeMount(name="input", mount_path="/input", read_only=False),
            k8s_client.V1VolumeMount(name="output", mount_path="/output"),
            k8s_client.V1VolumeMount(name="landseer-bin", mount_path="/landseer-bin", read_only=True),
            k8s_client.V1VolumeMount(name="dshm", mount_path="/dev/shm"),
        ]

        env_from = [
            k8s_client.V1EnvFromSource(
                secret_ref=k8s_client.V1SecretEnvSource(name=self.minio_secret_name)
            )
        ]

        # initContainer: pull input bundle from MinIO into /input AND drop the
        # mc binary into /landseer-bin so the main container can use it
        # without needing mc baked into the tool image.
        init_script = (
            "set -e\n"
            'echo "[init] preparing mc binary and pulling input"\n'
            'cp "$(command -v mc)" /landseer-bin/mc\n'
            "chmod +x /landseer-bin/mc\n"
            'mc alias set src "$MINIO_URL" "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" --api S3v4\n'
            'if mc ls "src/$MINIO_BUCKET/$LANDSEER_INPUT_PREFIX/" >/dev/null 2>&1; then\n'
            '  mc cp --recursive "src/$MINIO_BUCKET/$LANDSEER_INPUT_PREFIX/" /input/\n'
            "else\n"
            '  echo "[init] no input prefix found at $LANDSEER_INPUT_PREFIX (continuing with empty /input)"\n'
            "fi\n"
        )
        init_container = k8s_client.V1Container(
            name="fetch-input",
            image=self.mc_image,
            image_pull_policy="IfNotPresent",
            command=["/bin/sh", "-c", init_script],
            env=shared_env,
            env_from=env_from,
            volume_mounts=init_mounts,
        )

        # Main container command wrapper. Order matters:
        #   1. Run the user's command, capture its exit code.
        #   2. Push /output to MinIO regardless (best-effort, swallow upload errors).
        #   3. Exit with the user's exit code so K8s reports the right state.
        # We rely on /bin/sh being present in the tool image. All Landseer
        # tool images derive from python:3.x-* which ships sh.
        wrapper = (
            "set +e\n"
            f"{command}\n"
            "_rc=$?\n"
            "/landseer-bin/mc alias set out \"$MINIO_URL\" \"$MINIO_ROOT_USER\" \"$MINIO_ROOT_PASSWORD\" --api S3v4 >/dev/null 2>&1\n"
            "/landseer-bin/mc cp --recursive /output/ \"out/$MINIO_BUCKET/$LANDSEER_OUTPUT_PREFIX/\" 2>&1 || "
            'echo "[wrapper] output upload failed (rc=$?), proceeding with task exit code $_rc"\n'
            "exit $_rc\n"
        )

        # Memory: request == limit. Pod stays under its own memory ceiling
        #         and won't be OOM-killed by overcommit on the node.
        # CPU:    request only (no limit). Avoids cgroup CPU throttling,
        #         which hurts ML throughput far more than it helps fairness.
        # GPU:    limit only — that's the contract the NVIDIA device plugin
        #         expects (resource isn't exposed via requests).
        # QoS class is Burstable (Guaranteed would require request==limit
        # for BOTH memory and cpu). Burstable Pods can in theory be evicted
        # under node memory pressure, but only after BestEffort Pods are
        # gone and only if they exceed their own request — which we make
        # impossible here by setting request == limit on memory.
        limits: Dict[str, str] = {"memory": self.tool_memory}
        requests: Dict[str, str] = {
            "memory": self.tool_memory,
            "cpu": self.tool_cpu,
        }
        if self.gpu_id is not None:
            limits["nvidia.com/gpu"] = "1"
        resources = k8s_client.V1ResourceRequirements(
            requests=requests,
            limits=limits,
        )

        main_container = k8s_client.V1Container(
            name="tool",
            image=image,
            image_pull_policy="IfNotPresent",
            command=["/bin/sh", "-c", wrapper],
            env=shared_env + landseer_env + task_env,
            env_from=env_from,
            volume_mounts=main_mounts,
            resources=resources,
        )

        pod_spec = k8s_client.V1PodSpec(
            restart_policy="Never",
            service_account_name=self.service_account,
            init_containers=[init_container],
            containers=[main_container],
            volumes=volumes,
            image_pull_secrets=(
                [k8s_client.V1LocalObjectReference(name=self.image_pull_secret)]
                if self.image_pull_secret
                else None
            ),
        )

        # active_deadline_seconds enforces our timeout server-side. 0 in the
        # CLI means "no timeout"; we then leave the field unset.
        active_deadline = self.timeout if self.timeout and self.timeout > 0 else None

        job = k8s_client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=k8s_client.V1ObjectMeta(
                name=job_name,
                namespace=self.namespace,
                labels={"app.kubernetes.io/managed-by": "landseer-worker"},
            ),
            spec=k8s_client.V1JobSpec(
                backoff_limit=0,
                ttl_seconds_after_finished=600,
                active_deadline_seconds=active_deadline,
                template=k8s_client.V1PodTemplateSpec(
                    metadata=k8s_client.V1ObjectMeta(
                        labels={
                            "app.kubernetes.io/managed-by": "landseer-worker",
                            "landseer.io/job": job_name,
                        }
                    ),
                    spec=pod_spec,
                ),
            ),
        )
        return job

    def _wait_for_job(self, job_name: str) -> int:
        """Block until the Job's Pod terminates; return the main container exit code.

        We poll instead of using watch() to keep the control flow simple and
        identical to the DockerRunner deadline pattern. Polls every 2s.
        """
        start = time.monotonic()
        while True:
            try:
                job = self._batch.read_namespaced_job(job_name, self.namespace)
            except ApiException as e:
                logger.warning(f"Job read failed for {job_name}: {e}")
                time.sleep(2.0)
                continue

            status = job.status or k8s_client.V1JobStatus()
            if status.failed and status.failed > 0:
                exit_code = self._fetch_main_exit_code(job_name)
                if exit_code is None:
                    # Pod was killed before terminated state was recorded
                    # (e.g. activeDeadlineSeconds). Use a non-zero sentinel.
                    exit_code = -1
                logger.info(f"Job {job_name} reported failed (exit_code={exit_code})")
                return exit_code
            if status.succeeded and status.succeeded > 0:
                exit_code = self._fetch_main_exit_code(job_name) or 0
                logger.info(f"Job {job_name} succeeded (exit_code={exit_code})")
                return exit_code

            elapsed = time.monotonic() - start
            if self.timeout and self.timeout > 0 and elapsed > self.timeout + 30:
                # active_deadline_seconds should have killed the job already.
                # If we are 30s past that and still nothing, give up locally.
                logger.error(
                    f"Job {job_name} did not finish within {self.timeout}s + 30s grace"
                )
                return -1
            time.sleep(2.0)

    def _fetch_main_exit_code(self, job_name: str) -> Optional[int]:
        """Find the Pod for a Job and return its 'tool' container exit code."""
        try:
            pods = self._core.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"landseer.io/job={job_name}",
            )
        except ApiException as e:
            logger.warning(f"Could not list pods for job {job_name}: {e}")
            return None

        for pod in pods.items:
            for cs in (pod.status.container_statuses or []):
                if cs.name != "tool":
                    continue
                terminated = getattr(cs.state, "terminated", None)
                if terminated and terminated.exit_code is not None:
                    return int(terminated.exit_code)
        return None

    def _stream_pod_logs(
        self,
        job_name: str,
        buffer: List[str],
        lock: threading.Lock,
        stop: threading.Event,
    ) -> None:
        """Best-effort log tail. Reconnects until the stop event is set.

        We wait for the Pod to exist and for the 'tool' container to have
        started before opening the log stream — otherwise the API returns
        ContainerCreating errors and we burn retries.
        """
        pod_name: Optional[str] = None
        deadline = time.monotonic() + 60.0
        while not stop.is_set() and pod_name is None and time.monotonic() < deadline:
            try:
                pods = self._core.list_namespaced_pod(
                    namespace=self.namespace,
                    label_selector=f"landseer.io/job={job_name}",
                )
                if pods.items:
                    pod = pods.items[0]
                    statuses = pod.status.container_statuses or []
                    tool_ready = any(
                        cs.name == "tool" and (cs.state.running or cs.state.terminated)
                        for cs in statuses
                    )
                    if tool_ready:
                        pod_name = pod.metadata.name
                        break
            except ApiException:
                pass
            time.sleep(1.0)

        if pod_name is None:
            logger.debug(f"Log stream giving up — no tool container started for {job_name}")
            return

        # Reconnect loop. Long-running tasks (training jobs that span hours)
        # frequently get their log stream dropped by the API server / load
        # balancer; without reconnect we silently lose tail logs after the
        # first drop. We exit the loop only when:
        #   - stop event is set (worker decided the Job is done), or
        #   - the Pod is gone (404 / 400 from the API), or
        #   - too many consecutive failures (cap to avoid burning CPU on a
        #     permanently-broken pod).
        consecutive_failures = 0
        while not stop.is_set():
            try:
                stream = self._core.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=self.namespace,
                    container="tool",
                    follow=True,
                    _preload_content=False,
                )
                for raw_line in stream.stream(amt=None, decode_content=True):
                    if stop.is_set():
                        return
                    if not raw_line:
                        continue
                    if isinstance(raw_line, bytes):
                        raw_line = raw_line.decode("utf-8", errors="replace")
                    with lock:
                        buffer.append(raw_line)
                # Stream ended cleanly — usually means the container has
                # exited. Reset failure counter; the outer while+stop check
                # handles termination.
                consecutive_failures = 0
            except ApiException as e:
                # Pod deleted while we were tailing. Nothing more to do.
                if e.status in (400, 404):
                    logger.debug(f"Log stream: pod {pod_name} gone ({e.status})")
                    return
                consecutive_failures += 1
                logger.debug(
                    f"Log stream API error for {pod_name} "
                    f"(attempt {consecutive_failures}): {e}"
                )
            except Exception as e:
                consecutive_failures += 1
                logger.debug(
                    f"Log stream error for {pod_name} "
                    f"(attempt {consecutive_failures}): {e}"
                )

            if consecutive_failures >= 5:
                logger.warning(
                    f"Log stream giving up after 5 consecutive failures for {pod_name}"
                )
                return
            # Wait briefly before reconnecting; abort early if stop fires.
            if stop.wait(2.0):
                return
