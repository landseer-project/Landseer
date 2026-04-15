"""
Resolve OCI config labels for container images (pipeline stage validation).

Uses ``docker image inspect`` when available, then registry HTTP (GHCR / Docker Hub)
via httpx, aligned with the legacy landseer label fetcher.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _labels_via_docker_inspect(image: str, timeout_s: float = 30.0) -> Optional[Dict[str, str]]:
    try:
        proc = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{json .Config.Labels}}",
                image,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            logger.debug("docker inspect failed for %s: %s", image, proc.stderr.strip())
            return None
        raw = proc.stdout.strip()
        if not raw or raw == "null":
            return {}
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) if v is not None else "" for k, v in data.items()}
    except FileNotFoundError:
        logger.debug("docker CLI not found; skipping local inspect for %s", image)
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        logger.debug("docker inspect error for %s: %s", image, e)
        return None


def _labels_via_registry_http(image: str) -> Dict[str, str]:
    import httpx

    if ":" in image:
        path, tag = image.rsplit(":", 1)
    else:
        path, tag = image, "latest"

    if path.startswith("ghcr.io/"):
        registry = "ghcr.io"
        repo = path[len("ghcr.io/") :]
        token = os.getenv("GHCR_TOKEN")
        if not token:
            token_resp = httpx.get(
                f"https://ghcr.io/token?scope=repository:{repo}:pull&service=ghcr.io",
                timeout=30.0,
            )
            token_resp.raise_for_status()
            token = token_resp.json().get("token")
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": (
                "application/vnd.oci.image.manifest.v1+json, "
                "application/vnd.docker.distribution.manifest.v2+json"
            ),
        }
    elif path.startswith("docker.io/") or "/" not in path:
        registry = "registry-1.docker.io"
        repo = path.replace("docker.io/", "") if path.startswith("docker.io/") else f"library/{path}"
        token_url = (
            f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
        )
        token_resp = httpx.get(token_url, timeout=30.0)
        token_resp.raise_for_status()
        token = token_resp.json()["token"]
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": (
                "application/vnd.oci.image.manifest.v1+json, "
                "application/vnd.docker.distribution.manifest.v2+json"
            ),
        }
    else:
        raise ValueError(f"Unsupported registry for remote label fetch: {image}")

    manifest_url = f"https://{registry}/v2/{repo}/manifests/{tag}"
    with httpx.Client(timeout=60.0) as client:
        manifest_resp = client.get(manifest_url, headers=headers)
        manifest_resp.raise_for_status()
        manifest: Dict[str, Any] = manifest_resp.json()

    if "config" not in manifest:
        raise ValueError("Image manifest does not contain 'config'.")

    config_digest = manifest["config"]["digest"]
    config_url = f"https://{registry}/v2/{repo}/blobs/{config_digest}"
    with httpx.Client(timeout=60.0) as client:
        config_resp = client.get(config_url, headers={"Authorization": f"Bearer {token}"})
        config_resp.raise_for_status()
        cfg = config_resp.json()

    labels = cfg.get("config", {}).get("Labels") or {}
    if not isinstance(labels, dict):
        return {}
    return {str(k): str(v) if v is not None else "" for k, v in labels.items()}


def get_container_labels_for_image(image: str, runtime: Optional[str] = None) -> Dict[str, str]:
    """
    Best-effort labels for ``image``.

    Skips docker inspect for apptainer/singularity runtimes. Tries local docker
    inspect, then GHCR / Docker Hub registry config blob.
    """
    if runtime and str(runtime).lower() in {"apptainer", "singularity"}:
        logger.debug("Skipping docker inspect for apptainer/singularity image %s", image)
    else:
        via_docker = _labels_via_docker_inspect(image)
        if via_docker is not None:
            return via_docker

    try:
        return _labels_via_registry_http(image)
    except Exception as e:
        logger.warning("Could not fetch labels for %s: %s", image, e)
        return {}
