import requests
import logging
import os
from typing import Dict
from requests.exceptions import HTTPError

logger = logging.getLogger(__name__)

def get_labels_from_image(image: str) -> Dict[str, str]:
    logger.debug(f"Fetching labels from Docker image: {image}")

    if ":" in image:
        path, tag = image.rsplit(":", 1)
    else:
        path, tag = image, "latest"

    if path.startswith("ghcr.io/"):
        registry = "ghcr.io"
        repo = path[len("ghcr.io/"):]
        token = os.getenv("GHCR_TOKEN")

        if not token:
            # Try to fetch public token if not set
            logger.info(f"No GHCR_TOKEN found. Attempting to fetch public token for {repo}")
            token_resp = requests.get(
                f"https://ghcr.io/token?scope=repository:{repo}:pull&service=ghcr.io"
            )
            if token_resp.status_code != 200:
                logger.error(f"Failed to fetch GHCR public token: {token_resp.text}")
                raise ValueError("GHCR access token missing and could not fetch public token.")
            token = token_resp.json().get("token")

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.docker.distribution.manifest.v2+json"
        }

    elif path.startswith("docker.io/") or "/" not in path:
        registry = "registry-1.docker.io"
        repo = path.replace("docker.io/", "") if path.startswith("docker.io/") else f"library/{path}"
        token_url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
        token_resp = requests.get(token_url)
        token = token_resp.json()["token"]
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.docker.distribution.manifest.v2+json"
        }

    else:
        raise ValueError(f"Unsupported registry in image: {image}")

    manifest_url = f"https://{registry}/v2/{repo}/manifests/{tag}"
    manifest_resp = requests.get(manifest_url, headers=headers)
    if manifest_resp.status_code != 200:
        logger.error(f"Manifest fetch failed: {manifest_resp.status_code}, {manifest_resp.text}")
        raise ValueError("Failed to fetch image manifest.")

    manifest = manifest_resp.json()
    if "config" not in manifest:
        logger.error("Manifest did not contain 'config'. Possibly non-standard schema.")
        raise ValueError("Image manifest does not contain 'config'.")

    config_digest = manifest["config"]["digest"]
    config_url = f"https://{registry}/v2/{repo}/blobs/{config_digest}"
    config_resp = requests.get(config_url, headers={"Authorization": f"Bearer {token}"})
    config_resp.raise_for_status()
    config = config_resp.json()

    labels = config.get("config", {}).get("Labels", {})
    if labels:
        logger.debug(f"Labels fetched from image {image}: {labels}")
    else:
        logger.warning(f"No labels found on image {image}")
    return labels

def get_image_digest(image: str) -> str:
    if ":" in image:
        path, tag = image.rsplit(":", 1)
    else:
        path, tag = image, "latest"

    if path.startswith("ghcr.io/"):
        registry = "ghcr.io"
        repo = path[len("ghcr.io/"):]
        token = os.getenv("GHCR_TOKEN")
        if not token:
            token_resp = requests.get(
                f"https://ghcr.io/token?scope=repository:{repo}:pull&service=ghcr.io"
            )
            token = token_resp.json().get("token")
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.docker.distribution.manifest.v2+json"
        }
    else:
        registry = "registry-1.docker.io"
        repo = path if "/" in path else f"library/{path}"
        token_resp = requests.get(
            f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
        )
        token = token_resp.json()["token"]
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.docker.distribution.manifest.v2+json"
        }

    manifest_url = f"https://{registry}/v2/{repo}/manifests/{tag}"
    resp = requests.get(manifest_url, headers=headers)
    resp.raise_for_status()

    # Docker registry returns digest in headers for v2 manifests
    digest = resp.headers.get("Docker-Content-Digest")
    if not digest:
        raise ValueError(f"Could not get digest for image {image}")
    return digest
