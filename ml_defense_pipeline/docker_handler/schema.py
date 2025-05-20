
from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Dict, Optional, Annotated, Self
import logging
import os
import requests


logger = logging.getLogger("defense_pipeline")

try:
    import docker
    DOCKER_SDK_AVAILABLE = True
except ImportError:
    DOCKER_SDK_AVAILABLE = False
    logger.warning(
        "Docker SDK not available. Falling back to subprocess for Docker operations.")


class DockerConfig(BaseModel):
    image: Annotated[str, "TODO: Validate the link for image"] = Field(description="Docker image name")
    command: str = Field(description="Command to run the tool")
    config_script: Optional[str] = Field(default="config_model.py", description="Path to the configuration script for the tool", validate_default=True)

    @property
    def get_labels(self) -> Dict[str, str]:
        if self.image:
            labels = get_labels_from_image(self.image)
            if not labels:
                raise ValueError(f"No labels found in Docker image '{self.image}'")
            if "stage" not in labels:
                raise ValueError(f"Label 'stage' not found in Docker image '{self.image}'")
            if "dataset" not in labels:
                raise ValueError(f"Label 'dataset' not found in Docker image '{self.image}'")
            return labels
        return {}
    
    @property
    def image_name(self) -> str:
        if self.image:
            image = self.image.split(":")[0]
            image_name = image.split("/")[-1]
            return image_name
        return ""
    
    @field_validator("config_script", mode="after")
    def check_config_script(cls, v):
        v = os.path.abspath(v)
        if not os.path.exists(v):
            raise ValueError(f"Config script '{v}' does not exist")
        if not v.endswith(".py"):
            raise ValueError(f"Config script '{v}' must be a Python file")
        return v

    
    @model_validator(mode="after")
    def validate_image_and_pull(self) -> Self:
        image_name = self.image.split(":")[0]
        try:
            client = docker.from_env()
            client.images.get(self.image)
        except docker.errors.ImageNotFound:
            try:
                client.images.pull(self.image)
            except Exception as e:
                raise ValueError(f"Failed to pull Docker image '{self.image}': {e}")
        except docker.errors.APIError as e:
            raise ValueError(f"Failed to check Docker image '{self.image}': {e}")
        return self
    
def get_labels_from_image(image: str) -> Dict[str, str]:
    print(f"Fetching labels from Docker image: {image}")
    if ":" in image:
        path, tag = image.rsplit(":", 1)
    else:
        path, tag = image, "latest"

    if path.startswith("ghcr.io/"):
        registry = "ghcr.io"
        repo = path[len("ghcr.io/"):]
        token = os.getenv("GHCR_TOKEN")

        if not token:
            raise ValueError("GHCR_TOKEN environment variable is not set. Please set it to access GitHub Container Registry.")
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.oci.image.manifest.v1+json"}
        print(f"Using token: {token}")
    elif path.startswith("docker.io/") or "/" not in path:
        registry = "registry-1.docker.io"
        repo = path.replace("docker.io/", "") if path.startswith("docker.io/") else f"library/{path}"
        token_url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
        token = requests.get(token_url).json()["token"]
    else:
        raise ValueError(f"Unsupported registry in image: {image}")

    print(f"1. Fetching manifest from {registry} for repo {repo} with tag {tag}")

    manifest_url = f"https://{registry}/v2/{repo}/manifests/{tag}"
    print(f"1. Manifest URL: {manifest_url}")
    resp = requests.get(manifest_url, headers=headers)
    print(f"2. Fetching manifest from {registry} for repo {repo} with tag {tag}")
    print(f"2. Response status code: {resp.status_code}")
    manifest = resp.json()
    print(f"3. Manifest: {manifest}")
    config_digest = manifest["config"]["digest"]

    config_url = f"https://{registry}/v2/{repo}/blobs/{config_digest}"
    resp = requests.get(config_url)
    print(f" Fetching config from {registry} for repo {repo} with digest {config_digest}")
    resp.raise_for_status()
    config = resp.json()
    print(f"Labels fetched from Docker image: {image}")
    return config.get("config", {}).get("Labels", {})