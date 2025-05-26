from enum import Enum
from typing import List, Optional, Dict, Annotated, Any, Self
from pydantic import BaseModel, model_validator, Field, field_validator, AnyUrl
import os
import logging
import requests
import importlib
import docker

logger = logging.getLogger("defense_pipeline")

DATASET_LOADER_FOLDER = os.path.abspath("src/dataset_handler/loaders")

class Dataset(BaseModel):
    name: str = Field(description="Name of the dataset")
    link: str #TODO: AnyUrl
    format: str
    sha1: str

    loader_module: Optional[Any] = None

    @model_validator(mode="after")
    def set_loader_module(self):
        loader_module = self.loader_module
        if loader_module is None:
            loader_module = f"dataset_handler.loaders.{self.name}_loader"
        else:
            if not os.path.exists(loader_module):
                raise ValueError(f"Loader module '{loader_module}' does not exist")
        try:
            module = importlib.import_module(loader_module)
            self.loader_module = module
        except Exception as e:
            raise ValueError(f"Failed to load dataset loader for '{self.name}': {e}")
        return self

    @field_validator("name", mode="before")
    def normalize_name(cls, v):
        print(f"Normalizing dataset name: {v}")
        if not v:
            raise ValueError("Dataset name cannot be empty")
        v = v.replace("_", "").replace("-", "").replace(" ", "").lower()
        return v

    @field_validator("name", mode="after")
    def check_name(cls, v):
        print(f"Validating dataset name: {v}")
        if not v:
            raise ValueError("Dataset name cannot be empty")
        dataset_preprocess_files = os.listdir(DATASET_LOADER_FOLDER)
        dataset_preprocess_files = [f.split("_")[0] for f in dataset_preprocess_files]
        if v not in dataset_preprocess_files:
            raise ValueError(f"Dataset '{v}' is not supported. Supported datasets: {dataset_preprocess_files}")
        return v
    
    @field_validator("link", mode="after")
    def check_link(cls, v):
        print(f"Validating dataset link: {v}")
        if not v:
            raise ValueError("Dataset link cannot be empty")
        v = str(v)
        if not v.startswith("http"):
            raise ValueError("Dataset link must start with http or https")
        print(f"Dataset link is valid: {v}")
        return v

class DockerConfig(BaseModel):
    image: Annotated[str, "TODO: Validate the link for image"] = Field(description="Docker image name")
    command: str = Field(description="Command to run the tool")
    config_script: Optional[str] = Field(default="configs/model/config_model.py", description="Path to the configuration script for the tool", validate_default=True)

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
        print(f"Validating Docker image: {self.image}")
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
        print(f"Docker image '{self.image}' is valid and available")
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
        # print(f"Using token: {token}")
    elif path.startswith("docker.io/") or "/" not in path:
        registry = "registry-1.docker.io"
        repo = path.replace("docker.io/", "") if path.startswith("docker.io/") else f"library/{path}"
        token_url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
        token = requests.get(token_url).json()["token"]
    else:
        raise ValueError(f"Unsupported registry in image: {image}")

    # print(f"1. Fetching manifest from {registry} for repo {repo} with tag {tag}")

    manifest_url = f"https://{registry}/v2/{repo}/manifests/{tag}"
    # print(f"1. Manifest URL: {manifest_url}")
    resp = requests.get(manifest_url, headers=headers)
    # print(f"2. Fetching manifest from {registry} for repo {repo} with tag {tag}")
    # print(f"2. Response status code: {resp.status_code}")
    manifest = resp.json()
    # print(f"3. Manifest: {manifest}")
    config_digest = manifest["config"]["digest"]

    config_url = f"https://{registry}/v2/{repo}/blobs/{config_digest}"
    resp = requests.get(config_url)
    # print(f" Fetching config from {registry} for repo {repo} with digest {config_digest}")
    resp.raise_for_status()
    config = resp.json()
    # print(f"Labels fetched from Docker image: {image}")
    return config.get("config", {}).get("Labels", {})

class ToolConfig(BaseModel):
    name: str = Field(description="Name of the tool")
    docker: DockerConfig = Field(description="Docker configuration for the tool")

    @property
    def tool_name(self) -> str:
        return self.name
    
    @property
    def tool_stage(self) -> str:
        return self.docker.get_labels.get("stage", "unknown")
    
    @property
    def tool_dataset(self) -> str:
        return self.docker.get_labels.get("dataset", "unknown")
    
    @property
    def tool_defense_type(self) -> str:
        return self.docker.get_labels.get("defense_type", "unknown")
    
class Stage(str, Enum):
    PRE_TRAINING = "pre_training"
    DURING_TRAINING = "during_training"
    POST_TRAINING = "post_training"

class StageConfig(BaseModel):
    tools: List[ToolConfig] = Field(default_factory=list, description="List of tools to be used in the stage")
    noop: Optional[ToolConfig] = Field(default=None, description="Noop tool for the stage")

class PipelineStructure(BaseModel):
    dataset: Dataset = Field(description="Dataset Info")
    pipeline: Dict[Stage, StageConfig]

    @field_validator("pipeline")
    def must_have_all_stages(cls, v):
        expected = {stage for stage in Stage}
        missing = expected - set(v.keys())
        if missing:
            raise ValueError(f"Missing pipeline stages: {missing}")
        return v
    
    @model_validator(mode="after")
    def validate_noop_for_necessary_stages(self):
        for stage, config in self.pipeline.items():
            if stage in {Stage.PRE_TRAINING, Stage.DURING_TRAINING} and config.noop is None:
                raise ValueError(f"Stage '{stage}' must have a noop tool")
            return self
      
    #@model_validator(mode="after")
    #def fetch_and_validate_labels(self):
    #    for stage in self.pipeline.keys():
    #        values = self.pipeline[stage].tools
    #        for tool in values:
    #            docker = tool.docker
    #            labels = docker.get_labels
    #            print(f"Labels: {labels}")
    #            label_stage = labels.get("stage")
    #            if stage and label_stage.lower() != stage:
    #                raise ValueError(f"Tool '{tool.tool_name}' is placed under stage '{stage}', "
    #                                 f"but its Docker label says '{label_stage}'")
    #            print(f"Tool {tool.tool_name}' is correctly placed under stage '{stage}'")
    #        return self