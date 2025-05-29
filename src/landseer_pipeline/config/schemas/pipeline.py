from enum import Enum
from typing import List, Optional, Dict, Annotated, Any, Self
from pydantic import BaseModel, model_validator, Field, field_validator, AnyUrl
import os
import logging
import requests
import importlib
import docker
from landseer_pipeline.utils.docker import get_labels_from_image

logger = logging.getLogger(__name__)

DATASET_LOADER_FOLDER = os.path.abspath("src/landseer_pipeline/dataset_handler/loaders")

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
            loader_module = f"landseer_pipeline.dataset_handler.loaders.{self.name}_loader"
        else:
            if not os.path.exists(loader_module):
                # we don't support this dataset yet so exit
                logger.error(f"Loader module '{loader_module}' does not exist")
                raise ValueError(f"Dataset not supported: {self.name}. Please implement a loader for this dataset.")
            # we don't support this dataset yet so exit

        try:
            module = importlib.import_module(loader_module)
            self.loader_module = module
        except Exception as e:
            raise ValueError(f"Failed to load dataset loader for '{self.name}': {e}")
        return self

    @field_validator("name", mode="before")
    def normalize_name(cls, v):
        logger.debug(f"Normalizing dataset name: {v}")
        if not v:
            raise ValueError("Dataset name cannot be empty")
        v = v.replace("_", "").replace("-", "").replace(" ", "").lower()
        return v

    @field_validator("name", mode="after")
    def check_name(cls, v):
        logger.debug(f"Validating dataset name: {v}")
        if not v:
            raise ValueError("Dataset name cannot be empty")
        dataset_preprocess_files = os.listdir(DATASET_LOADER_FOLDER)
        dataset_preprocess_files = [f.split("_")[0] for f in dataset_preprocess_files]
        if v not in dataset_preprocess_files:
            raise ValueError(f"Dataset '{v}' is not supported. Supported datasets: {dataset_preprocess_files}")
        return v
    
    @field_validator("link", mode="after")
    def check_link(cls, v):
        logger.debug(f"Validating dataset link: {v}")
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
        logger.debug(f"Validating Docker image: {self.image}")
        image_name = self.image.split(":")[0]
        try:
            client = docker.from_env()
            client.images.get(self.image)
        except docker.errors.ImageNotFound:
            logger.warning(f"Docker image '{self.image}' not found locally. Attempting to pull...")
            client.images.pull(self.image)
        except docker.errors.APIError as e:
            raise ValueError(f"Failed to check Docker image '{self.image}': {e}")
        logger.debug(f"Docker image '{self.image}' is valid and available")
        return self 

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