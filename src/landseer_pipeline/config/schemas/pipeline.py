from enum import Enum
from typing import List, Optional, Dict, Annotated, Any, Self
from pydantic import BaseModel, model_validator, Field, field_validator, AnyUrl
import os
import logging
import requests
import importlib
import docker
from landseer_pipeline.utils.docker import get_labels_from_image
import hashlib
import json

logger = logging.getLogger(__name__)

DATASET_LOADER_FOLDER = os.path.abspath("src/landseer_pipeline/dataset_handler/loaders")

class ModelConfig(BaseModel):
    """Top-level model configuration (centralized)."""
    script: str = Field(description="Path to the model config / construction script")
    framework: str = Field(default="pytorch", description="ML framework identifier")
    params: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters / architecture args")

    @field_validator("script", mode="after")
    def validate_script_exists(cls, v):
        v_abs = os.path.abspath(v)
        if not os.path.exists(v_abs):
            raise ValueError(f"Model script '{v_abs}' does not exist")
        if not v_abs.endswith('.py'):
            raise ValueError("Model script must be a Python file")
        return v_abs

    @property
    def content_hash(self) -> str:
        try:
            with open(self.script, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except Exception:
            return "unknown"

class Dataset(BaseModel):
    """Reduced dataset specification; provenance handled by loader metadata.
    Backward compatibility: accept legacy fields (link, format, sha1) but mark deprecated.
    """
    name: str = Field(description="Dataset name (used to resolve loader module)")
    version: Optional[str] = Field(default=None, description="Optional dataset version tag")
    variant: str = Field(default="clean", description="Variant label, e.g., clean or poisoned")
    params: Dict[str, Any] = Field(default_factory=dict, description="Loader parameters (subset_size, seed, poison_fraction, etc.)")
    loader_module: Optional[Any] = None

    @model_validator(mode="after")
    def set_loader_module(self):
        loader_module = self.loader_module
        if loader_module is None:
            loader_module = f"landseer_pipeline.dataset_handler.loaders.{self.name}_loader"
        else:
            if not os.path.exists(loader_module):
                logger.error(f"Loader module '{loader_module}' does not exist")
                raise ValueError(f"Dataset not supported: {self.name}. Please implement a loader for this dataset.")
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

class DockerConfig(BaseModel):
    image: Annotated[str, "TODO: Validate the link for image"] = Field(description="Docker image name")
    command: str = Field(description="Command to run the tool")
    config_script: Optional[str] = Field(default=None, description="(Deprecated) Per-tool model config script override", validate_default=True)

    @property
    def get_labels(self) -> Dict[str, str]:
        if self.image:
            labels = get_labels_from_image(self.image)
            if not labels:
                raise ValueError(f"No labels found in Docker image '{self.image}'")
            
            # Check for either defense_stage or stage labels (flexible validation)
            has_defense_stage = "org.opencontainers.image.defense_stage" in labels
            has_stage = "org.opencontainers.image.stage" in labels
            if not has_defense_stage and not has_stage:
                raise ValueError(f"Label 'defense_stage' or 'stage' not found in Docker image '{self.image}'")
            
            if "org.opencontainers.image.dataset" not in labels:
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
        if v is None:
            return v
        v_abs = os.path.abspath(v)
        if not os.path.exists(v_abs):
            raise ValueError(f"Config script '{v_abs}' does not exist")
        if not v_abs.endswith(".py"):
            raise ValueError(f"Config script '{v_abs}' must be a Python file")
        logger.warning("DockerConfig.config_script is deprecated; prefer top-level model.script")
        return v_abs

    
    @model_validator(mode="after")
    def validate_image_and_pull(self) -> Self:
        logger.debug(f"Validating Docker image: {self.image}")
        try:
            client = docker.from_env() 
            logger.warning(f"Attempting to pull docker image...")
            client.images.pull(self.image)
        except docker.errors.APIError as e:
            raise ValueError(f"Failed to check Docker image '{self.image}': {e}")
        logger.debug(f"Docker image '{self.image}' is valid and available")
        return self 

class AuxiliaryFile(BaseModel):
    """Configuration for auxiliary files needed by tools"""
    local_path: str = Field(description="Local path to the auxiliary file or directory")
    container_path: str = Field(description="Path where file will be mounted in container")
    required: bool = Field(default=False, description="Whether this auxiliary file is required for tool execution")
    description: Optional[str] = Field(default=None, description="Human-readable description of this auxiliary file")

class ToolConfig(BaseModel):
    name: str = Field(description="Name of the tool")
    docker: DockerConfig = Field(description="Docker configuration for the tool")
    output_path: Optional[str] = Field(default=None, description="Path to store the output of the tool")
    auxiliary_files: Optional[List[AuxiliaryFile]] = Field(default=None, description="Additional files/directories to mount in container")
    required_inputs: Optional[List[str]] = Field(default=None, description="Explicit artifact names required as inputs (optional)")

    @property
    def tool_name(self) -> str:
        return self.name
    
    @property
    def tool_stage(self) -> str:
        return self.docker.get_labels.get("defense_stage", "unknown")
    
    @property
    def tool_dataset(self) -> str:
        labels = self.docker.get_labels
        return labels.get("org.opencontainers.image.dataset", "unknown")
    
    @property
    def tool_defense_type(self) -> str:
        labels = self.docker.get_labels
        # Try defense_type first, then fall back to type
        defense_type = labels.get("org.opencontainers.image.defense_type")
        if defense_type:
            return defense_type
        return labels.get("org.opencontainers.image.type", "unknown")

    def set_output_path(self, output_path: str):
        self.output_path = output_path
        logger.debug(f"For tool '{self.name}', setting output path to: {self.output_path}")

    def has_output_path(self) -> bool:
        return self.output_path is not None and os.path.exists(self.output_path)
    
    def get_auxiliary_volume_mounts(self) -> Dict[str, Dict[str, str]]:
        """Generate Docker volume mounts for auxiliary files"""
        volumes = {}
        if self.auxiliary_files:
            for aux_file in self.auxiliary_files:
                if os.path.exists(aux_file.local_path):
                    volumes[os.path.abspath(aux_file.local_path)] = {
                        "bind": aux_file.container_path,
                        "mode": "ro"
                    }
                elif aux_file.required:
                    raise FileNotFoundError(f"Required auxiliary file not found: {aux_file.local_path}")
                else:
                    logger.warning(f"Optional auxiliary file not found: {aux_file.local_path}")
        return volumes
    
    def has_auxiliary_files(self) -> bool:
        """Check if tool has auxiliary files configured"""
        return self.auxiliary_files is not None and len(self.auxiliary_files) > 0
    
class Stage(str, Enum):
    PRE_TRAINING = "pre_training"
    DURING_TRAINING = "during_training"
    POST_TRAINING = "post_training"
    DEPLOYMENT = "deployment"

class DefenseType(str, Enum):
    ADVERSARIAL = "adversarial"
    OUTLIER = "outlier"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    WATERMARKING = "watermarking"
    UNKNOWN = "unknown"

class StageConfig(BaseModel):
    tools: List[ToolConfig] = Field(default_factory=list, description="List of tools to be used in the stage")
    noop: Optional[ToolConfig] = Field(default=None, description="Noop tool for the stage")

class PipelineStructure(BaseModel):
    dataset: Dataset = Field(description="Dataset Info")
    model: ModelConfig = Field(description="Central model configuration")
    pipeline: Optional[Dict[Stage, StageConfig]]

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
            if stage in {Stage.DURING_TRAINING} and config.noop is None:
                raise ValueError(f"Stage '{stage}' must have a noop tool")
        return self

    @model_validator(mode="after")
    def fetch_and_validate_labels(self):
        # Accept multiple synonyms for stage labels in docker images
        stage_synonyms = {
            Stage.PRE_TRAINING.value: {"pre_training", "pre", "pretrain", "pre_defense"},
            Stage.DURING_TRAINING.value: {"during_training", "during", "in", "train", "training", "in_training", "in_defense", "during_defense"},
            Stage.POST_TRAINING.value: {"post_training", "post", "after", "posttrain", "post_defense"},
            Stage.DEPLOYMENT.value: {"deployment", "deploy", "inference", "deploy_defense"},
        }
        for stage_enum, stage_cfg in self.pipeline.items():
            normalized_stage = stage_enum.value  # e.g. 'post_training'
            tools = stage_cfg.tools
            for tool in tools:
                labels = tool.docker.get_labels
                label_stage = labels.get("org.opencontainers.image.stage") or labels.get("org.opencontainers.image.defense_stage")
                if label_stage:
                    ls_norm = label_stage.strip().lower()
                    allowed = stage_synonyms.get(normalized_stage, {normalized_stage})
                    if ls_norm not in allowed:
                        raise ValueError(
                            f"Tool '{tool.tool_name}' is placed under stage '{normalized_stage}', "
                            f"but its Docker label says '{label_stage}'. Allowed synonyms: {sorted(allowed)}"
                        )
                logger.debug(f"Tool '{tool.tool_name}' labels validated for stage '{normalized_stage}'")
        return self