from pydantic import BaseModel, Field, AnyUrl, validator, field_validator, model_validator
import os
from typing import Any, Dict, Optional
import importlib
import logging

logger = logging.getLogger("defense_pipeline")

DATASET_LOADER_FOLDER = os.path.abspath("dataset_handler/loaders")

class Dataset(BaseModel):
    name: str = Field(description="Name of the dataset")
    link: AnyUrl
    format: str
    sha1: str

    loader_module: Optional[Any] = None

    @model_validator(mode="after")
    def set_loader_module(cls, values):
        name = values.name.replace("_", "").replace("-", "").replace(" ", "").lower()
        # TODO: change this handcoded path to a dynamic one
        module_name = f"dataset_handler.loaders.{name}_loader"

        try:
            module = importlib.import_module(module_name)
            values.loader_module = module
        except Exception as e:
            raise ValueError(f"Failed to load dataset loader for '{name}': {e}")

        return values

    @field_validator("name", mode="before")
    def normalize_name(cls, v):
        print(f"Normalizing dataset name: {v}")
        if not v:
            raise ValueError("Dataset name cannot be empty")
        v = v.replace("_", "").replace("-", "").replace(" ", "").lower()
        return v

    @validator("name")
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
        return v