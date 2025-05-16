from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, model_validator, Field, field_validator
from tools import ToolConfig
from dataset_handler import Dataset

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
    #            try:
    #                labels = docker.get_labels
    #                print(f"Labels: {labels}")
    #                label_stage = labels.get("stage")
    #                if stage and label_stage.lower() != stage:
    #                    raise ValueError(f"Tool '{tool.tool_name}' is placed under stage '{stage}', "
    #                                     f"but its Docker label says '{label_stage}'")
    #                print(f"Tool '{tool.tool_name}' is correctly placed under stage '{stage}'")
    #            except Exception as e:
    #                raise ValueError(f"Failed to inspect Docker image '{docker.image}': {e}")
    #        return self
            