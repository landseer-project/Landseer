from docker_handler import DockerConfig
from pydantic import BaseModel, validator, AnyUrl, model_validator, Field, field_validator
import logging 

logger = logging.getLogger("defense_pipeline")

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