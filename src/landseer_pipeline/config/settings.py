import os
import torch
from dataclasses import dataclass, field
from pathlib import Path
import logging
from landseer_pipeline.config.schemas import PipelineStructure, AttackSchema

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Settings():
    config: PipelineStructure
    attacks: AttackSchema
    pipeline_id: str 
    data_dir: str = "./data"
    logs_dir: str = "./logs" 
    output_dir: str = "./cache"
    results_dir: str = "./results"
    timestamp: str = field(default_factory=lambda: str(int(torch.timestamp() * 1000)))
    use_gpu: bool = True
    use_cache : bool = True
    log_level: str = "INFO"

    def __post_init__(self):        
        object.__setattr__(self, "device", "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "logs_dir", Path(self.logs_dir))
        object.__setattr__(self, "results_dir", Path(self.results_dir) / str(self.pipeline_id) / str(self.timestamp))
        object.__setattr__(self, "output_dir", Path(self.output_dir) / str(self.pipeline_id))

        config_model_path = (
        self.config.pipeline.get("during_training").noop.docker.config_script
        if self.config.pipeline.get("during_training") and self.config.pipeline.get("during_training").noop
        else None
        )
        object.__setattr__(self, "config_model_path", config_model_path)
      
        self._create_directories()
    
    def _create_directories(self):
        directories = [self.data_dir, self.logs_dir, self.output_dir, self.results_dir]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    