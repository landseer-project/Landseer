import os
import torch
from dataclasses import dataclass, field
from pathlib import Path
import logging
from landseer_pipeline.config.schemas import PipelineStructure, AttackSchema
from typing import Optional

logger = logging.getLogger(__name__)

# Global settings instance for accessing dry_run mode
_current_settings: Optional['Settings'] = None
_temp_dry_run: bool = False  # Temporary dry-run state before Settings is created

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
    dry_run: bool = False  # Simple dry-run mode flag
    log_level: str = "INFO"
    # Experimental content-addressable artifact cache (opt-in): when True uses new global artifact cache
    experimental_artifact_cache: bool = False
    # Global (cross-pipeline) artifact store root (content-addressable). Kept outside pipeline_id for reuse.
    artifact_store_root: str = "./artifact_store"
    # Dynamically added (post-init) docker resource knobs (can also be provided after construction)
    # docker_shm_size: e.g. '1g', '2g'; docker_mem_limit: e.g. '8g'. Declared here for introspection / IDEs.
    docker_shm_size: str = '1g'
    docker_mem_limit: str | None = None

    def __post_init__(self):        
        object.__setattr__(self, "device", "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "logs_dir", Path(self.logs_dir))
        object.__setattr__(self, "results_dir", Path(self.results_dir) / str(self.pipeline_id) / str(self.timestamp))
        object.__setattr__(self, "output_dir", Path(self.output_dir) / str(self.pipeline_id))

        # New centralized model script path
        model_script_path = self.config.model.script if self.config and self.config.model else None
        object.__setattr__(self, "config_model_path", model_script_path)
        object.__setattr__(self, "model_framework", self.config.model.framework if self.config and self.config.model else None)
        object.__setattr__(self, "model_params", self.config.model.params if self.config and self.config.model else {})
        object.__setattr__(self, "model_script_hash", self.config.model.content_hash if self.config and self.config.model else "unknown")
        
        self._create_directories()
        
        # Set this as the current global settings instance
        global _current_settings
        _current_settings = self
    
    def _create_directories(self):
        artifact_root_path = Path(self.artifact_store_root)
        directories = [self.data_dir, self.logs_dir, self.output_dir, self.results_dir, artifact_root_path]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


def is_dry_run() -> bool:
    """Check if we're currently in dry-run mode"""
    global _current_settings, _temp_dry_run
    if _current_settings:
        return _current_settings.dry_run
    else:
        # Before Settings is created, check temporary state
        return _temp_dry_run


def set_temp_dry_run(dry_run: bool) -> None:
    """Set temporary dry-run state before Settings is created"""
    global _temp_dry_run
    _temp_dry_run = dry_run


def get_current_settings() -> Optional['Settings']:
    """Get the current global settings instance"""
    global _current_settings
    return _current_settings

