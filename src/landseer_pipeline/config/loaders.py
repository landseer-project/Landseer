import os
import yaml
import logging
from landseer_pipeline.config.schemas import PipelineStructure, AttackSchema

logger = logging.getLogger(__name__)

def validate_and_load_pipeline_config(config_path: str):
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            config = PipelineStructure.model_validate(data)
            logger.info("Configuration validation passed.") 
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration file: {e}")
    return config

def validate_and_load_attack_config(config_path: str):
    try:
        if config_path is not None:
            config_path = config_path
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Attack configuration file {config_path} not found.")
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            config = AttackSchema.model_validate(data)
            logger.info("Attack configuration validation passed.")
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML attack configuration file: {e}")
    return config