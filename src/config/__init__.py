import yaml
import os
import logging
from config.settings import Settings
from config.schemas import PipelineStructure, AttackSchema, ToolConfig

logger = logging.getLogger("defense_pipeline")

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
        if attack_config_path is not None:
            attack_config_path = attack_config_path
            if not os.path.exists(attack_config_path):
                raise FileNotFoundError(f"Attack configuration file {attack_config_path} not found.")
        with open(attack_config_path, "r") as f:
            data = yaml.safe_load(f)
            config = AttackSchema.model_validate(data)
            logger.info("Attack configuration validation passed.")
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML attack configuration file: {e}")
    return config