"""
Configuration management for ML Defense Pipeline
"""
import json
import logging
import torch
import os
from typing import Dict, List, Optional, Any
import itertools

logger = logging.getLogger("defense_pipeline")

class PipelineConfig:
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or interactive input"""
        self.config = {}
        self.tools_db = {}
        self.combinations = dict()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        if config_path:
            self._load_tools(config_path)
        else:
            raise ValueError("No configuration file provided. Please specify a config file path.")
    
    def _load_tools(self, config_path: str):
        """Load configuration from a JSON file"""
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
            if not self.config:
                raise ValueError("Configuration file is empty.")
            # TODO: now verify the config file structure
            # config_sanity_check(self.config)
            # TODO: Make various combinations of tools
            self.make_combinations()

            # self.combinations = combinations.make_combinations(self.config)     
            logger.info(f"Loaded configuration from {config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found.")
            raise
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise

    def make_combinations(self):
        """Create combinations of tools based on the configuration with noop"""
        pipeline = self.config.get("pipeline", {})
        stages = ["pre_training", "during_training", "post_training"]
        options_per_stage = []
        for stage in stages:
            stage_config = pipeline.get(stage, {})
            tools = stage_config.get("tools", [])
            noop = stage_config.get("noop", {})
            stage_options = [noop] + tools
            options_per_stage.append(stage_options)
        all_combinations = list(itertools.product(*options_per_stage))
        self.combinations = {}
        for idx, combo in enumerate(all_combinations):
            key = f"comb_{idx:03d}"
            self.combinations[key] = dict(zip(stages, combo))
        print(f"generated combinations: {self.combinations}")
        logger.info(f"Generated {len(self.combinations)} combinations.")

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about all selected datasets"""
        # TODO: from sanity check get the dataset_name, for now just use the first one
        dataset_config = self.config.get("dataset", {})
        dataset_name = next(iter(dataset_config), None)
        dataset_info = dataset_config.get(dataset_name, {})
        try:
            print(f"{dataset_name} info (frozen):", frozenset(dataset_info.items()))
        except TypeError:
            print(f"{dataset_name} info contains unhashable values.")    
        return {dataset_name: dataset_info}

    def get_tools_for_stage(self, stage: str) -> List[Dict]:
        """Get the list of tools for a specific stage"""
        return self.config.get("pipeline",[]).get(stage, []).get("tools", [])
    
    def get_tools_for_combination(self, combination: str, stage: str) -> List[Dict]:
        """Get the list of tools for a specific combination and stage"""
        if combination not in self.combinations:
            raise ValueError(f"Combination '{combination}' not found.")
        tools = self.combinations[combination].get(stage, {})
        return [tools] if tools else []
