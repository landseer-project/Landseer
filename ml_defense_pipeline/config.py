"""
Configuration management for ML Defense Pipeline
"""
import json
import logging
import os
from typing import Dict, List, Optional, Any

logger = logging.getLogger("defense_pipeline")

class PipelineConfig:
    DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs/default_config.json")
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or interactive input"""
        self.config = {}
        self.tools_db = {}
        self._load_tools_db()
        
        if config_path:
            self._load_from_file(config_path)
        else:
            self._interactive_setup()
    
    def _load_tools_db(self):
        """Load the database of available tools"""
        try:
            with open(self.DEFAULT_CONFIG_PATH, "r") as f:
                self.tools_db = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Default config file {self.DEFAULT_CONFIG_PATH} not found.")
            self.tools_db = {"pipeline": {"pre_training": [], "during_training": [], "post_training": []}}
    
    def _load_from_file(self, config_path: str):
        """Load configuration from a JSON file"""
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    def _interactive_setup(self):
        """Build configuration through interactive user prompts"""
        return

    from typing import Dict, Any

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about all selected datasets"""
        selected_datasets = self.config.get("dataset", {})
        print("Selected datasets:", list(selected_datasets.keys()))
    
        all_info = {}
    
        for dataset_name in selected_datasets:
            dataset_info = self.tools_db.get("dataset", {}).get(dataset_name, {})
            all_info[dataset_name] = dataset_info
    
            # Optional: Safe debug print
            try:
                print(f"{dataset_name} info (frozen):", frozenset(dataset_info.items()))
            except TypeError:
                print(f"{dataset_name} info contains unhashable values.")
    
        return all_info

    def get_tools_for_stage(self, stage: str) -> List[Dict]:
        """Get the list of tools for a specific stage"""
        return self.config.get("pipeline",[]).get(stage, [])