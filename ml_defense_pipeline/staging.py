"""
Main pipeline orchestration for ML Defense Pipeline
"""
import logging
import os
from typing import List, Dict
import datetime
import hashlib
import yaml
import torch
import itertools

from docker_handler import DockerRunner
from dataset_handler import DatasetManager
from tools import ToolRunner
from evaluator import ModelEvaluator, AttackSchema
from logging_manager import LoggingManager
from pipeline import Stage, PipelineStructure

logger = logging.getLogger("defense_pipeline")

class Stager:

    def __init__(self, config_path: str, attack_config_path: str=None):
        self.config_path = config_path
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        if attack_config_path is not None:
            self.attack_config_path = attack_config_path
            if not os.path.exists(attack_config_path):
                raise FileNotFoundError(f"Attack configuration file {attack_config_path} not found.")
        self.pipeline_id = hash_file(self.config_path)
        self._setup_logger()
        self.config = self._validate_and_load_pipeline_config()
        self.attack_config = self._validate_and_load_attack_config()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.combinations = self.make_combinations()
        
        self.docker_manager = DockerRunner(self)
        self.dataset_manager = DatasetManager(self)
        self.tool_runner = ToolRunner(self)
        self.model_evaluator = ModelEvaluator(self)

    def run(self):
        """Execute the pipeline based on the configuration"""
        dataset_name = self.config.dataset.name
        # list all tools in the pipeline
        # TODO make this a propery of dataset
        #logger.info(f"Pipeline tools: {self.config_manager.get_all_tools()}")
        #if "fineprune" in self.config_manager.get_all_tools():
        #    logger.info("Fineprune tool found in the pipeline.")
        #    add_backdoor_trigger = True
        dataset_dir = self.dataset_manager.prepare_dataset(self.config.dataset)
        logger.info(
            f"Using dataset '{dataset_name}' in directory: {dataset_dir}")
        for combination in self.combinations:
            logger.info(f"---------Running combination: {combination}---------")
            self.run_combination(combination, dataset_dir)
            logger.info(f"Completed combination: {combination}")
            logger.info(f"Pipeline completed successfully for combination: {combination}")
        logger.info("All combinations completed successfully.")    
    
    def _validate_and_load_pipeline_config(self):
        try:
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)
                config = PipelineStructure.model_validate(data)
                logger.info("Configuration validation passed.") 
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML configuration file: {e}")
        return config
    
    def _validate_and_load_attack_config(self):
        try:
            with open(self.attack_config_path, "r") as f:
                data = yaml.safe_load(f)
                config = AttackSchema.model_validate(data)
                logger.info("Attack configuration validation passed.")
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML attack configuration file: {e}")
        return config
    
    def _setup_logger(self):
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        log_file_id = f"pipeline_{self.pipeline_hash}_{timestamp}"
        LoggingManager.setup_logging(log_file_id)

    def make_combinations(self):
        """Create combinations of tools based on the configuration with noop"""
        pipeline = self.config.pipeline
        options_per_stage = []
        stages = [stage for stage in Stage]
        for stage in stages:
            stage_config = pipeline.get(stage, {})
            tools = stage_config.tools
            noop = stage_config.noop
            stage_options = [noop] + tools
            options_per_stage.append(stage_options)
        all_combinations = list(itertools.product(*options_per_stage))
        combinations = {}
        for idx, combo in enumerate(all_combinations):
            key = f"comb_{idx:03d}"
            combinations[key] = dict(zip(stages, combo))
        logger.info(f"Generated {len(combinations)} combinations.")
        return combinations
    
    def store_results(self, combination: str, dataset_name: str, dataset_dir: str, final_model_path: str, final_acc: float):
        """Store the results of the pipeline run"""
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, "pipeline_{pipeline_hash}.csv")
        if not os.path.exists(results_file):
            with open(results_file, "w") as f:
                f.write("combination,dataset_name,dataset_dir,final_model_path,final_acc\n")
        with open(results_file, "a") as f:
            f.write(f"{combination},{dataset_name},{dataset_dir},{final_model_path},{final_acc}\n")

def hash_file(path, bits=64):
    hasher = hashlib.blake2s(digest_size=bits // 8)
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

