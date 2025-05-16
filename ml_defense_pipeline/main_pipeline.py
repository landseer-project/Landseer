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
from model_evaluator import ModelEvaluator
from logging_manager import LoggingManager
from pipeline import Stage, PipelineStructure

logger = logging.getLogger("defense_pipeline")

class DefensePipeline:

    def __init__(self, config_path: str):
        self.config_path = config_path
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        self.pipeline_hash = hash_file(self.config_path)
        self._setup_logger()
        self.config = self._validate_and_load_config()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.combinations = self.make_combinations()
        print(f"Combinations: {self.combinations}")
        
        self.docker_manager = DockerRunner(self)
        self.utils = PipelineUtils(self.combinations)
        self.dataset_manager = DatasetManager(self)
        self.tool_runner = ToolRunner(self)
        self.model_evaluator = ModelEvaluator(self)

    def run(self):
        """Execute the pipeline based on the configuration"""
        dataset_name = self.config.dataset.name
        # list all tools in the pipeline
        # TODO make this a propery of dataset
        add_backdoor_trigger = False
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
    
    def run_combination(self, combination, dataset_dir):
        """Run a specific combination of tools"""
        current_input = dataset_dir
        stages = [stage.value for stage in Stage]
        for stage in stages:
            tools = self.utils.get_tools_for_combination(combination, stage)
            if not tools:
                logger.info(
                    f"No tools configured for stage '{stage}'. Skipping.")
                continue
            logger.info(f"Starting stage '{stage}' with {len(tools)} tool(s)")
            for tool in tools:
                print(f"Tool: ", tool)
                logger.info(f"Running tool '{tool.name}'...")
                if tool.name == "noop" and stage == "post_training":
                    logger.info(
                        f"[-] Skipping '{tool.name}' in stage '{stage}'.")
                    continue
                output_path = self.tool_runner.run_tool(
                        tool=tool,
                        stage=stage,
                        dataset_dir=dataset_dir,
                        input_path=current_input
                    )
                print(f"Tool '{tool.name}' output path: {output_path}")

                current_input = output_path
                if stage == "pre_training":
                    dataset_dir = current_input
                    #import ipdb
                    #ipdb.set_trace()
                    print("Dataset directory:", dataset_dir)
                    logger.info(
                        f"Updated dataset directory: {dataset_dir}")
                    # exit(0)
                logger.info(
                        f"Tool '{tool.name}' completed successfully.")
            logger.info(f"Completed stage '{stage}'")
        final_model_path = current_input
        final_dataset_path = os.path.join(dataset_dir)
        # baseline_model_path = os.path.join(dataset_dir, "baseline_model.pt")
        # if not os.path.exists(baseline_model_path):
        #    logger.info("Training baseline model for comparison...")
        #    baseline_acc = self.model_evaluator.train_baseline_model(
        #        final_dataset_path, baseline_model_path, device=self.config_manager.device)
        # else:
        # logger.info("Using existing baseline model for comparison...")
        # baseline_acc = self.model_evaluator.evaluate_clean(baseline_model_path, final_dataset_path, device=self.config_manager.device)

        logger.info("Evaluating final model...")
        final_acc = self.model_evaluator.evaluate_model(
            f"{final_model_path}/model.pt", final_dataset_path)

        logger.info("PIPELINE EVALUATION RESULTS")
        logger.info(f"Accuracy: {final_acc}")

        print(f"\nPipeline completed successfully!")
        print(f"Accuracy: {final_acc}")
    
    def _validate_and_load_config(self):
        try:
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)
                config = PipelineStructure.model_validate(data)
                logger.info("Configuration validation passed.") 
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML configuration file: {e}")
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
        # print(f"generated combinations: {self.combinations}")
        logger.info(f"Generated {len(combinations)} combinations.")
        return combinations

def hash_file(path, bits=64):
    hasher = hashlib.blake2s(digest_size=bits // 8)
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

class PipelineUtils(DefensePipeline):

    def __init__(self, combinations: str):
        self.combinations = combinations
        """Initialize configuration from file or interactive input"""
        
    def get_tools_for_stage(self, stage: str) -> List[Dict]:
        """Get the list of tools for a specific stage"""
        return self.config.get("pipeline",[]).get(stage, []).get("tools", [])
    
    def get_tools_for_combination(self, combination: str, stage: str) -> List[Dict]:
        """Get the list of tools for a specific combination and stage"""
        if combination not in self.combinations:
            raise ValueError(f"Combination '{combination}' not found.")
        tools = self.combinations[combination].get(stage, {})
        return [tools] if tools else []
