#!/usr/bin/env python3
"""
ML Defense Pipeline - Entry Point
"""
import argparse
import os
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / ".." / "src"))  # Adjust the path to your src directory

import logging
from dataset_handler import DatasetManager
from tools import ToolRunner
from evaluator import ModelEvaluator
from docker_handler import DockerRunner
from pipeline import PipelineRunner
from config import Settings, validate_and_load_pipeline_config, validate_and_load_attack_config
from utils import hash_file, setup_logger

def main():
    parser = argparse.ArgumentParser(description="ML Defense Pipeline")
    parser.add_argument("--config", "-c", type=str, help="Path to configuration JSON for preconfigured mode")
    parser.add_argument("--attack_config", "-a", type=str, help="Path to attack configuration JSON for preconfigured mode")
    args = parser.parse_args()
    pipeline_config = args.config
    attack_config = args.attack_config

    settings = Settings()  
    pipeline_id = hash_file(pipeline_config)
    setup_logger(pipeline_id)
    config = validate_and_load_pipeline_config(pipeline_config)
    attacks = validate_and_load_attack_config(attack_config)
        
    dataset_manager = DatasetManager(settings)
    tool_runner = ToolRunner(settings)
    model_evaluator = ModelEvaluator(settings)
    pipeline = PipelineRunner(settings)

    dataset_manager.prepare_dataset()
    #go to pipeline handler and let it run the pipeline
    pipeline.run()  

if __name__ == "__main__":
    main()