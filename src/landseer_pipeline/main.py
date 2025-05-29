#!/usr/bin/env python3
"""
ML Defense Pipeline - Entry Point
"""
import argparse
import os
import sys
import torch
from pathlib import Path
import datetime
sys.path.insert(0, str(Path(__file__).parent / ".." / "src"))  # Adjust the path to your src directory

import logging
from .dataset_handler import DatasetManager
from .pipeline import PipelineExecutor
from .config import Settings, validate_and_load_pipeline_config, validate_and_load_attack_config
from .utils import hash_file, setup_logger

def parse_arguments():
    parser = argparse.ArgumentParser(description='ML Defense Pipeline')
    parser.add_argument(
        '--config', '-c',
        required=True,
        type=Path,
        help='Path to pipeline configuration YAML file'
    )

    parser.add_argument(
        '--attack-config', '-a',
        required=True,
        type=Path,
        help='Path to attack configuration YAML file'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path("./cache"),
        help='Override output directory from config'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without running pipeline'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage'
    )

    #no cache
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching of datasets'
    )

    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path("./data"),
        help='Directory to store datasets'
    )

    parser.add_argument(
        '--log-dir',
        type=Path,
        default=Path("./logs"),
        help='Directory to store logs'
    )

    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path("./results"),
        help='Directory to store results'
    )
    return parser.parse_args()

def main():
    args = parse_arguments()

    pipeline_id = hash_file(args.config)
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logger(timestamp, log_level=log_level, pipeline_id=pipeline_id, log_dir=args.log_dir)
    logger = logging.getLogger("defense_pipeline")

    config = validate_and_load_pipeline_config(args.config)
    attacks = validate_and_load_attack_config(args.attack_config)
    
    settings = Settings(
        config=config,
        attacks=attacks,
        pipeline_id=pipeline_id,
        output_dir=args.output_dir,
        use_gpu=not args.no_gpu,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        logs_dir=args.log_dir,
        timestamp=timestamp,
        use_cache=not args.no_cache,
        log_level=log_level
    )
        
    dataset_manager = DatasetManager(settings)
    # tool_runner = ToolRunner(settings)
    # model_evaluator = ModelEvaluator(settings)
    pipeline_executor = PipelineExecutor(settings, dataset_manager=dataset_manager)
    # components = create_components(settings)
    
    dataset_manager.prepare_dataset()
    #go to pipeline handler and let it run the pipeline
    pipeline_executor.run_all_combinations_parallel()
    # pipeline_executor.run_pipeline()
if __name__ == "__main__":
    main()