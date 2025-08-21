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
from typing import Optional
sys.path.insert(0, str(Path(__file__).parent / ".." / "src"))  # Adjust the path to your src directory

import logging
from .dataset_handler import DatasetManager
from .pipeline import PipelineExecutor
from .config import Settings, validate_and_load_pipeline_config, validate_and_load_attack_config
from .utils import hash_file, setup_logger
from .utils.temp_manager import temp_manager
from .gpu_manager import GPUManager

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

    parser.add_argument('--max-temp', type=float, default=80.0, help='Maximum GPU temperature')
    parser.add_argument('--cooldown-time', type=int, default=300, help='GPU cooldown time in seconds')

    return parser.parse_args()

def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )

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

    # Setup logging
    log_dir = Path('run_logs')
    log_dir.mkdir(exist_ok=True)
    config_name = Path(args.config).stem
    attack_name = Path(args.attack_config).stem
    log_file = log_dir / f"{config_name}__{attack_name}.log"
    setup_logging(str(log_file))

    # Initialize GPU manager
    gpu_manager = GPUManager(max_temp=args.max_temp, cooldown_time=args.cooldown_time) 
    # Clean up any leftover temporary directories from previous runs
    temp_manager.cleanup_existing_temp_dirs()
    
    try:
        # Get available GPU
        gpu_id = gpu_manager.get_available_gpu()
        if gpu_id is None:
            logger.error("No available GPU found that meets temperature requirements")
            sys.exit(1)
        logger.info(f"Using GPU {gpu_id}")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        dataset_manager = DatasetManager(settings)
        dataset_manager.prepare_dataset()
        pipeline_executor = PipelineExecutor(settings, dataset_manager=dataset_manager)
        pipeline_executor.run_all_combinations_parallel()
        Path(settings.results_dir, ".success").touch()
    except KeyboardInterrupt:
        logger.error("Pipeline interrupted by user!")
        # Ensure cleanup of temporary directories
        temp_manager.cleanup_all()
        Path(settings.results_dir, ".failed").touch()
        raise
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        Path(settings.results_dir, ".failed").touch()
        raise e
    finally:
        if 'gpu_id' in locals():
            gpu_manager.release_gpu(gpu_id)

if __name__ == "__main__":
    main()