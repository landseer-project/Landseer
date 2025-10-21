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
from .config.settings import set_temp_dry_run

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

    # Docker resource controls
    parser.add_argument('--docker-shm-size', type=str, default='1g', help='Set Docker shared memory size (e.g. 512m, 1g, 2g)')
    parser.add_argument('--docker-mem-limit', type=str, default=None, help='Set Docker container memory limit (e.g. 8g). Omit for no explicit limit')

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
    
    # Set temporary dry-run state before config validation
    if args.dry_run:
        set_temp_dry_run(True)

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
        dry_run=args.dry_run,  # Add dry_run flag
        log_level=log_level
    )
    # Inject docker resource overrides (not part of original frozen dataclass signature; using object.__setattr__)
    try:
        object.__setattr__(settings, 'docker_shm_size', args.docker_shm_size)
        object.__setattr__(settings, 'docker_mem_limit', args.docker_mem_limit)
    except Exception as _e:
        logging.getLogger().warning(f"Failed to set docker resource attributes: {_e}")

    # Setup secondary per-(config,attack) log (legacy run_logs) in addition to unified pipeline log
    log_dir = Path('run_logs')
    log_dir.mkdir(exist_ok=True)
    config_name = Path(args.config).stem
    attack_name = Path(args.attack_config).stem
    log_file = log_dir / f"{config_name}__{attack_name}.log"
    setup_logging(str(log_file))

    # --- Context / provenance logging block ---
    try:
        root_logger = logging.getLogger()  # root logger already has file + console handlers
        root_logger.info("\n========== PIPELINE CONTEXT ==========")
        root_logger.info(f"Pipeline ID          : {pipeline_id}")
        root_logger.info(f"Timestamp            : {timestamp}")
        root_logger.info(f"Config file          : {Path(args.config).resolve()}")
        root_logger.info(f"Attack config file   : {Path(args.attack_config).resolve()}")
        # Dataset details
        ds = config.dataset
        root_logger.info("-- Dataset --")
        root_logger.info(f"  Name               : {ds.name}")
        root_logger.info(f"  Variant            : {getattr(ds, 'variant', 'clean')}")
        if getattr(ds, 'version', None):
            root_logger.info(f"  Version            : {ds.version}")
        if ds.params:
            # Limit extremely long param dumps
            import json as _json
            params_json = _json.dumps(ds.params, sort_keys=True)
            if len(params_json) > 400:
                params_json = params_json[:400] + "... (truncated)"
            root_logger.info(f"  Params             : {params_json}")
        # Model details
        model_cfg = config.model
        root_logger.info("-- Model --")
        root_logger.info(f"  Script             : {model_cfg.script}")
        root_logger.info(f"  Framework          : {model_cfg.framework}")
        root_logger.info(f"  Script hash        : {model_cfg.content_hash}")
        if model_cfg.params:
            import json as _json
            mparams_json = _json.dumps(model_cfg.params, sort_keys=True)
            if len(mparams_json) > 400:
                mparams_json = mparams_json[:400] + "... (truncated)"
            root_logger.info(f"  Params             : {mparams_json}")
        # Tools by stage
        root_logger.info("-- Tools By Stage --")
        # To preserve stage order use enum ordering from schemas.pipeline.Stage
        from .config.schemas.pipeline import Stage as _Stage
        for stage in _Stage:
            stage_cfg = config.pipeline.get(stage)
            if not stage_cfg:
                root_logger.warning(f"  {stage.value}: <missing stage config>")
                continue
            tool_entries = []
            for t in (stage_cfg.tools or []):
                # include docker image short name and (optional) dataset label
                image = getattr(t.docker, 'image', '')
                image_short = image.split('/')[-1]
                tool_entries.append(f"{t.name}[{image_short}]")
            if not tool_entries:
                tool_entries = ["<none>"]
            root_logger.info(f"  {stage.value:15s}: {', '.join(tool_entries)}")
        root_logger.info("======================================\n")
    except Exception as _ctx_e:
        logging.getLogger().warning(f"Failed to log pipeline context: {_ctx_e}")

    # Initialize GPU manager
    gpu_manager = GPUManager(max_temp=args.max_temp, cooldown_time=args.cooldown_time) 
    # Clean up any leftover temporary directories from previous runs
    temp_manager.cleanup_existing_temp_dirs()
    
    try:
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

if __name__ == "__main__":
    main()