import logging
import os
from pathlib import Path
from filelock import FileLock
from landseer_pipeline.config import Settings
from typing import List, Dict
from landseer_pipeline.config import Stage
from concurrent.futures import ThreadPoolExecutor
import itertools
from landseer_pipeline.tools import ToolRunner
from landseer_pipeline.evaluator import ModelEvaluator
from landseer_pipeline.pipeline.cache import CacheManager
from landseer_pipeline.utils import ResultLogger, GPUAllocator
from landseer_pipeline.utils.files import copy_or_link_log
import time
import torch

logger = logging.getLogger(__name__)

class PipelineExecutor():
    
    def __init__(self, settings: Settings, dataset_manager=None):
        """Initialize the pipeline executor with settings"""
        self.settings = settings
        self.config = settings.config
        self.attacks = settings.attacks
        self.dataset_manager = dataset_manager
        self.model_evaluator = ModelEvaluator(
            settings=settings,
            dataset_manager=self.dataset_manager,
            attacks=self.attacks.attacks,
            device=settings.device
        )
        self.cache_manager = CacheManager(settings)
        self.logger = ResultLogger(settings.results_dir, settings.pipeline_id)
        self.gpu_allocator = GPUAllocator()

    @property
    def dataset_dir(self):
        """Returns the dataset directory"""
        return self.dataset_manager.dataset_dir
    
    @property
    def pipeline_dataset_dir(self):
        if self.attacks.attacks.backdoor and self.dataset_manager.poisoned_dataset_dir:
            return self.dataset_manager.poisoned_dataset_dir
        return self.dataset_manager.clean_dataset_dir
    
    @property
    def pipeline_dataset_type(self):
        if self.attacks.attacks.backdoor and self.dataset_manager.poisoned_dataset_dir:
            return "poisoned"
        return "clean"
        
    def get_tools_for_stage(self, stage: str) -> List[Dict]:
        """Get the list of tools for a specific stage"""
        return self.config.get("pipeline",[]).get(stage, []).get("tools", [])
    
    def run_all_combinations_parallel(self):
        self.combinations = self.make_combinations()
        with ThreadPoolExecutor(max_workers=min(len(self.combinations), os.cpu_count())) as executor:
            futures = [executor.submit(self.run_combination, combo) for combo in self.combinations]
            for future in futures:
                future.result()
    
    def get_tools_for_combination(self, combination: str, stage: str) -> List[Dict]:
        """Get the list of tools for a specific combination and stage"""
        if combination not in self.combinations:
            raise ValueError(f"Combination '{combination}' not found.")
        tools = self.combinations[combination].get(stage, {})
        return [tools] if tools else []
    
    def run_pipeline(self):
        # run for even different dataset combinations like clean poisoned etc.
        self.combinations = self.make_combinations()
        logger.info("\n============================\n PIPELINE STARTED\n============================")
        for combination in self.combinations:
            logger.info(f"---------Running combination: {combination}---------")
            self.run_combination(combination)
            logger.info(f"Completed combination: {combination}")
            logger.info(f"Pipeline completed successfully for combination: {combination}")
        logger.info("\n============================\n PIPELINE COMPLETED\n============================")

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
        # print(f"Combinations generated: {combinations}")
        return combinations
    
    def run_combination(self, combination):
        current_input = self.pipeline_dataset_dir
        dataset_dir = self.pipeline_dataset_dir
        stages = [stage.value for stage in Stage]
        comb_start = time.time()
        tools_by_stage = {"pre_training": [], "during_training": [], "post_training": []}

        for stage in stages:
            logger.info(f"{combination}--- STAGE: {stage.upper()} ---")
            tools = self.get_tools_for_combination(combination, stage)
            if not tools:
                logger.debug(f"{combination}: No tools configured for stage '{stage}'. Skipping.")
                continue
            tools_by_stage[stage] = tools if tools else []
            
            for tool in tools:
                logger.info(f"[Tool: {tool.name}] Starting execution...")
                
                if tool.name == "noop" and stage == "post_training":
                    logger.info(f"[+] {combination}: Noop done")
                    continue
                
                cache_key = self.cache_manager.compute_cache_key(tool, stage, current_input, self.pipeline_dataset_dir)
                cache_path, lock = self.cache_manager.safe_cache_path(cache_key)
                in_progress_marker = cache_path / ".in_progress"
                success_marker = cache_path / ".success"
                
                try:
                    if success_marker.exists() and self.settings.use_cache:
                        logger.info(f"[+] {combination}: Using cached at {cache_path}")
                        tool_output_path = cache_path / "output"
                    else:
                        logger.info(f"[+] {combination}: Running tool '{tool.name}' at stage '{stage}'")
                        os.makedirs(cache_path / "output", exist_ok=True)
                        in_progress_marker.touch()

                        try:
                            gpu_id = self.gpu_allocator.allocate_gpu()
                            tool_runner = ToolRunner(
                            self.settings, tool, stage, dataset_dir=dataset_dir,
                            input_path=current_input,
                            output_path=cache_path / "output", 
                            gpu_id=gpu_id
                            )
                            tool_output_path, toolrun_duration = tool_runner.run_tool(
                                combination_id=combination
                            )
                            success_marker.touch()
                            self.logger.log_tool(combination, stage, tool.name, cache_key, str(tool_output_path), toolrun_duration, "success")

                            if in_progress_marker.exists():
                                in_progress_marker.unlink()
                        except Exception as e:
                            logger.error(f"{combination}: Tool '{tool.name}' failed at stage '{stage}': {e}")
                            self.cache_manager.mark_as_failed(cache_key)
                            self.logger.log_tool(
                                combination,
                                stage,
                                tool.name,
                                cache_key,
                                str(tool_output_path),
                                0,  
                                "failure"
                            )
                            raise                            
                    current_input = str(tool_output_path)
                    if stage == "pre_training":
                        dataset_dir = current_input
                        logger.debug(f"[+] {combination}: Updated dataset directory: {dataset_dir}")
                    logger.info(f"{combination}: Tool '{tool.name}' completed successfully.")
                    cache_log = os.path.join(
                             cache_path, "tool_logs", f"{stage}_{tool.name.replace(' ', '_')}.log")
                    result_log = os.path.join(
                             self.settings.results_dir, "tool_logs", combination, f"{stage}_{tool.name.replace(' ', '_')}.log")
                    if os.path.exists(cache_log):
                        copy_or_link_log(cache_log, result_log, method="copy")
                    else:
                        logger.warning(f"{combination}: Cache log not found at {cache_log}. Skipping log copy.")
                finally:
                    lock.release()

        comb_duration = time.time() - comb_start
        model_path = os.path.join(current_input, "model.pt")
        if os.path.exists(model_path):
            logger.info(f"{combination}:Evaluating final model...")
            final_acc = self.model_evaluator.evaluate_model(model_path, self.pipeline_dataset_dir)
            self.logger.log_combination(combination, tools_by_stage=tools_by_stage, dataset_name=self.settings.config.dataset.name,dataset_type=self.pipeline_dataset_type, acc=final_acc, duration=comb_duration)
            logger.info(f"{combination}: Evaluation completed : {final_acc}")
        else:
            logger.warning(f"{combination}: Model not found at {model_path}. Skipping evaluation.")