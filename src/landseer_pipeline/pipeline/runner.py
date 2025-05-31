import logging
import os
from pathlib import Path
from filelock import FileLock
from landseer_pipeline.config import Settings
from typing import List, Dict, Optional
from landseer_pipeline.config import Stage
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from landseer_pipeline.tools import ToolRunner
from landseer_pipeline.evaluator import ModelEvaluator
from landseer_pipeline.pipeline.cache import CacheManager
from landseer_pipeline.utils import ResultLogger, GPUAllocator
from landseer_pipeline.utils.files import copy_or_link_log
from landseer_pipeline.config import ToolConfig, Stage
from landseer_pipeline.dataset_handler import DatasetManager
from landseer_pipeline.config import ToolConfig, DockerConfig
import time
import csv
import torch

from itertools import permutations, product

logger = logging.getLogger(__name__)

class PipelineExecutor():
	
	def __init__(self, settings: Settings, dataset_manager=None):
		"""Initialize the pipeline executor with settings"""
		self.settings = settings
		self.config = settings.config
		self.attacks = settings.attacks
		self.dataset_manager = dataset_manager
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
	
	def run_all_combinations_parallel_new(self):
		max_workers = torch.cuda.device_count() if self.settings.use_gpu else os.cpu_count()
	
	def run_all_combinations_parallel(self):
		self.make_combinations()
		with ThreadPoolExecutor(max_workers=min(len(self.combinations), os.cpu_count())) as executor:
			future_to_combo = { executor.submit(self.run_combination, combo): combo for combo in self.combinations }
			for future in as_completed(future_to_combo):
				combo = future_to_combo[future]
				try:
					future.result()
				except Exception as e:
					logger.error(f"Combination {combo} failed with error: {e}")

	
	def get_tools_for_combination(self, combination: str, stage: str) -> List[Dict]:
		"""Get the list of tools for a specific combination and stage"""
		if combination not in self.combinations:
			raise ValueError(f"Combination '{combination}' not found.")
		tools = self.combinations[combination].get(stage, {})
		return [tools] if tools else []
	
	def run_pipeline(self):
		# run for even different dataset combinations like clean poisoned etc.
		self.make_combinations()
		logger.info("\n============================\n PIPELINE STARTED\n============================")
		for combination in self.combinations:
			logger.info(f"---------Running combination: {combination}---------")
			self.run_combination(combination)
			logger.info(f"Completed combination: {combination}")
			logger.info(f"Pipeline completed successfully for combination: {combination}")
		logger.info("\n============================\n PIPELINE COMPLETED\n============================")

	def make_combinations(self):
		pipeline = self.config.pipeline
		stages = [stage for stage in Stage]
		options_per_stage = []
		
		for stage in stages:
			stage_config = pipeline.get(stage)
			if stage_config is None:
				raise ValueError(f"Missing StageConfig for stage: {stage}")
			
			tools: List[ToolConfig] = stage_config.tools or []
			noop: Optional[ToolConfig] = stage_config.noop
			
			# Create noop if it doesn't exist
			if noop is None:
				dummy_docker = DockerConfig(image="ghcr.io/landseer-project/post_noop:v1", command="python main.py")
				noop = ToolConfig(name="noop", docker=dummy_docker)
			
			# Initialize stage options with noop
			stage_options = [(noop,)]
			
			# Add all permutations of tools if tools exist
			if tools:
				for r in range(1, len(tools) + 1):
					stage_options.extend(permutations(tools, r))
			
			# Add this stage's options to the main list
			options_per_stage.append(stage_options)
			print(f"Stage {stage.value} has {len(stage_options)} options with noop included.")
		
		all_combinations = list(product(*options_per_stage))
		combinations = {}
		print(f"Generated {len(all_combinations)} combinations with permutations.")
		
		for idx, combo in enumerate(all_combinations):
			key = f"comb_{idx:03d}"
			stage_combo_dict = {stage: list(combo[i]) for i, stage in enumerate(stages)}
			combinations[key] = stage_combo_dict

		def count_noops(stage_combo_dict):
			return sum(1 for tools in stage_combo_dict.values() if len(tools) == 1 and getattr(tools[0], "name", "") == "noop")
		
		sorted_combos = sorted(combinations.items(), key=lambda x: count_noops(x[1]), reverse=True)
		self.combinations = dict(sorted_combos)
		export_combinations_to_csv("all_pipeline_combinations.csv", self.combinations)
		logger.info(f"Generated {len(self.combinations)} combinations with permutations.")
	
	def run_combination(self, combination):
		combination_dict = self.combinations[combination]
		current_input = self.pipeline_dataset_dir
		dataset_dir = self.pipeline_dataset_dir
		stages = [stage.value for stage in Stage]
		comb_start = time.time()
		tools_by_stage = {"pre_training": [], "during_training": [], "post_training": []}

		all_output_paths = {}
		for stage in stages:
			stage_tool_outputs = []
			tools = combination_dict[stage]
			print(f"Tools for stage {stage}: {tools}")
			logger.info(f"{combination}--- STAGE: {stage.upper()} ---")
			if not tools:
				logger.debug(f"{combination}: No tools configured for stage '{stage}'. Skipping.")
				continue
			tools_by_stage[stage] = tools if tools else []			
			for tool in tools:
				logger.info(f"[Tool: {tool.name}] Starting execution...")
				
				if tool.name == "noop" and stage == "post_training":
					logger.info(f"[+] {combination}: Noop done")
					continue
				
				cache_key = self.cache_manager.compute_cache_key(tools, tool, stage, current_input, self.pipeline_dataset_dir)
				cache_path, lock = self.cache_manager.safe_cache_path(cache_key)
				in_progress_marker = cache_path / ".in_progress"
				success_marker = cache_path / ".success"

				logger.info(f"[CACHE] {combination}: Tool '{tool.name}' at stage '{stage}' -> cache_key: {cache_key}")
				if success_marker.exists():
					logger.info(f"[CACHE] HIT: Cache for {tool.name} at stage {stage} is ready.")
				elif in_progress_marker.exists():
					logger.info(f"[CACHE] WAIT: Cache for {tool.name} at stage {stage} is in progress.")
				else:
					logger.info(f"[CACHE] MISS: Will compute for {tool.name} at stage {stage}.")
				
				try:
					if success_marker.exists() and not in_progress_marker.exists() and self.settings.use_cache:
						logger.info(f"[+] {combination}: Using cached output for tool '{tool.name}' at stage '{stage}': {cache_path}")
						tool_output_path = cache_path / "output"
					else:
						logger.info(f"[+] {combination}: Running tool '{tool.name}' at stage '{stage}'")
						os.makedirs(cache_path / "output", exist_ok=True)
						in_progress_marker.touch()

						try:
							gpu_id = self.gpu_allocator.allocate_gpu()
							logging.debug(f"{combination}: Allocated GPU ID: {gpu_id} for tool '{tool.name}' at stage '{stage}'")
							try:
								tool_runner = ToolRunner(
							self.settings, tool, stage, dataset_dir=dataset_dir,
							input_path=current_input,
							output_path=cache_path / "output", 
							gpu_id=gpu_id
							)
							finally:
								self.gpu_allocator.release_gpu(gpu_id)
								
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
							self.logger.log_tool(combination, stage, tool.name, cache_key, str(tool_output_path), 0, "failure")
							raise 
					if stage != "post_training":
						current_input = str(tool_output_path)
					elif Path(tool_output_path).is_dir() and Path(tool_output_path/"model.pt").exists():
						current_input = tool_output_path
					stage_tool_outputs.append(tool_output_path)
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
			all_output_paths[stage] = stage_tool_outputs

		comb_duration = time.time() - comb_start
		logger.info(f"{combination}:Evaluating final model...")
		gpu_id = self.gpu_allocator.allocate_gpu()
		model_evaluator = ModelEvaluator(settings=self.settings,dataset_manager=self.dataset_manager,attacks=self.attacks.attacks,outputs=all_output_paths, gpu_id=gpu_id)
		final_acc = model_evaluator.evaluate_model(self.pipeline_dataset_dir)
		self.gpu_allocator.release_gpu(gpu_id)
		self.logger.log_combination(combination, tools_by_stage=tools_by_stage, dataset_name=self.settings.config.dataset.name,dataset_type=self.pipeline_dataset_type, acc=final_acc, duration=comb_duration)
		logger.info(f"{combination}: Evaluation completed : {final_acc}")

def export_combinations_to_csv(filename="generated_combinations.csv", combinations=None):
    if not combinations:
        raise ValueError("No combinations generated. Run make_combinations() first.")
    
    fieldnames = ["combination_id"] + [stage.value for stage in Stage]

    with open(filename, mode="w", newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for combo_id, stage_tool_dict in combinations.items():
            row = {"combination_id": combo_id}
            for stage in Stage:
                tools = stage_tool_dict.get(stage, [])
                tool_names = " → ".join(tool.name for tool in tools if tool)
                row[stage.value] = tool_names
            writer.writerow(row)

    logger.info(f"Exported {len(combinations)} combinations to {filename}")