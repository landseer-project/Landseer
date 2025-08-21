import logging
import os
import shutil
from pathlib import Path
from filelock import FileLock
from landseer_pipeline.config import Settings
from typing import List, Dict, Optional, Tuple
from landseer_pipeline.config import Stage
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import itertools
from landseer_pipeline.tools import ToolRunner
from landseer_pipeline.evaluator import ModelEvaluator
from landseer_pipeline.pipeline.cache import CacheManager
from landseer_pipeline.utils import ResultLogger, GPUAllocator, temp_manager
from landseer_pipeline.utils.files import copy_or_link_log
from landseer_pipeline.config import ToolConfig, Stage
from landseer_pipeline.dataset_handler import DatasetManager
from landseer_pipeline.config import ToolConfig, DockerConfig
import time
import csv
import torch
import json

from itertools import permutations, product

logger = logging.getLogger(__name__)

class Combination:
	def __init__(self, idx: int, settings: Settings, dataset_manager: Optional[DatasetManager] = None, stage_combo_dict: Optional[Dict] = None):
		self.id = idx
		self.combo_output_dir = settings.results_dir / "output" / f"comb_{idx:03d}"
		self.tools_by_stage = stage_combo_dict or {}
		self.file_provenance = {}  # Track which tool created each file
		
		# Create the combination output directory
		self.combo_output_dir.mkdir(parents=True, exist_ok=True)

	def copy_output_to_results_dir(self, combination: str, stage: str, tool_name: str, tool_output_path: Path) -> None:
		"""Copy tool outputs to combination directory and track file sources"""
		if not tool_output_path.exists():
			logger.warning(f"Tool output path does not exist: {tool_output_path}")
			return
			
		json_file = self.combo_output_dir / "fin_output_paths.json"
		
		# Load existing data or create new
		data = {}
		if json_file.exists():
			try:
				data = json.loads(json_file.read_text())
			except json.JSONDecodeError:
				logger.warning(f"Invalid JSON in {json_file}, starting fresh")
				data = {}
		
		# Copy files and track their sources
		if tool_output_path.is_file():
			dest_file = self.combo_output_dir / tool_output_path.name
			shutil.copy2(tool_output_path, dest_file)
			data[tool_output_path.name] = {
				"source_path": str(tool_output_path.resolve()),
				"stage": stage,
				"tool_name": tool_name
			}
		elif tool_output_path.is_dir():
			for file_path in tool_output_path.rglob("*"):
				if file_path.is_file():
					relative_path = file_path.relative_to(tool_output_path)
					dest_file = self.combo_output_dir / relative_path
					dest_file.parent.mkdir(parents=True, exist_ok=True)
					shutil.copy2(file_path, dest_file)
					data[str(relative_path)] = {
						"source_path": str(file_path.resolve()),
						"stage": stage,
						"tool_name": tool_name
					}
		
		# Save updated tracking file
		json_file.write_text(json.dumps(data, indent=4))
		logger.debug(f"Copied outputs from {tool_name} ({stage}) to {self.combo_output_dir}")

class PipelineExecutor:
	
	def __init__(self, settings: Settings, dataset_manager: Optional[DatasetManager] = None):
		"""Initialize the pipeline executor with settings"""
		self.settings = settings
		self.config = settings.config
		self.attacks = settings.attacks
		self.dataset_manager = dataset_manager
		self.cache_manager = CacheManager(settings)
		self.logger = ResultLogger(settings.results_dir, settings.pipeline_id)
		self.gpu_allocator = GPUAllocator()
		self.combinations: Dict[str, Combination] = {}

	@property
	def dataset_dir(self) -> Path:
		"""Returns the dataset directory"""
		return self.dataset_manager.dataset_dir
	
	@property
	def pipeline_dataset_dir(self) -> Path:
		if self.attacks.attacks.backdoor and self.dataset_manager.poisoned_dataset_dir:
			return self.dataset_manager.poisoned_dataset_dir
		return self.dataset_manager.clean_dataset_dir
	
	@property
	def pipeline_dataset_type(self) -> str:
		if self.attacks.attacks.backdoor and self.dataset_manager.poisoned_dataset_dir:
			return "poisoned"
		return "clean"
		
	def get_tools_for_stage(self, stage: str) -> List[ToolConfig]:
		"""Get the list of tools for a specific stage"""
		stage_config = self.config.pipeline.get(stage)
		return stage_config.tools if stage_config else []
	
	def get_tools_for_combination(self, combination: str, stage: str) -> List[ToolConfig]:
		"""Get the list of tools for a specific combination and stage"""
		if combination not in self.combinations:
			raise ValueError(f"Combination '{combination}' not found.")
		return self.combinations[combination].tools_by_stage.get(stage, [])
	
	def run_all_combinations_parallel(self) -> None:
		"""Execute all combinations in parallel using ThreadPoolExecutor"""
		self.make_combinations()
		
		# Limit workers to available GPUs to prevent blocking
		available_gpus = self.gpu_allocator.num_gpus
		max_workers = min(len(self.combinations), available_gpus)
		
		logger.info(f"Starting parallel execution with {max_workers} workers (GPU-limited from {available_gpus} GPUs)")
		
		try:
			with ThreadPoolExecutor(max_workers=max_workers) as executor:
				# Submit all combinations
				future_to_combo = {
					executor.submit(self.run_combination, combo_id): combo_id 
					for combo_id in self.combinations
				}
				
				# Process completed combinations
				completed = 0
				total = len(future_to_combo)
				
				for future in as_completed(future_to_combo):
					combo_id = future_to_combo[future]
					completed += 1
					
					try:
						future.result()
						logger.info(f"✓ [{completed}/{total}] Combination {combo_id} completed successfully")
					except Exception as e:
						logger.error(f"✗ [{completed}/{total}] Combination {combo_id} failed: {e}")
		
		except KeyboardInterrupt:
			logger.warning("Parallel execution interrupted by user. Cleaning up...")
			# The temp_manager will handle cleanup via signal handlers
			raise
		
		logger.info("Parallel execution completed")

	
	def get_tools_for_combination(self, combination: str, stage: str) -> List[Dict]:
		"""Get the list of tools for a specific combination and stage"""
		if combination not in self.combinations:
			raise ValueError(f"Combination '{combination}' not found.")
		tools = self.combinations[combination].get(stage, {})
		return [tools] if tools else []
	
	def run_pipeline(self) -> None:
		"""Execute the complete pipeline for all combinations"""
		self.make_combinations()
		logger.info("\n" + "="*60)
		logger.info("PIPELINE STARTED")
		logger.info("="*60)
		
		total_combinations = len(self.combinations)
		
		for i, combination in enumerate(self.combinations, 1):
			logger.info(f"Running combination {i}/{total_combinations}: {combination}")
			try:
				self.run_combination(combination)
				logger.info(f"✓ Completed combination: {combination}")
			except Exception as e:
				logger.error(f"✗ Combination {combination} failed: {e}")
				# Continue with next combination instead of stopping
				continue
		
		logger.info("\n" + "="*60)
		logger.info("PIPELINE COMPLETED")
		logger.info("="*60)

	def make_combinations(self) -> None:
		"""Generate all possible combinations of tools across stages"""
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
				dummy_docker = DockerConfig(
					image="ghcr.io/landseer-project/post_noop:v1", 
					command="python main.py"
				)
				noop = ToolConfig(name="noop", docker=dummy_docker)
			
			# Initialize stage options with noop
			stage_options = [(noop,)]
			
			# Add all permutations of tools if tools exist
			if tools:
				for r in range(1, len(tools) + 1):
					stage_options.extend(permutations(tools, r))
			
			options_per_stage.append(stage_options)
			logger.debug(f"Stage {stage.value} has {len(stage_options)} options (including noop)")
		
		# Generate all combinations
		all_combinations = list(product(*options_per_stage))
		combinations = {}
		logger.info(f"Generated {len(all_combinations)} total combinations")
		
		for idx, combo in enumerate(all_combinations):
			key = f"comb_{idx:03d}"
			stage_combo_dict = {stage: list(combo[i]) for i, stage in enumerate(stages)}
			combinations[key] = Combination(idx, self.settings, self.dataset_manager, stage_combo_dict)

		def count_noops(combination: Combination) -> int:
			return sum(
				1 for tools in combination.tools_by_stage.values() 
				if len(tools) == 1 and getattr(tools[0], "name", "") == "noop"
			)

		# Sort combinations (noop-heavy combinations first)
		sorted_combos = sorted(combinations.items(), key=lambda x: count_noops(x[1]), reverse=True)
		self.combinations = dict(sorted_combos)
		
		# Export to CSV
		export_combinations_to_csv("all_pipeline_combinations.csv", self.combinations)
		logger.info(f"Successfully generated {len(self.combinations)} combinations")
	
	def run_combination(self, combination: str) -> None:
		"""Execute a single combination of tools across all stages"""
		combination_obj = self.combinations[combination]
		context = self._initialize_combination_context(combination)
		
		try:
			logger.info(f"Starting combination: {combination}")
			
			# Execute all stages
			for stage in [stage.value for stage in Stage]:
				stage_outputs = self._execute_stage(combination, stage, combination_obj, context)
				context["all_output_paths"][stage] = stage_outputs
			
			# Evaluate final model
			duration = time.time() - context["start_time"]
			self._evaluate_and_log_combination(combination, combination_obj, context, duration)
			
			logger.info(f"Completed combination: {combination}")
			
		except Exception as e:
			logger.error(f"Combination {combination} failed: {e}")
			raise

	def _initialize_combination_context(self, combination: str) -> Dict:
		"""Initialize context for combination execution"""
		return {
			"current_input": self.pipeline_dataset_dir,
			"dataset_dir": self.pipeline_dataset_dir,
			"start_time": time.time(),
			"toolnames_by_stage": {stage.value: [] for stage in Stage},
			"cache_key": hashlib.sha256(b"").hexdigest() + "_init",
			"all_output_paths": {}
		}

	def _execute_stage(self, combination: str, stage: str, combination_obj: Combination, context: Dict) -> List[Path]:
		"""Execute all tools in a specific stage"""
		tools = combination_obj.tools_by_stage.get(stage, [])
		logger.info(f"{combination} --- STAGE: {stage.upper()} ---")
		
		if not tools:
			logger.debug(f"{combination}: No tools for stage '{stage}'. Skipping.")
			return []
		
		stage_outputs = []
		context["toolnames_by_stage"][stage] = [tool.name for tool in tools]
		
		for tool in tools:
			logger.info(f"[Tool: {tool.name}] Starting execution...")
			
			if self._should_skip_tool(tool, stage):
				logger.info(f"[+] {combination}: Skipping {tool.name}")
				continue
			
			tool_output_path = self._execute_single_tool(combination, tool, stage, context)
			
			# Copy outputs to combination directory
			combination_obj.copy_output_to_results_dir(combination, stage, tool.name, tool_output_path)
			
			# Update context
			self._update_context_after_tool(context, stage, tool_output_path)
			stage_outputs.append(tool_output_path)
			
			# Copy tool logs
			self._copy_tool_logs(combination, tool, stage, context["cache_key"])
			
			logger.info(f"{combination}: Tool '{tool.name}' completed successfully.")
		
		return stage_outputs

	def _should_skip_tool(self, tool: ToolConfig, stage: str) -> bool:
		"""Determine if a tool should be skipped"""
		return tool.name == "noop" and stage in ["post_training", "deployment"]

	def _execute_single_tool(self, combination: str, tool: ToolConfig, stage: str, context: Dict) -> Path:
		"""Execute a single tool with caching and GPU management"""
		cache_key = self.cache_manager.compute_cache_key(
			context["cache_key"], tool, stage, context["current_input"], context["dataset_dir"]
		)
		cache_path, lock = self.cache_manager.safe_cache_path(cache_key)
		
		try:
			tool_output_path = self._run_tool_with_caching(
				combination, tool, stage, cache_path, cache_key, context
			)
			
			# Update cache key for next tool
			context["cache_key"] = hashlib.sha256(f"{cache_key}_{tool_output_path}".encode()).hexdigest()
			
			return tool_output_path
			
		finally:
			lock.release()

	def _run_tool_with_caching(self, combination: str, tool: ToolConfig, stage: str, 
							  cache_path: Path, cache_key: str, context: Dict) -> Path:
		"""Run tool with cache logic"""
		in_progress_marker = cache_path / ".in_progress"
		success_marker = cache_path / ".success"
		
		logger.info(f"[CACHE] {combination}: Tool '{tool.name}' at stage '{stage}'")
		
		if success_marker.exists() and not in_progress_marker.exists() and self.settings.use_cache:
			logger.info(f"[CACHE] HIT: Using cached output for {tool.name}")
			return cache_path / "output"
		else:
			logger.info(f"[CACHE] MISS: Computing {tool.name}")
			return self._run_tool_fresh(combination, tool, stage, cache_path, cache_key, context)

	def _run_tool_fresh(self, combination: str, tool: ToolConfig, stage: str, 
					   cache_path: Path, cache_key: str, context: Dict) -> Path:
		"""Execute tool without using cache"""
		output_path = cache_path / "output"
		os.makedirs(output_path, exist_ok=True)
		
		in_progress_marker = cache_path / ".in_progress"
		success_marker = cache_path / ".success"
		in_progress_marker.touch()
		
		gpu_id = None
		try:
			gpu_id = self.gpu_allocator.allocate_gpu()
			logger.debug(f"{combination}: Allocated GPU {gpu_id} for {tool.name}")
			
			tool_runner = ToolRunner(
				self.settings, tool, stage,
				dataset_dir=context["dataset_dir"],
				input_path=context["current_input"],
				output_path=output_path,
				gpu_id=gpu_id
			)
			
			tool_output_path, duration = tool_runner.run_tool(combination_id=combination)
			tool.set_output_path(str(tool_output_path))
			
			success_marker.touch()
			if in_progress_marker.exists():
				in_progress_marker.unlink()
			
			self.logger.log_tool(
				combination, stage, tool.name, str(cache_path), 
				str(tool_output_path), duration, "success"
			)
			
			return tool_output_path
			
		except Exception as e:
			logger.error(f"{combination}: Tool '{tool.name}' failed: {e}")
			if in_progress_marker.exists():
				in_progress_marker.unlink()
			
			self.cache_manager.mark_as_failed(cache_key)  # Use cache_key instead of str(cache_path)
			self.logger.log_tool(
				combination, stage, tool.name, str(cache_path), 
				"failed", 0, "failure"
			)
			raise
		finally:
			if gpu_id is not None:
				self.gpu_allocator.release_gpu(gpu_id)

	def _update_context_after_tool(self, context: Dict, stage: str, tool_output_path: Path) -> None:
		"""Update execution context after tool completion"""
		if stage != "deployment":
			context["current_input"] = tool_output_path
		elif tool_output_path.is_dir() and (tool_output_path / "model.pt").exists():
			context["current_input"] = tool_output_path
		
		if stage == "pre_training":
			context["dataset_dir"] = context["current_input"]
			logger.debug(f"Updated dataset directory: {context['dataset_dir']}")
		
		# Handle multi-component defenses
		if stage == "post_training":
			self._handle_multi_component_defense(tool_output_path)
	
	def _handle_multi_component_defense(self, tool_output_path: Path) -> None:
		"""Process multi-component defenses and create composite models"""
		metadata_file = tool_output_path / "defense_metadata.json"
		
		if metadata_file.exists():
			logger.info(f"Detected multi-component defense at {tool_output_path}")
			
			try:
				import json
				with open(metadata_file, 'r') as f:
					metadata = json.load(f)
				
				if metadata.get('defense_attribute') == 'multi_component':
					integration_script = metadata.get('integration_script')
					
					if integration_script:
						# Run the integration script to create composite model
						script_path = tool_output_path / integration_script
						if script_path.exists():
							import subprocess
							import sys
							
							cmd = [
								sys.executable, str(script_path),
								str(metadata_file),
								str(tool_output_path),
								str(tool_output_path / "model_composite.pt")
							]
							
							result = subprocess.run(cmd, capture_output=True, text=True)
							
							if result.returncode == 0:
								# Replace original model.pt with composite
								composite_path = tool_output_path / "model_composite.pt"
								original_path = tool_output_path / "model.pt"
								
								if composite_path.exists():
									import shutil
									shutil.move(str(composite_path), str(original_path))
									logger.info("Successfully created composite model for multi-component defense")
								else:
									logger.warning("Composite model not created, using original model")
							else:
								logger.error(f"Integration script failed: {result.stderr}")
								logger.warning("Falling back to original model")
				
			except Exception as e:
				logger.error(f"Error processing multi-component defense: {e}")
				logger.warning("Falling back to original model")

	def _copy_tool_logs(self, combination: str, tool: ToolConfig, stage: str, cache_key: str) -> None:
		"""Copy tool logs to results directory"""
		cache_log = Path(cache_key) / "tool_logs" / f"{stage}_{tool.name.replace(' ', '_')}.log"
		result_log = self.settings.results_dir / "tool_logs" / combination / f"{stage}_{tool.name.replace(' ', '_')}.log"
		
		if cache_log.exists():
			copy_or_link_log(str(cache_log), str(result_log), method="copy")
		else:
			logger.warning(f"{combination}: Cache log not found. Skipping log copy.")

	def _evaluate_and_log_combination(self, combination: str, combination_obj: Combination, 
									 context: Dict, duration: float) -> None:
		"""Evaluate final model and log combination results"""
		logger.info(f"{combination}: Evaluating final model...")
		
		gpu_id = None
		try:
			gpu_id = self.gpu_allocator.allocate_gpu()
			
			model_evaluator = ModelEvaluator(
				settings=self.settings,
				dataset_manager=self.dataset_manager,
				attacks=self.attacks.attacks,
				tools_by_stage=combination_obj.tools_by_stage,
				gpu_id=gpu_id,
				combination_output=combination_obj.combo_output_dir
			)
			
			final_acc = model_evaluator.evaluate_model(str(self.pipeline_dataset_dir))
			
			self.logger.log_combination(
				combination,
				tools_by_stage=context["toolnames_by_stage"],
				dataset_name=self.settings.config.dataset.name,
				dataset_type=self.pipeline_dataset_type,
				acc=final_acc,
				duration=duration
			)
			
			logger.info(f"{combination}: Evaluation completed: {final_acc}")
			
		finally:
			if gpu_id is not None:
				self.gpu_allocator.release_gpu(gpu_id)

def export_combinations_to_csv(filename: str = "generated_combinations.csv", 
							  combinations: Optional[Dict[str, Combination]] = None) -> None:
	"""Export combinations to CSV file with improved error handling"""
	if not combinations:
		raise ValueError("No combinations provided. Run make_combinations() first.")
	
	fieldnames = ["combination_id"] + [stage.value for stage in Stage]

	try:
		with open(filename, mode="w", newline='') as csv_file:
			writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
			writer.writeheader()

			for combo_id, combination_obj in combinations.items():
				row = {"combination_id": combo_id}
				for stage in Stage:
					tools = combination_obj.tools_by_stage.get(stage, [])
					tool_names = " → ".join(tool.name for tool in tools if tool)
					row[stage.value] = tool_names or "none"
				writer.writerow(row)

		logger.info(f"Exported {len(combinations)} combinations to {filename}")
		
	except IOError as e:
		logger.error(f"Failed to export combinations to {filename}: {e}")
		raise