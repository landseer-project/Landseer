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
from landseer_pipeline.pipeline.artifact_cache import ArtifactCache
from landseer_pipeline.utils import ResultLogger, GPUAllocator, temp_manager
from landseer_pipeline.gpu_manager import GPUManager
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
	def __init__(self, key: str, settings: Settings, dataset_manager: Optional[DatasetManager] = None, stage_combo_dict: Optional[Dict] = None):
		self.id = key
		self.combo_output_dir = settings.results_dir / "output" / f"{key}"
		self.tools_by_stage = stage_combo_dict or {}
		self.file_provenance = {}  # Track which tool created each file
		
		# Create the combination output directory
		self.combo_output_dir.mkdir(parents=True, exist_ok=True)

	def log_tool_output_dir_path(self, combination: str, stage: str, tool_name: str, tool_output_path: Path) -> None:
		"""Log tool outputs to combination directory and track file sources"""
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
		# Experimental global artifact cache (content-addressable)
		# Global artifact store independent of pipeline_id for cross-pipeline reuse
		self.artifact_cache = ArtifactCache(Path(settings.artifact_store_root))
		self._use_artifact_cache = True  # Legacy pipeline cache removed; always use artifact cache
		self.logger = ResultLogger(settings.results_dir, settings.pipeline_id)
		self.gpu_allocator = GPUAllocator()
		self.combinations: Dict[str, Combination] = {}
		# Track failing tool info per combination: combination_id -> (tool_name, artifact_log_path)
		self._failure_logs: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
		# For web UI: record per-tool artifact node hashes
		self._artifact_mappings_file = self.settings.results_dir / "artifact_mappings.json"
		if not self._artifact_mappings_file.exists():
			self._artifact_mappings_file.write_text("{}")

	def _record_artifact_mapping(self, combination_id: str, stage: str, tool_name: str, node_hash: str, cache_hit: bool) -> None:
		"""Append/update mapping of (combination, stage, tool) -> artifact node.

		Structure stored in JSON:
		{
		  "combination_id": {
		     "stage": [ { "tool": name, "node_hash": h, "cache_hit": bool } , ...]
		  },
		  ...
		}
		"""
		try:
			import json
			data = {}
			if self._artifact_mappings_file.exists():
				try:
					data = json.loads(self._artifact_mappings_file.read_text())
				except Exception:
					data = {}
			combo_entry = data.setdefault(combination_id, {})
			stage_list = combo_entry.setdefault(stage, [])
			stage_list.append({"tool": tool_name, "node_hash": node_hash, "cache_hit": cache_hit})
			self._artifact_mappings_file.write_text(json.dumps(data, indent=2))
		except Exception:
			logger.debug("Failed to record artifact mapping", exc_info=True)

	@property
	def dataset_dir(self) -> Path:
		"""Returns the dataset directory"""
		return self.dataset_manager.dataset_dir
	
	@property
	def pipeline_dataset_dir(self) -> Path:
		# Only use poisoned dataset if (a) attack config enables backdoor AND (b) directory actually exists
		# or the dataset variant explicitly requested is 'poisoned'.
		variant_requested = getattr(self.settings.config.dataset, 'variant', 'clean').lower()
		poison_dir = self.dataset_manager.poisoned_dataset_dir
		if variant_requested == 'poisoned':
			if poison_dir.exists():
				logger.debug("Using poisoned dataset (variant explicitly requested)")
				return poison_dir
			else:
				logger.warning("Variant 'poisoned' requested but poisoned dataset directory not found; falling back to clean")
		return self.dataset_manager.clean_dataset_dir
	
	@property
	def pipeline_dataset_type(self) -> str:
		variant_requested = getattr(self.settings.config.dataset, 'variant', 'clean').lower()
		poison_dir = self.dataset_manager.poisoned_dataset_dir
		if variant_requested == 'poisoned' and poison_dir.exists():
			return 'poisoned'
		if getattr(self.attacks.attacks, 'backdoor', False) and poison_dir.exists():
			return 'poisoned'
		return 'clean'
		
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
		# Track statuses in-memory for instant summary
		combination_statuses = {}
		
		for i, combination in enumerate(self.combinations, 1):
			logger.info(f"Running combination {i}/{total_combinations}: {combination}")
			try:
				self.run_combination(combination)
				logger.info(f"✓ Completed combination: {combination}")
				combination_statuses[combination] = "success"
			except Exception as e:
				logger.error(f"✗ Combination {combination} failed: {e}")
				combination_statuses[combination] = "failure"
				# Continue with next combination instead of stopping
				continue
		
		logger.info("\n" + "="*60)
		logger.info("PIPELINE COMPLETED")
		logger.info("="*60)
		# Summarize results (in-memory + fallback to CSV scan if needed)
		try:
			self._print_combination_summary_simple(combination_statuses)
		except Exception as e:  # noqa: BLE001
			logger.warning(f"Failed to print combination summary from in-memory data: {e}; attempting CSV fallback")
			try:
				self._print_combination_summary_simple(self._load_statuses_from_csv())
			except Exception as csv_e:  # noqa: BLE001
				logger.error(f"Failed to produce combination summary from CSV: {csv_e}")

	def _print_combination_summary_simple(self, statuses: Dict[str, str]) -> None:
		"""Print counts and for failures: combination id, failing tool name, and its log file."""
		if not statuses:
			logger.info("No combinations executed.")
			return
		failure_ids = [c for c, s in statuses.items() if s == "failure"]
		success_ct = len(statuses) - len(failure_ids)
		logger.info(f"Combinations: success={success_ct} failure={len(failure_ids)}")
		if not failure_ids:
			return
		logger.info("Failed combinations (id -> tool :: log):")
		for combo in failure_ids:
			tool_name, log_path = self._locate_failure_log(combo)
			if log_path:
				logger.info(f"  {combo}: {tool_name or '?'} :: {log_path}")
			else:
				logger.info(f"  {combo}: log not found")

	def _locate_failure_log(self, combination: str) -> Tuple[Optional[str], Optional[str]]:
		"""Find a failing tool log and extract tool name.

		Returns (log_path, tool_name). Tool name is inferred from filename pattern
		<stage>_<tool-name>.log written by _copy_tool_logs.
		"""
		logs_root = self.settings.results_dir / "tool_logs" / combination
		if not logs_root.exists():
			return None, None
		best: Optional[Path] = None
		best_score = -1  # prefer logs containing 'error'
		for log_file in logs_root.glob("*.log"):
			try:
				text = log_file.read_text(errors="ignore")
			except Exception:
				text = ""
			score = 1 if "error" in text.lower() else 0
			if score > best_score:
				best_score = score
				best = log_file
		if not best:
			return None, None
		# Extract tool name: file pattern '<stage>_<toolname>.log'
		tool_name = best.stem.split('_', 1)[1] if '_' in best.stem else best.stem
		return str(best), tool_name

	def _load_statuses_from_csv(self) -> Dict[str, str]:
		csv_path = self.settings.results_dir / "results_combinations.csv"
		statuses: Dict[str, str] = {}
		if not csv_path.exists():
			return statuses
		try:
			with open(csv_path, "r") as f:
				head = f.readline().strip().split(",")
				try:
					combo_idx = head.index("combination")
					status_idx = head.index("combination_status")
				except ValueError:
					return statuses
				for line in f:
					parts = line.strip().split(",")
					if len(parts) <= max(combo_idx, status_idx):
						continue
					statuses[parts[combo_idx]] = parts[status_idx]
		except Exception:
			pass
		return statuses

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
			combinations[key] = Combination(key, self.settings, self.dataset_manager, stage_combo_dict)

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
				self._execute_stage(combination, stage, combination_obj, context)
			
			# Evaluate final model
			duration = time.time() - context["start_time"]
			self._evaluate_and_log_combination(combination, combination_obj, context, duration, status="success")
			
			logger.info(f"Completed combination: {combination}")
			
		except Exception as e:
			logger.error(f"Combination {combination} failed: {e}")
			# Record failing tool info captured in context
			failed_tool = context.get("last_failed_tool_name")
			failed_log = context.get("last_failed_tool_artifact_log")
			self._failure_logs[combination] = (failed_tool, failed_log)
			# Log failure row if not already logged
			if not context.get("combination_logged"):
				try:
					duration = time.time() - context["start_time"]
					# Provide empty metrics dict (ResultLogger will fill defaults)
					self.logger.log_combination(
						combination,
						tools_by_stage=context["toolnames_by_stage"],
						dataset_name=self.settings.config.dataset.name,
						dataset_type=self.pipeline_dataset_type,
						acc={},
						duration=duration,
						status="failure"
					)
					context["combination_logged"] = True
				except Exception as log_err:  # noqa: BLE001
					logger.warning(f"Failed to log failed combination row for {combination}: {log_err}")
			raise

	def _initialize_combination_context(self, combination: str) -> Dict:
		"""Initialize context for combination execution"""
		#open dataset metadata file 
		target_file = self.dataset_manager.dataset_meta_file
		if not target_file.exists():
			logger.error(f"Dataset metadata file not found: {target_file}")
			raise FileNotFoundError(f"Dataset metadata file not found: {target_file}")
		#read contents to use as cache seed
		with open(target_file, 'r') as f:
			cache_seed = f.read()
		context = {
			"current_input": self.pipeline_dataset_dir,
			"dataset_dir": self.pipeline_dataset_dir,
			"start_time": time.time(),
			"toolnames_by_stage": {stage.value: [] for stage in Stage},
			# Retained for backward compatibility referencing logs if needed
			"cache_key": hashlib.sha256(cache_seed.encode()).hexdigest() + "_init",
		}
		if self._use_artifact_cache:
			# Seed parent hashes with dataset + model hashes for stability & reuse across runs
			variant_hash = self.artifact_cache.dataset_hash(target_file, self.pipeline_dataset_type)
			model_hash = self.artifact_cache.model_hash(
				getattr(self.settings, "config_model_path", None),
				getattr(self.settings, "model_params", {})
			)
			context["artifact_parents"] = [variant_hash, model_hash]
		return context

	def _execute_stage(self, combination: str, stage: str, combination_obj: Combination, context: Dict) -> List[Path]:
		"""Execute all tools in a specific stage"""
		tools_for_stage = combination_obj.tools_by_stage.get(stage, [])
		logger.info(f"{combination} --- STAGE: {stage.upper()} ---")

		if not tools_for_stage:
			logger.debug(f"{combination}: No tools for stage '{stage}'. Skipping.")
			return []
		
		context["toolnames_by_stage"][stage] = [tool.name for tool in tools_for_stage]
		
		for tool in tools_for_stage:
			logger.info(f"[Tool: {tool.name}] Starting execution...")
			
			if self._should_skip_tool(tool, stage):
				logger.info(f"[+] {combination}: Skipping {tool.name}")
				continue
			
			tool_output_path = self._execute_single_tool(combination_obj, tool, stage, context)

			
			# Update context
			self._update_context_after_tool(context, stage, tool_output_path)

			# Log outputs to json for combination
			combination_obj.log_tool_output_dir_path(combination, stage, tool.name, tool_output_path)
			
			
			# Copy tool logs (artifact-aware)
			self._copy_tool_logs(combination, tool, stage, tool_output_path)
			
			logger.info(f"{combination}: Tool '{tool.name}' completed successfully.")
		
		return

	def _should_skip_tool(self, tool: ToolConfig, stage: str) -> bool:
		"""Determine if a tool should be skipped"""
		# Allow noop to run in pre_training so that it can register dataset provenance.
		# Still skip noop in post_training and deployment to avoid redundant container runs.
		return tool.name == "noop" and stage in ["post_training", "deployment"]

	def _execute_single_tool(self, combination_obj: Combination, tool: ToolConfig, stage: str, context: Dict) -> Path:
		"""Execute a single tool with caching and GPU management.

		If experimental_artifact_cache flag enabled, use content-addressable artifacts;
		otherwise retain legacy chained cache behavior.
		"""

		# New artifact cache path
		parent_hashes = context.setdefault("artifact_parents", [])
		tool_id = self.artifact_cache.tool_identity_hash(tool)
		node_hash = self.artifact_cache.node_hash(parent_hashes, tool_id)
		node_dir = self.artifact_cache.path_for(node_hash)
		output_dir = node_dir / "output"
		success_marker = node_dir / ".success"

		if self.artifact_cache.exists(node_hash) and self.settings.use_cache:
			logger.info(f"[ARTIFACT CACHE HIT] {tool.name} ({stage}) -> {node_hash[:12]}")
			parent_hashes.append(node_hash)
			# Record mapping for UI even on cache hit
			self._record_artifact_mapping(combination_obj.id, stage, tool.name, node_hash, cache_hit=True)
			return output_dir

		logger.info(f"[ARTIFACT CACHE MISS] {tool.name} ({stage}) -> {node_hash[:12]}")
		lock = self.artifact_cache.lock(node_hash)
		try:
			if self.artifact_cache.exists(node_hash) and self.settings.use_cache:
				logger.info(f"[ARTIFACT CACHE LATE HIT] {tool.name}")
				parent_hashes.append(node_hash)
				return output_dir

			# Ensure a clean directory (handle prior failed attempt archiving)
			if node_dir.exists() and not success_marker.exists():
				failed_marker = node_dir / ".failed"
				if failed_marker.exists():
					# Archive previous failed attempt to preserve logs & outputs
					failed_attempts_dir = node_dir / "failed_attempts"
					failed_attempts_dir.mkdir(exist_ok=True)
					import time as _time
					archive_dir = failed_attempts_dir / _time.strftime("%Y%m%d-%H%M%S")
					archive_dir.mkdir()
					for item in ["output", "tool_logs", "failure_reason.txt"]:
						p = node_dir / item
						if p.exists():
							shutil.move(str(p), str(archive_dir / p.name))
					# Remove marker to allow fresh retry
					failed_marker.unlink(missing_ok=True)
				else:
					shutil.rmtree(node_dir)
			output_dir.mkdir(parents=True, exist_ok=True)

			gpu_id = None
			tool_output_path = output_dir  # default for failure path
			try:
				gpu_id = self.gpu_allocator.allocate_gpu()
				tool_runner = ToolRunner(
					self.settings, tool, stage,
					context,
					combination_obj,
					output_path=output_dir,
					gpu_id=gpu_id
				)
				tool_output_path, duration = tool_runner.run_tool()
			except Exception as e:
				# Mark node failure & persist reason
				failure_marker = node_dir / ".failed"
				failure_marker.touch()
				try:
					(node_dir / "failure_reason.txt").write_text(str(e))
				except Exception:
					pass
				# Copy logs for inspection into results
				try:
					self._copy_tool_logs(combination_obj.id, tool, stage, tool_output_path)
				except Exception:
					pass
				# Track failure info (artifact log path preferred)
				artifact_log = node_dir / "tool_logs" / f"{stage}_{tool.name.replace(' ', '_')}.log"
				context["last_failed_tool_name"] = tool.name
				context["last_failed_tool_artifact_log"] = str(artifact_log) if artifact_log.exists() else None
				logger.error(f"[ARTIFACT CACHE FAIL] {tool.name} ({stage}) -> {node_hash[:12]}: {e}")
				raise
			finally:
				if gpu_id is not None:
					self.gpu_allocator.release_gpu(gpu_id)

			# Build manifest (lightweight)
			manifest = {
				"node_hash": node_hash,
				"tool_identity": tool_id,
				"tool_name": tool.name,
				"parents": parent_hashes.copy(),
				"stage": stage,
				"duration_sec": duration,
				"files": []
			}
			for f in output_dir.rglob("*"):
				if f.is_file():
					try:
						manifest["files"].append({"rel": f.relative_to(node_dir).as_posix(), "size": f.stat().st_size})
					except Exception:
						pass
			self.artifact_cache.write_success(node_hash, manifest)
			parent_hashes.append(node_hash)
			# Record mapping for UI
			self._record_artifact_mapping(combination_obj.id, stage, tool.name, node_hash, cache_hit=False)
			return output_dir
		finally:
			lock.release()

	# Removed legacy _run_tool_with_caching/_run_tool_fresh methods

	def _update_context_after_tool(self, context: Dict, stage: str, tool_output_path: Path) -> None:
		"""Update execution context after tool completion"""
		if tool_output_path.is_dir() and (tool_output_path / "model.pt").exists():
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

	def _copy_tool_logs(self, combination: str, tool: ToolConfig, stage: str, tool_output_path: Path) -> None:
		"""Copy tool logs from artifact node directory (output/../tool_logs)."""
		# Logs are written alongside output/.. (see ToolRunner)
		logs_dir = tool_output_path.parent / "tool_logs"
		log_file = logs_dir / f"{stage}_{tool.name.replace(' ', '_')}.log"
		result_log = self.settings.results_dir / "tool_logs" / combination / f"{stage}_{tool.name.replace(' ', '_')}.log"
		if log_file.exists():
			copy_or_link_log(str(log_file), str(result_log), method="copy")
		else:
			logger.debug(f"{combination}: No log file found for {tool.name} at {log_file}")

	def _evaluate_and_log_combination(self, combination: str, combination_obj: Combination, 
								 context: Dict, duration: float, status: str) -> None:
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
				combination_id=combination,
				combination_output=combination_obj.combo_output_dir
			)
			
			final_acc = model_evaluator.evaluate_model(str(self.pipeline_dataset_dir))
			
			self.logger.log_combination(
				combination,
				tools_by_stage=context["toolnames_by_stage"],
				dataset_name=self.settings.config.dataset.name,
				dataset_type=self.pipeline_dataset_type,
				acc=final_acc,
				duration=duration,
				status=status
			)
			
			logger.info(f"{combination}: Evaluation completed: {final_acc}")
			# Mark combination logged to avoid duplicate failure logging
			context["combination_logged"] = True
			
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