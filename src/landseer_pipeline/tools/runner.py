"""
Tool execution for ML Defense Pipeline
"""
import logging
import time
import os
from pathlib import Path
from typing import Dict
import shutil
import json
import errno
from landseer_pipeline.container_handler.factory import get_container_runner
from landseer_pipeline.utils import ResultLogger
from landseer_pipeline.utils.files import merge_directories
from landseer_pipeline.utils.temp_manager import temp_manager
from landseer_pipeline.utils.auxiliary import AuxiliaryFileManager
from dataclasses import dataclass
import torch

logger = logging.getLogger(__name__)

class ToolRunner:
    tool_name: str
    tool_stage: str

    def __init__(self, settings, ToolConfig, stage, context, combination_obj, output_path, gpu_id):
        self.settings = settings
        self.tool_config = ToolConfig
        self.context = context
        self.combination_obj = combination_obj
        self.combination_id = combination_obj.id
        self.tool_name = self.tool_config.name
        self.tool_stage = stage
        self.input_path = self.create_input_directory()
        output_path = f"{output_path}"
        #if output_path does not exist, create it
        self.output_path =  Path(output_path)
        print(f"Output path: {self.output_path}")
        self.docker_manager = get_container_runner(self.settings)
        self.gpu_id = gpu_id
        
        self.auxiliary_manager = AuxiliaryFileManager(self.output_path.parent)
        self.model_script = self._resolve_model_script()
        print(f"{combination_obj.id}: Initialised {self.tool_stage} toolrunner for {self.tool_name}:\n\tUsing GPU: {gpu_id}\n\tInput dir: {self.input_path}\n\tOutput dir: {self.output_path}\n\tModel script: {self.model_script}")

    def _provenance_file(self) -> Path:
        return self.combination_obj.combo_output_dir / "fin_output_paths.json"

    def _load_provenance(self) -> Dict[str, Dict]:
        prov_file = self._provenance_file()
        if prov_file.exists():
            try:
                return json.loads(prov_file.read_text())
            except Exception:
                logger.warning(f"Failed to parse provenance file: {prov_file}")
        return {}

    def _select_files(self, provenance: Dict[str, Dict]) -> Dict[str, Path]:
        names = provenance.keys()
        selected = []
        if getattr(self.tool_config, "required_inputs", None):
            for req in self.tool_config.required_inputs:
                if req in names:
                    selected.append(req)
                elif req == "model_composite.pt" and "model.pt" in names:
                    selected.append("model.pt")
                else:
                    logger.warning(f"{self.combination_id}: Required input '{req}' missing for tool {self.tool_name}")
        else:
            selected = list(names)
        return {name: Path(provenance[name]["source_path"]) for name in selected if name in provenance}

    def create_input_directory(self) -> Path:
        """Create a temporary input directory populated with selected files from provenance.
        Falls back to current_input (dataset) if no provenance yet (first stage)."""
        provenance = self._load_provenance()
        if not provenance:
            return merge_directories(self.context["current_input"], self.context["dataset_dir"])
        selected = self._select_files(provenance)
        staging_root = self.settings.results_dir / "staged_inputs" / self.combination_id
        staging_dir = staging_root / f"{self.tool_stage}_{self.tool_name.replace(' ', '_')}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        for name, src in selected.items():
            if not src.exists() or not src.is_file():
                continue
            dest = staging_dir / Path(name).name
            try:
                os.link(src, dest)
            except OSError as e:
                if e.errno in (errno.EXDEV, errno.EPERM, errno.EACCES):
                    shutil.copy2(src, dest)
                else:
                    raise
        logger.debug(f"{self.combination_id}/{self.tool_name}: Prepared staged input dir {staging_dir} with {len(list(staging_dir.iterdir()))} files")
        return staging_dir

    def _resolve_model_script(self) -> str:
        """Determine which model script to mount: tool override (deprecated) or top-level model."""
        override = self.tool_config.container.config_script
        top_level = getattr(self.settings, "config_model_path", None)
        if override and top_level and os.path.abspath(override) != os.path.abspath(top_level):
            logger.warning(f"Tool '{self.tool_name}' uses per-tool config_script override: {override}")
            return os.path.abspath(override)
        if override and not top_level:
            return os.path.abspath(override)
        if top_level:
            return os.path.abspath(top_level)
        return None

    def run_tool(self) -> str:
        tool_name = self.tool_name
        stage = self.tool_stage
        dataset_dir = self.context["dataset_dir"]
        input_path = self.input_path
        output_dir_path = self.output_path
        command = self.tool_config.container.command
        image_name = self.tool_config.container.image
        
        logger.info(f"{self.combination_id}/{tool_name}: Image to run {image_name}")
        logger.debug(f"{self.combination_id}/{tool_name}: Output path: {output_dir_path}")

        env ={}
        print(f"Before environment setting gpus - {torch.cuda.device_count()}")
        #env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env["NVIDIA_VISIBLE_DEVICES"] = str(self.gpu_id)
        #env["NVIDIA_DRIVER_CAPABILITIES"] = "compute,utility"
        env["NVIDIA_DRIVER_CAPABILITIES"] = "all"
        logger.debug(f"{self.combination_id}/{tool_name}: Setting NVIDIA_VISIBLE_DEVICES to {env['NVIDIA_VISIBLE_DEVICES']}")
        print(f"After environment setting gpus - {torch.cuda.device_count()}")

        data_dir = None  # Initialize to avoid UnboundLocalError
        auxiliary_staging_dir = None  # Track auxiliary staging for cleanup
        
        try:
            data_dir = self.input_path
            logger.debug(f"{self.combination_id}/{tool_name}: Temporary input directory created at: {data_dir}")
            
            # Prepare auxiliary files if configured
            if self.tool_config.has_auxiliary_files():
                auxiliary_staging_dir = self.auxiliary_manager.prepare_auxiliary_directory(
                    tool_name, self.tool_config.auxiliary_files
                )
                logger.info(f"{self.combination_id}/{tool_name}: Prepared auxiliary files in {auxiliary_staging_dir}")
            
            # Ensure host paths are absolute for Docker volume mounts
            host_data_dir = os.path.abspath(str(data_dir))
            host_output_dir = os.path.abspath(str(output_dir_path))

            if stage != "pre_training":
                volumes = {
                    host_data_dir: {"bind": "/data", "mode": "ro"},
                    host_output_dir: {"bind": "/output", "mode": "rw"},
                }
                if self.model_script:
                    volumes[self.model_script] = {"bind": "/app/config_model.py", "mode": "ro"}
                    logger.debug(f"{self.combination_id}/{tool_name}: Mounting model script: {self.model_script}")
            else:
                volumes = {
                    host_data_dir: {"bind": "/data", "mode": "ro"},
                    host_output_dir: {"bind": "/output", "mode": "rw"},
                }
                if self.model_script:
                    volumes[self.model_script] = {"bind": "/app/config_model.py", "mode": "ro"}
                    logger.debug(f"{self.combination_id}/{tool_name}: Mounting model script (pre_training): {self.model_script}")
            
            if auxiliary_staging_dir:
                aux_volumes = self.auxiliary_manager.get_standard_volume_mount(auxiliary_staging_dir)
                volumes.update(aux_volumes)
                logger.info(f"{self.combination_id}/{tool_name}: Added standardized auxiliary mount: {aux_volumes}")
            
            if self.tool_config.has_auxiliary_files():
                try:
                    declarative_volumes = self.tool_config.get_auxiliary_volume_mounts()
                    volumes.update(declarative_volumes)
                    if declarative_volumes:
                        logger.info(f"{self.combination_id}/{tool_name}: Added declarative auxiliary mounts: {declarative_volumes}")
                except Exception as e:
                    logger.warning(f"{self.combination_id}/{tool_name}: Failed to add declarative auxiliary mounts: {e}")
            
            tool_args = self.tool_config.container.command
            command = f"{tool_args} --output /output"
            logger.info(f"{self.combination_id}/{tool_name}: Container command: {command}")
            start = time.time()
            exit_code = None
            logs = None    
            exit_code, logs, container_info = self.docker_manager.run_container(
                image_name=image_name,
                command=command,
                environment=env,
                volumes=volumes,
                gpu_id=self.gpu_id,
                combination_id=self.combination_id
            )
        
            # Force container cleanup to release GPU
            #try:
            #    self.docker_manager.cleanup_container(container)
            #except Exception as cleanup_error:
            #    logger.warning(f"{combination_id}/{tool_name}: Container cleanup failed: {cleanup_error}")
            duration = time.time() - start
            # Write logs only if we have them
            if logs is not None:
                tool_log_path = os.path.join(
                    output_dir_path,"..", "tool_logs",
                f"{stage}_{tool_name.replace(' ', '_')}.log")
                os.makedirs(os.path.dirname(tool_log_path), exist_ok=True)
                with open(tool_log_path, "w") as f:
                    f.write(logs)
            
            if exit_code != 0:
                raise RuntimeError(
                f"{self.combination_id}/{tool_name}: Failed with exit code {exit_code}")
            logger.info(
                f"{self.combination_id}/{tool_name}: Completed successfully and output saved to {output_dir_path}")
        #exception of ctrl+c
        except KeyboardInterrupt:
            logger.error(f"{self.combination_id}/{tool_name}: Execution interrupted by user.")
            raise 
        finally:
            # Always cleanup temporary directory if it was created
            if data_dir is not None:
                logger.debug(f"{self.combination_id}/{tool_name}: Cleaning up temporary directory: {data_dir}")
                temp_manager.cleanup_temp_dir(data_dir)
            else:
                logger.debug(f"{self.combination_id}/{tool_name}: No temporary directory to clean up")

            # Cleanup auxiliary staging
            if auxiliary_staging_dir is not None:
                try:
                    self.auxiliary_manager.cleanup_staging(tool_name)
                    logger.debug(f"{self.combination_id}/{tool_name}: Cleaned up auxiliary staging")
                except Exception as e:
                    logger.warning(f"{self.combination_id}/{tool_name}: Failed to cleanup auxiliary staging: {e}")
        return output_dir_path, duration