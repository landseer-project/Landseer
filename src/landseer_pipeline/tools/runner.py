"""
Tool execution for ML Defense Pipeline
"""
import logging
import time
import os
from pathlib import Path
from typing import Dict
import shutil
from landseer_pipeline.docker_handler import DockerRunner
from landseer_pipeline.utils import ResultLogger
from landseer_pipeline.utils.files import merge_directories
from landseer_pipeline.utils.temp_manager import temp_manager
from landseer_pipeline.utils.auxiliary import AuxiliaryFileManager
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class ToolRunner:
    tool_name: str
    tool_stage: str

    def __init__(self, settings, ToolConfig, stage, dataset_dir, input_path, output_path, gpu_id):
        self.settings = settings
        self.tool_config = ToolConfig
        
        self.tool_name = self.tool_config.name
        self.tool_stage = stage
        self.dataset_dir = dataset_dir
        self.input_path = input_path
        output_path = f"{output_path}"
        #if output_path does not exist, create it
        self.output_path =  Path(output_path)
        print(f"Output path: {self.output_path}")
        self.docker_manager = DockerRunner(self.settings)
        self.gpu_id = gpu_id
        
        # Initialize auxiliary file manager
        self.auxiliary_manager = AuxiliaryFileManager(self.output_path.parent)
        self.model_script = self._resolve_model_script()

    def _resolve_model_script(self) -> str:
        """Determine which model script to mount: tool override (deprecated) or top-level model."""
        override = self.tool_config.docker.config_script
        top_level = getattr(self.settings, "config_model_path", None)
        if override and top_level and os.path.abspath(override) != os.path.abspath(top_level):
            logger.warning(f"Tool '{self.tool_name}' uses per-tool config_script override: {override}")
            return os.path.abspath(override)
        if override and not top_level:
            return os.path.abspath(override)
        if top_level:
            return os.path.abspath(top_level)
        return None

    def run_tool(self, combination_id) -> str:
        tool_name = self.tool_name
        stage = self.tool_stage
        dataset_dir = self.dataset_dir
        input_path = self.input_path
        output_dir_path = self.output_path
        command = self.tool_config.docker.command
        image_name = self.tool_config.docker.image
        
        logger.info(f"{combination_id}/{tool_name}: Image to run {image_name}")
        logger.debug(f"{combination_id}/{tool_name}: Output path: {output_dir_path}")

        env ={}
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        logger.debug(f"{combination_id}/{tool_name}: Setting CUDA_VISIBLE_DEVICES to {env['CUDA_VISIBLE_DEVICES']}")

        data_dir = None  # Initialize to avoid UnboundLocalError
        auxiliary_staging_dir = None  # Track auxiliary staging for cleanup
        
        try:
            data_dir = merge_directories(input_path, dataset_dir)
            logger.debug(f"{combination_id}/{tool_name}: Temporary input directory created at: {data_dir}")
            
            # Prepare auxiliary files if configured
            if self.tool_config.has_auxiliary_files():
                auxiliary_staging_dir = self.auxiliary_manager.prepare_auxiliary_directory(
                    tool_name, self.tool_config.auxiliary_files
                )
                logger.info(f"{combination_id}/{tool_name}: Prepared auxiliary files in {auxiliary_staging_dir}")
            
            if stage != "pre_training":
                # Use resolved model script if available
                volumes = {
                    data_dir: {"bind": "/data", "mode": "ro"},
                    os.path.abspath(output_dir_path): {"bind": "/output", "mode": "rw"},
                }
                if self.model_script:
                    volumes[self.model_script] = {"bind": "/app/config_model.py", "mode": "ro"}
                    logger.debug(f"{combination_id}/{tool_name}: Mounting model script: {self.model_script}")
            else:
                volumes = {
                    os.path.abspath(data_dir): {"bind": "/data", "mode": "ro"},
                    os.path.abspath(output_dir_path): {"bind": "/output", "mode": "rw"},
                }
                if self.model_script:
                    volumes[self.model_script] = {"bind": "/app/config_model.py", "mode": "ro"}
                    logger.debug(f"{combination_id}/{tool_name}: Mounting model script (pre_training): {self.model_script}")
            
            # Add auxiliary file volumes using hybrid approach
            if auxiliary_staging_dir:
                # Standardized auxiliary directory mount
                aux_volumes = self.auxiliary_manager.get_standard_volume_mount(auxiliary_staging_dir)
                volumes.update(aux_volumes)
                logger.info(f"{combination_id}/{tool_name}: Added standardized auxiliary mount: {aux_volumes}")
            
            # Add declarative auxiliary file mounts
            if self.tool_config.has_auxiliary_files():
                try:
                    declarative_volumes = self.tool_config.get_auxiliary_volume_mounts()
                    volumes.update(declarative_volumes)
                    if declarative_volumes:
                        logger.info(f"{combination_id}/{tool_name}: Added declarative auxiliary mounts: {declarative_volumes}")
                except Exception as e:
                    logger.warning(f"{combination_id}/{tool_name}: Failed to add declarative auxiliary mounts: {e}")
            
            tool_args = self.tool_config.docker.command
            command = f"{tool_args} --output /output"
            logger.info(f"{combination_id}/{tool_name}: Container command: {command}")
            start = time.time()
            exit_code = None
            logs = None    
            exit_code, logs, container = self.docker_manager.run_container(
                image_name=image_name,
                command=command,
                environment=env,
                volumes=volumes,
                gpu_id=self.gpu_id
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
                f"{combination_id}/{tool_name}: Failed with exit code {exit_code}")
            logger.info(
                f"{combination_id}/{tool_name}: Completed successfully and output saved to {output_dir_path}")
        #exception of ctrl+c
        except KeyboardInterrupt:
            logger.error(f"{combination_id}/{tool_name}: Execution interrupted by user.")
            raise 
        finally:
            # Always cleanup temporary directory if it was created
            if data_dir is not None:
                logger.debug(f"{combination_id}/{tool_name}: Cleaning up temporary directory: {data_dir}")
                temp_manager.cleanup_temp_dir(data_dir)
            else:
                logger.debug(f"{combination_id}/{tool_name}: No temporary directory to clean up")
            
            # Cleanup auxiliary staging
            if auxiliary_staging_dir is not None:
                try:
                    self.auxiliary_manager.cleanup_staging(tool_name)
                    logger.debug(f"{combination_id}/{tool_name}: Cleaned up auxiliary staging")
                except Exception as e:
                    logger.warning(f"{combination_id}/{tool_name}: Failed to cleanup auxiliary staging: {e}")    
        return output_dir_path, duration