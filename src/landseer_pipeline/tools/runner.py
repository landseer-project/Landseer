"""
Tool execution for ML Defense Pipeline
"""
import logging
import time
import os
from pathlib import Path
from typing import Dict
import shutil
from landseer_pipeline.config import ToolConfig
from landseer_pipeline.docker_handler import DockerRunner
from landseer_pipeline.utils import ResultLogger
from landseer_pipeline.utils.files import merge_directories
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
        self.output_path = os.path.abspath(output_path)
        print(f"Output path: {self.output_path}")
        self.docker_manager = DockerRunner(self.settings)
        self.gpu_id = gpu_id

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

        data_dir = merge_directories(input_path, dataset_dir)
        logger.debug(f"{combination_id}/{tool_name}: Temporary input directory created at: {data_dir}")
        if stage != "pre_training":
            config_script_path = self.tool_config.docker.config_script
            logging.debug(f"{combination_id}/{tool_name}: Config script path: {config_script_path}")
            volumes = {
                data_dir: {"bind": "/data", "mode": "ro"},
                os.path.abspath(output_dir_path): {"bind": "/output", "mode": "rw"},
                config_script_path: {"bind": "/app/config_model.py", "mode": "rw"},
            }
            logger.debug(f"{combination_id}/{tool_name}: Mounting config script: {config_script_path}")
        else:
            volumes = {
                os.path.abspath(data_dir): {"bind": "/data", "mode": "ro"},
                os.path.abspath(output_dir_path): {"bind": "/output", "mode": "rw"},
            }
        logger.debug(f"{combination_id}/{tool_name}: Volume bindings: {volumes}")

        tool_args = self.tool_config.docker.command
        command = (f"{tool_args} --output /output")

        logger.info(f"{combination_id}/{tool_name}: Container command: {command}")
        
        start = time.time()
        exit_code, logs = self.docker_manager.run_container(
            image_name=image_name,
            command=command,
            environment=env,
            volumes=volumes
        )
        duration = time.time() - start
        tool_log_path = os.path.join(
            output_dir_path,"..",
            "tool_logs",
            f"{stage}_{tool_name.replace(' ', '_')}.log")
        os.makedirs(os.path.dirname(tool_log_path), exist_ok=True)
        with open(tool_log_path, "w") as f:
            f.write(logs)
        
        logger.debug(f"{combination_id}/{tool_name}: Cleaning up temporary directory: {data_dir}")
        shutil.rmtree(data_dir, ignore_errors=True)

        if exit_code != 0:
            raise RuntimeError(
                f"{combination_id}/{tool_name}: Failed with exit code {exit_code}")
        else:
            logger.info(
                f"{combination_id}/{tool_name}: Completed successfully and output saved to {output_dir_path}")
        return output_dir_path, duration