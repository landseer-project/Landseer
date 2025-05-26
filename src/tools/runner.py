"""
Tool execution for ML Defense Pipeline
"""
import logging
import os
from pathlib import Path
from typing import Dict
import shutil
from config import ToolConfig
from docker_handler import DockerRunner
from dataclasses import dataclass

logger = logging.getLogger("defense_pipeline")

@dataclass
class ToolRunner:
    tool_name: str
    tool_stage: str

    def __init__(self, Settings):
        self.settings = Settings
        self.docker_manager = DockerRunner(self.settings)

    @property
    def from_config(ToolConfig):
        pass


    def run_tool(self, tool: ToolConfig, stage: str, dataset_dir: str, input_path: str) -> str:
        tool_name = tool.tool_name
        output_path = f"output/{stage}/{tool_name}"
        command = tool.docker.command
        image_name = tool.docker.image_name
        output_dir_path = os.path.join(os.path.join(os.path.abspath("data"), output_path))
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path, exist_ok=True)
        
        logger.debug(f"Tool name: {tool_name}")
        logger.debug(f"Stage: {stage}")
        logger.debug(f"Image name: {image_name}")
        logger.debug(f"Output directory: {output_dir_path}")

        logger.debug(f"Preparing volumes for tool {tool_name}")
        env = {}
        data_dir = self.merge_directories(input_path, dataset_dir)
        logger.debug(f"Data directory: {data_dir}")
        if stage != "pre_training":
            config_script_path = tool.docker.config_script
            config_script = os.path.basename(config_script_path)
            volumes = {
                data_dir: {"bind": "/data", "mode": "ro"},
                os.path.abspath(output_dir_path): {"bind": "/output", "mode": "rw"},
                config_script_path: {"bind": "/app/config_model.py", "mode": "rw"},
            }
            logger.debug(f"Mounting config file to /app: {os.path.abspath(str(config_script))}")
        else:
            volumes = {
                os.path.abspath(data_dir): {"bind": "/data", "mode": "ro"},
                os.path.abspath(output_dir_path): {"bind": "/output", "mode": "rw"},
            }
        logger.debug(f"Mounting input dir: {os.path.abspath(data_dir)}")
        logger.debug(f"Mounting output dir: {os.path.abspath(output_dir_path)}")
        logger.debug(f"Volumes mounted")

        tool_args = tool.docker.command
       
        command = (f"{tool_args} --output /output")
        logger.info(
            f"Running tool with command {command}")

        exit_code, logs = self.docker_manager.run_container(
            image_name=image_name,
            command=command,
            environment=env,
            volumes=volumes
        )
        
        logger.debug(f"Cleaning up temporary directories: {data_dir}")
        shutil.rmtree(data_dir, ignore_errors=True)
        if exit_code != 0:
            logger.error(
                f"Tool {tool_name} failed with exit code {exit_code}. Logs:\n{logs}")
            raise RuntimeError(
                f"Tool {tool_name} failed with exit code {exit_code}")
        else:
            logger.info(
                f"Tool {tool_name} completed successfully and output saved to {output_dir_path}")
            logger.debug(f"Tool logs:\n{logs}")
        return output_dir_path

    def merge_directories(self, input_path: str, dataset_dir: str) -> str:
        # check if input_path and dataset_dir is same path
        input_dir = os.path.abspath("data") + "/" + "temp_input"
        logger.debug(f"Merging directories: {input_path} and {dataset_dir}")

        if not os.path.exists(input_dir):
            os.makedirs(input_dir, exist_ok=True)
        input_path = os.path.abspath(input_path)
        dataset_dir = os.path.abspath(dataset_dir)
        if os.path.isdir(input_path) and os.path.exists(dataset_dir):
            for file in os.listdir(input_path):
                file_path = os.path.join(input_path, file)
                if os.path.isfile(file_path):
                    shutil.copy(file_path, input_dir)
            logger.debug(f"Copying file from {dataset_dir} to {input_dir}")
            if os.path.abspath(dataset_dir) == os.path.abspath(input_path):
                logger.debug(f"Input path and dataset path are same")
                return os.path.abspath(input_dir)
            for file in os.listdir(dataset_dir):
                file_path = os.path.join(dataset_dir, file)
                if os.path.isfile(file_path):
                    shutil.copy(file_path, input_dir)
        else:
            shutil.copy(input_path, input_dir)
            shutil.copy(dataset_dir, input_dir)
        # exit(0)
        return os.path.abspath(input_dir)
