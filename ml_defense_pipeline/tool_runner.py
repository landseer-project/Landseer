"""
Tool execution for ML Defense Pipeline
"""
import logging
import os
from pathlib import Path
from typing import Dict
import shutil

from docker_manager import DockerManager

logger = logging.getLogger("defense_pipeline")


class ToolRunner:

    def __init__(self, docker_manager: DockerManager):
        self.docker_manager = docker_manager
        self.scripts_dir = Path("./scripts")
        self.scripts_dir.mkdir(exist_ok=True)

    def run_tool(self, tool: Dict, stage: str, dataset_dir: str, input_path: str) -> str:
        tool_name = tool["tool_name"]
        output_path = tool.get("output_path", f"output/{stage}/{tool_name}")
        command = tool["docker"]["command"]
        image_name = tool["docker"]["image"]

        output_dir_path = os.path.join(
            os.path.join(os.path.abspath("data"), output_path))
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path, exist_ok=True)
        print("Output directory:", output_dir_path)

        env = {}

        data_dir = self.merge_directories(input_path, dataset_dir)

        tool_args = tool["docker"].get("command", "")
        if stage != "pre_training":
            config_script = tool["docker"].get(
                "config_script", "config_model.py")
            config_script_path = os.path.abspath(config_script)
            self._ensure_config_exists(config_script_path)
            volumes = {
                os.path.abspath(data_dir): {"bind": "/data", "mode": "ro"},
                os.path.abspath(output_dir_path): {"bind": "/output", "mode": "rw"},
                config_script_path: {"bind": "/app/config_model.py", "mode": "ro"},
            }
            print("Mounting config file to /app:",
                  os.path.abspath(str(config_script)))
        else:
            volumes = {
                os.path.abspath(data_dir): {"bind": "/data", "mode": "ro"},
                os.path.abspath(output_dir_path): {"bind": "/output", "mode": "rw"},
            }
        print("Mounting input dir:", os.path.abspath(data_dir))
        print("Mounting output dir:", os.path.abspath(output_dir_path))

        tool_args = (f"{tool_args} --output /output")

        command = (f"{tool_args}")

        logger.info(
            f"Running tool '{tool_name}' in stage '{stage}' using image '{image_name}'... with command {command}")

        exit_code, logs = self.docker_manager.run_container(
            image_name=image_name,
            command=command,
            environment=env,
            volumes=volumes
        )

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

    def _ensure_config_exists(self, script_path: str):
        """
        Ensure the required script exists, creating a template if needed

        Args:
            script_name: Name of the script file
        """

        if not os.path.exists(script_path):
            logger.error(f"Script {script_path} not found, cannot continue")
            exit(1)

    def merge_directories(self, input_path: str, dataset_dir: str) -> str:
        # check if input_path and dataset_dir is same path

        input_dir = os.path.abspath("data") + "/" + "temp_input"
        print("Merging directories:", input_path, dataset_dir)

        if not os.path.exists(input_dir):
            os.makedirs(input_dir, exist_ok=True)
        input_path = os.path.abspath(input_path)
        dataset_dir = os.path.abspath(dataset_dir)
        if os.path.isdir(input_path) and os.path.exists(dataset_dir):
            for file in os.listdir(input_path):
                file_path = os.path.join(input_path, file)
                if os.path.isfile(file_path):
                    shutil.copy(file_path, input_dir)
            print(f"Copying file from {dataset_dir} to {input_dir}")
            if os.path.abspath(dataset_dir) == os.path.abspath(input_path):
                print("Input path and dataset path are same")
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
