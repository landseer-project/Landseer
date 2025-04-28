"""
Tool execution for ML Defense Pipeline
"""
import logging
import os
from pathlib import Path
from typing import Dict

from docker_manager import DockerManager

logger = logging.getLogger("defense_pipeline")

class ToolRunner:
    """Handles the execution of defense tools"""
    
    def __init__(self, docker_manager: DockerManager):
        """
        Initialize with a Docker manager
        
        Args:
            docker_manager: Manager for Docker operations
        """
        self.docker_manager = docker_manager
        self.scripts_dir = Path("./scripts")
        self.scripts_dir.mkdir(exist_ok=True)
    
    def run_tool(self, tool: Dict, stage: str, dataset_dir: str, input_path: str) -> str:
        """
        Run a defense tool in a Docker container
        
        Args:
            tool: Tool configuration
            stage: Pipeline stage (pre_training, during_training, post_training)
            dataset_dir: Directory containing dataset and intermediate files
            input_path: Path to the input file for this tool
            
        Returns:
            Path to the output file produced by the tool
        """
        tool_name = tool["tool_name"]
        output_path = tool["output_path"]
        command = tool["docker"]["command"]

        image_name = self.docker_manager.build_image(tool)
        tool["_image"] = image_name
        
        # Determine pre/post-processor scripts
        input_type, input_format = tool.get("input", ["dataset", "h5"])
        pre_script = f"{stage}_{input_type}_h5_preprocessor.py"
        post_script = f"{stage}_{input_type}_h5_postprocessor.py"
        
        # Ensure scripts exist
        self._ensure_script_exists(pre_script)
        self._ensure_script_exists(post_script)
        print("Input Path:", input_path )
        
        # Set up environment variables
        env = {
            "DATASET_DIR": "/data",
            "INPUT_IR": f"/{input_path}",
            "OUTPUT_IR": f"/data/{os.path.basename(output_path)}",
            "TOOL_INPUT_TYPE": input_type,
            "TOOL_INPUT_FORMAT": input_format
        }
        
        # Set up volume mounts
        tool_args = tool["docker"].get("command", "")
        volumes = {
            os.path.abspath("data"): {"bind": "/data", "mode": "rw"},
            os.path.abspath(str(self.scripts_dir)): {"bind": "/scripts", "mode": "ro"}
        }
        print("Mounting scripts dir:", os.path.abspath(str(self.scripts_dir)))
        print("Mounting dataset dir:", os.path.abspath(dataset_dir))
        
        command =  (f"python /scripts/{pre_script} && {tool_args} && python /scripts/{post_script}")
        
        logger.info(f"Running tool '{tool_name}' in stage '{stage}' using image '{image_name}'... with command {command}")
        
        # Run the container
        exit_code, logs = self.docker_manager.run_container(
            image_name=image_name,
            command=command,
            environment=env,
            volumes=volumes
        )
        print(logs)
        
        if exit_code != 0:
            logger.error(f"Tool {tool_name} failed with exit code {exit_code}. Logs:\n{logs}")
            raise RuntimeError(f"Tool {tool_name} failed with exit code {exit_code}")
        else:
            logger.info(f"Tool {tool_name} completed successfully.")
            logger.debug(f"Tool logs:\n{logs}")
        
        return os.path.join(dataset_dir, output_path)
    
    def _ensure_script_exists(self, script_name: str):
        """
        Ensure the required script exists, creating a template if needed
        
        Args:
            script_name: Name of the script file
        """
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            logger.error(f"Script {script_name} not found, cannot continue")
            exit(1)