import argparse
import docker
import logging
import sys
from enum import Enum
from typing import List, Optional

tool_dirs = {"XGBOD_forked", "WatermarkNN", "influence-release_fork"}

class Stage(Enum):
    PRE = "pre"
    DURING = "during"
    POST = "post"

class DockerToolRunner:
    def __init__(self, verbose: bool = False, use_registry: bool = True):
        self.client = docker.from_env()
        self.verbose = verbose
        self.use_registry = use_registry
        
        # Configure logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def pull_image(self, image_name: str) -> bool:
        """Pull Docker image from registry."""
        try:
            self.logger.info(f"Pulling image: {image_name}")
            self.client.images.pull(image_name)
            return True
        except docker.errors.ImageNotFound:
            self.logger.error(f"Image not found: {image_name}")
            return False
        except Exception as e:
            self.logger.error(f"Error pulling image {image_name}: {str(e)}")
            return False

    def run_tool(self, image_name: str, dataset_path: str, tool_args: Optional[List[str]] = None) -> bool:
        """Run a Docker tool on the dataset."""
        try:
            if self.use_registry and not self.pull_image(image_name):
                return False

            self.logger.info(f"Running tool: {image_name}")
            
            volumes = {
                dataset_path: {
                    'bind': '/data',
                    'mode': 'rw'
                }
            }

            command = tool_args if tool_args else []

            container = self.client.containers.run(
                image_name,
                command=command,
                volumes=volumes,
                detach=True
            )

            if self.verbose:
                for log in container.logs(stream=True, follow=True):
                    self.logger.debug(log.decode().strip())

            # wait for container to finish
            result = container.wait()
            container.remove()

            if result['StatusCode'] != 0:
                self.logger.error(f"Tool {image_name} failed with exit code {result['StatusCode']}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error running tool {image_name}: {str(e)}")
            return False

    def run_stage(self, stage: Stage, tools: List[dict]) -> bool:
        """Run all tools for a specific stage."""
        self.logger.info(f"Starting {stage.value} stage")
        
        for tool in tools:
            success = self.run_tool(
                tool['image'],
                tool['dataset_path'],
                tool.get('args')
            )
            if not success:
                self.logger.error(f"Failed during {stage.value} stage")
                return False
        
        self.logger.info(f"Completed {stage.value} stage")
        return True

def main():
    parser = argparse.ArgumentParser(description='Run ML defense tools on dataset')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--local', action='store_true', help='Use local images instead of registry')
    parser.add_argument('--stage', type=str, choices=['pre', 'during', 'post', 'all'],
                       default='all', help='Stage to run until')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')

    args = parser.parse_args()

    tools_config = {
        Stage.PRE: [
            {
                'image': 'xgbod:latest',
                'dataset_path': args.dataset,
                'args': ['--optimize']
            }
        ],
        Stage.DURING: [
            {
                'image': 'watermarknn:latest',
                'dataset_path': args.dataset,
                'args': ['--deep-scan']
            }
        ],
        Stage.POST: [
            {
                'image': 'influence-release:latest',
                'dataset_path': args.dataset,
                'args': ['--format', 'json']
            }
        ]
    }

    runner = DockerToolRunner(
        verbose=args.verbose,
        use_registry=not args.local
    )

    stages = []
    if args.stage == 'all':
        stages = list(Stage)
    else:
        target_stage = Stage(args.stage)
        stages = [stage for stage in Stage if stage.value <= target_stage.value]

    for stage in stages:
        if not runner.run_stage(stage, tools_config[stage]):
            sys.exit(1)

    print("All stages completed successfully")

if __name__ == "__main__":
    main()