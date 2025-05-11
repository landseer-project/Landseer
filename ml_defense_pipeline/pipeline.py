"""
Main pipeline orchestration for ML Defense Pipeline
"""
import logging
import os
from typing import Optional

from config import PipelineConfig
from docker_manager import DockerManager
from dataset_manager import DatasetManager
from tool_runner import ToolRunner
from model_evaluator import ModelEvaluator
from logging_manager import LoggingManager

logger = logging.getLogger("defense_pipeline")


class DefensePipeline:

    def __init__(self, config_path: Optional[str] = None):

        LoggingManager.setup_logging()
        self.config_manager = PipelineConfig(config_path)
        self.combinations = self.config_manager.get_combinations()
        self.docker_manager = DockerManager()
        self.dataset_manager = DatasetManager()
        self.tool_runner = ToolRunner(self.docker_manager)
        self.model_evaluator = ModelEvaluator()

    def run(self):
        """Execute the pipeline based on the configuration"""
        datasets = self.config_manager.config.get("dataset")
        dataset_name = next(iter(datasets), None)
        dataset_info = self.config_manager.get_dataset_info()

        if not dataset_name:
            logger.error("No dataset specified in configuration")
            return

        dataset_dir = self.dataset_manager.prepare_dataset(
            dataset_name, dataset_info)
        logger.info(
            f"Using dataset '{dataset_name}' in directory: {dataset_dir}")

        current_input = dataset_dir

        #run a combination of tools iterate over the combinations
        for combination in self.combinations:
            logger.info(f"Running combination: {combination}")
            self.run_combination(combination)
            logger.info(f"Completed combination: {combination}")
            logger.info("-" * 50)
            logger.info(f"Pipeline completed successfully for combination: {combination}")
            logger.info("-" * 50)
        logger.info("All combinations completed successfully.")
        logger.info("-" * 50)
    
    def run_combination(self, combination):
        """Run a specific combination of tools"""
        for stage in ["pre_training", "during_training", "post_training"]:
            # get tools for this combination for the current stage
            tools = self.config_manager.get_tools_for_combination(combination, stage)

            if not tools:
                logger.info(
                    f"No tools configured for stage '{stage}'. Skipping.")
                continue

            logger.info(f"Starting stage '{stage}' with {len(tools)} tool(s)")

            for tool in tools:
                try:
                    output_path = self.tool_runner.run_tool(
                        tool=tool,
                        stage=stage,
                        dataset_dir=dataset_dir,
                        input_path=current_input
                    )

                    current_input = output_path
                    if stage == "pre_training":
                        dataset_dir = current_input
                        #import ipdb
                        #ipdb.set_trace()
                        print("Dataset directory:", dataset_dir)
                        logger.info(
                            f"Updated dataset directory: {dataset_dir}")
                        # exit(0)
                    logger.info(
                        f"Tool '{tool['tool_name']}' completed successfully.")
                except Exception as e:
                    logger.error(f"Tool '{tool['tool_name']}' failed: {e}")
                    raise

            logger.info(f"Completed stage '{stage}'")

        final_model_path = current_input
        final_dataset_path = os.path.join(dataset_dir, "final_dataset.npy")
        baseline_model_path = os.path.join(dataset_dir, "baseline_model.pt")
        if not os.path.exists(baseline_model_path):
            logger.info("Training baseline model for comparison...")
            baseline_acc = self.model_evaluator.train_baseline_model(
                final_dataset_path, baseline_model_path, device=self.config_manager.device)
        else:
            logger.info("Using existing baseline model for comparison...")
            baseline_acc = self.model_evaluator.evaluate_clean(
                baseline_model_path, final_dataset_path, device=self.config_manager.device)

        logger.info("Evaluating final model...")
        final_acc = self.model_evaluator.evaluate_model(
            final_model_path, final_dataset_path)

        logger.info("-" * 50)
        logger.info("PIPELINE EVALUATION RESULTS")
        logger.info("-" * 50)
        logger.info(f"Baseline model accuracy: {baseline_acc:.4f}")
        logger.info(f"Defended model accuracy: {final_acc:.4f}")
        logger.info(f"Difference: {final_acc - baseline_acc:.4f}")
        logger.info("-" * 50)

        print(f"\nPipeline completed successfully!")
        print(f"Baseline model accuracy: {baseline_acc:.4f}")
        print(f"Defended model accuracy: {final_acc:.4f}")
        print(f"Improvement: {final_acc - baseline_acc:.4f}")
