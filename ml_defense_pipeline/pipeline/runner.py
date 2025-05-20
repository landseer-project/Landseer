import logging
import os
from pipeline import Stage, PipelineStructure

logger = logging.getLogger("defense_pipeline")

class DefensePipeline:

    def run_combination(self, combination, dataset_dir):
        """Run a specific combination of tools"""
        current_input = dataset_dir
        stages = [stage.value for stage in Stage]
        for stage in stages:
            tools = self.utils.get_tools_for_combination(combination, stage)
            if not tools:
                logger.info(
                    f"No tools configured for stage '{stage}'. Skipping.")
                continue
            logger.info(f"Starting stage '{stage}' with {len(tools)} tool(s)")
            for tool in tools:
                logger.info(f"Running tool '{tool.name}'...")
                if tool.name == "noop" and stage == "post_training":
                    logger.info(
                        f"[-] Skipping '{tool.name}' in stage '{stage}'.")
                    continue
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
                    logger.info(
                        f"Updated dataset directory: {dataset_dir}")
                    # exit(0)
                logger.info(
                        f"Tool '{tool.name}' completed successfully.")
            logger.info(f"Completed stage '{stage}'")
        final_model_path = current_input
        final_dataset_path = os.path.join(dataset_dir)
        # baseline_model_path = os.path.join(dataset_dir, "baseline_model.pt")
        # if not os.path.exists(baseline_model_path):
        #    logger.info("Training baseline model for comparison...")
        #    baseline_acc = self.model_evaluator.train_baseline_model(
        #        final_dataset_path, baseline_model_path, device=self.config_manager.device)
        # else:
        # logger.info("Using existing baseline model for comparison...")
        # baseline_acc = self.model_evaluator.evaluate_clean(baseline_model_path, final_dataset_path, device=self.config_manager.device)

        logger.info("Evaluating final model...")
        final_acc = self.model_evaluator.evaluate_model(
            f"{final_model_path}/model.pt", final_dataset_path)
        self.store_results(
            combination=combination,
            dataset_name=self.config.dataset.name,
            dataset_dir=dataset_dir,
            final_model_path=final_model_path,
            final_acc=final_acc)
            

        logger.info("PIPELINE EVALUATION RESULTS")
        logger.info(f"Accuracy: {final_acc}")

        print(f"\nPipeline completed successfully!")
        print(f"Accuracy: {final_acc}")

class PipelineUtils(Stager):

    def __init__(self, combinations: str):
        self.combinations = combinations
        """Initialize configuration from file or interactive input"""
        
    def get_tools_for_stage(self, stage: str) -> List[Dict]:
        """Get the list of tools for a specific stage"""
        return self.config.get("pipeline",[]).get(stage, []).get("tools", [])
    
    def get_tools_for_combination(self, combination: str, stage: str) -> List[Dict]:
        """Get the list of tools for a specific combination and stage"""
        if combination not in self.combinations:
            raise ValueError(f"Combination '{combination}' not found.")
        tools = self.combinations[combination].get(stage, {})
        return [tools] if tools else []