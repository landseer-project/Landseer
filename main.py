#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import sys
import subprocess
from typing import Dict, List
try:
    import docker  
except ImportError:
    docker = None  

logger = logging.getLogger("defense_pipeline")
logger.setLevel(logging.DEBUG)  

file_handler = logging.FileHandler("pipeline.log", mode='w')
file_handler.setLevel(logging.DEBUG)
file_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_fmt)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_fmt = logging.Formatter("%(levelname)s: %(message)s")
console_handler.setFormatter(console_fmt)
logger.addHandler(console_handler)

logger.info("Initialized logging. Log file: pipeline.log")

with open("/share/landseer/workspace-ayushi/Landseer/default_config.json", "r") as f:
    TOOLS_DB = json.load(f)

# ---------------------------- Helper Functions ----------------------------

def build_docker_image(tool: Dict) -> str:
    """
    Build or pull the Docker image for a given tool. Returns the image name (tag) to use.
    """
    docker_info = tool.get("docker", {})
    image_name = None
    try:
        if docker_info.get("Dockerfile"):
            dockerfile_path = docker_info["Dockerfile"]
            context_dir = os.path.dirname(dockerfile_path) if os.path.dirname(dockerfile_path) else "."
            # Tag the image using tool_name (stage included for uniqueness)
            image_name = f"{tool['tool_name'].lower()}_img"
            logger.info(f"Building Docker image for tool '{tool['tool_name']}' from {dockerfile_path}...")
            if docker:
                client = docker.from_env()
                # Build the image 
                client.images.build(path=context_dir, dockerfile=os.path.basename(dockerfile_path), tag=image_name)
            else:
                # Fallback to subprocess
                subprocess.run(["docker", "build", "-t", image_name, "-f", dockerfile_path, context_dir], check=True)
            logger.info(f"Successfully built image '{image_name}' for tool {tool['tool_name']}.")
        elif docker_info.get("image") != None and docker_info.get("image") != "":
            #[TODO]: build a new dockerfile which pulls this image and also puts in preprocessing and post processing scripts and calls them and calls the tool
            image_name = docker_info["image"]
            logger.info(f"Pulling Docker image for tool '{tool['tool_name']}': {image_name} ...")
            if docker:
                client = docker.from_env()
                client.images.pull(image_name)
            else:
                subprocess.run(["docker", "pull", image_name], check=True)
            logger.info(f"Successfully pulled image '{image_name}' for tool {tool['tool_name']}.")
        else:
            raise ValueError(f"No Docker image or Dockerfile specified for tool {tool['tool_name']}.")
    except Exception as e:
        logger.error(f"Failed to build/pull Docker image for tool {tool['tool_name']}: {e}")
        raise
    return image_name

def run_tool_in_container(tool: Dict, stage: str, dataset_dir: str, input_path: str) -> str:
    tool_name = tool['tool_name']
    image_name = tool.get("_image", None)  
    if not image_name:
        image_name = tool['docker'].get("image") or f"{tool_name.lower()}_img"
    output_path = tool['output_path']
    #[TODO]: based on input type of tool determine the preprocessor script
    pre_script_name = f"{stage}_preprocessor.py"
    post_script_name = f"{stage}_postprocessor.py"

    logger.info(f"Running tool '{tool_name}' in stage '{stage}' using image '{image_name}'...")
    try:
        env = {
            "DATASET_DIR": "/data",        
            "INPUT_IR": f"/data/{os.path.basename(input_path)}",   
            "OUTPUT_IR": f"/data/{os.path.basename(output_path)}" 
        }
        volumes = {
            os.path.abspath(dataset_dir): {"bind": "/data", "mode": "rw"}
        }
        scripts_dir = os.path.abspath("./scripts")  
        if os.path.isdir(scripts_dir):
            volumes[scripts_dir] = {"bind": "/pipeline_scripts", "mode": "ro"}
            env["PRE_SCRIPT"] = f"/pipeline_scripts/{pre_script_name}"
            env["POST_SCRIPT"] = f"/pipeline_scripts/{post_script_name}"
        command = ""
        if os.path.isdir(scripts_dir):
            command += f"python /pipeline_scripts/{pre_script_name} && "
        
        command += ":"  
        if os.path.isdir(scripts_dir):
            command += f" && python /pipeline_scripts/{post_script_name}"
        if command == ":":
            command = None  
        else:
            command = command.replace(": &&", "")  
        if docker:
            client = docker.from_env()
            container = client.containers.run(image_name, command=command, environment=env, volumes=volumes, detach=True)
            exit_code = container.wait()  
            exit_code = exit_code.get("StatusCode", exit_code)
            logs = container.logs().decode('utf-8')
            container.remove()  
            if exit_code != 0:
                logger.error(f"Tool {tool_name} in stage {stage} failed with exit code {exit_code}. Logs:\n{logs}")
                raise RuntimeError(f"Tool {tool_name} failed (exit code {exit_code})")
            else:
                logger.info(f"Tool {tool_name} completed successfully. Output written to {output_path}.")
        else:
            docker_cmd = ["docker", "run", "--rm"]  
            # Add env variables
            for k, v in env.items():
                docker_cmd += ["-e", f"{k}={v}"]
            for host_path, mount in volumes.items():
                bind_path = mount["bind"]
                mode = mount.get("mode", "rw")
                docker_cmd += ["-v", f"{host_path}:{bind_path}:{mode}"]
            docker_cmd.append(image_name)
            if command:
                # Use bash to run multiple commands
                docker_cmd += ["bash", "-c", command]
            logger.debug(f"Running subprocess: {' '.join(docker_cmd)}")
            result = subprocess.run(docker_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Tool {tool_name} failed with exit code {result.returncode}. Error:\n{result.stderr}")
                raise RuntimeError(f"Tool {tool_name} in {stage} failed.")
            else:
                logger.info(f"Tool {tool_name} completed successfully. Output written to {output_path}.")
    except Exception as e:
        logger.error(f"Error while running tool {tool_name} in stage {stage}: {e}")
        raise
    return os.path.join(dataset_dir, output_path)


#def evaluate_model(model_path: str, dataset_path: str) -> float:
#    """
#    Evaluate the model (at model_path) on the test portion of the dataset (dataset_path).
#    Returns the accuracy as a float.
#    """
#    logger.info(f"Evaluating model {model_path} against test dataset in {dataset_path} ...")
#    try:
#        import h5py
#    except ImportError:
#        logger.warning("h5py not installed, cannot directly read dataset. Skipping actual evaluation.")
#        return 0.0
#    acc = 0.0
#    try:
#        with h5py.File(dataset_path, 'r') as ds:
#            if "X_test" in ds and "y_test" in ds:
#                X_test = ds["X_test"][:]
#                y_test = ds["y_test"][:]
#            else:
#                logger.error("Dataset IR does not contain test data for evaluation.")
#                return acc
#        try:
#            import tensorflow as tf
#            model = tf.keras.models.load_model(model_path)
#            results = model.evaluate(X_test, y_test, verbose=0)
#            if isinstance(results, list):
#                acc = results[-1]
#            else:
#                acc = results  
#        except Exception as e:
#            # when model is not Tensorflow or Keras
#            logger.warning(f"Could not load/evaluate model using Keras. Error: {e}")
#            acc = 0.0
#        logger.info(f"Model accuracy: {acc:.4f}")
#    except Exception as e:
#        logger.error(f"Failed to evaluate model {model_path}: {e}")
#    return acc


def prepare_dataset(dataset_name):
    # check if this dataset exist in /data directory 
    ##if not, get the link of dataset from config file 
    # download dataset in /data directory
    #check if it is in h5 format 
    #if not convert it to h5
    # and store it
    return



def run_pipeline(config: Dict):
    """
    Execute the defense pipeline based on the provided configuration.
    The config dictionary should contain the dataset and lists of tools for each stage.
    """
    dataset_name = config.get("dataset")
    dataset_dir = prepare_dataset(dataset_name)
   
    logger.info(f"Using dataset '{dataset_name}' (IR directory: {dataset_dir})")
    
    current_input = os.path.join(dataset_dir, f"{dataset_name}.h5")
    if not os.path.exists(current_input):
        logger.warning(f"Dataset IR file {current_input} not found. The pipeline will proceed assuming it appears in preprocessing steps.")
    
    for stage in ["pre_training", "during_training", "post_training"]:
        tools = config.get(stage, [])
        if not tools:
            logger.info(f"No tools configured for stage '{stage}'. Skipping this stage.")
            continue
        logger.info(f"Starting stage '{stage}' with {len(tools)} tool(s).")
        for tool in tools:
            image_name = build_docker_image(tool)
            output_path = run_tool_in_container(tool, stage, dataset_dir, current_input)
            current_input = output_path
        logger.info(f"Completed stage '{stage}'. Current intermediate result: {current_input}")
    
    final_model_path = current_input
    defended_acc = evaluate_model(final_model_path, current_input if 'dataset' in current_input else os.path.join(dataset_dir, f"{dataset_name}.h5"))
    
    logger.info("Computing baseline model (training with no defenses) for comparison...")
    baseline_model_path = os.path.join(dataset_dir, "baseline_model.h5")
    baseline_acc = 0.0
    try:
        if os.path.exists(baseline_model_path):
            baseline_acc = evaluate_model(baseline_model_path, os.path.join(dataset_dir, f"{dataset_name}.h5"))
        else:
            if os.path.exists(final_model_path):
                import shutil
                shutil.copy(final_model_path, baseline_model_path)
                baseline_acc = defended_acc 
            else:
                baseline_acc = 0.0
    except Exception as e:
        logger.error(f"Failed to compute baseline model: {e}")
    
    logger.info(f"Defended model accuracy: {defended_acc:.4f}")
    logger.info(f"Baseline model accuracy: {baseline_acc:.4f}")
    acc_diff = defended_acc - baseline_acc
    logger.info(f"Accuracy difference (defended - baseline): {acc_diff:.4f}")
    print(f"Defended model accuracy: {defended_acc:.4f}")
    print(f"Baseline model accuracy: {baseline_acc:.4f} (No defenses)")
    print(f"Accuracy difference: {acc_diff:.4f}")

# ---------------------------- Main entry point ----------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Modular ML Defense Pipeline")
    parser.add_argument("--config", "-c", type=str, help="Path to configuration JSON for preconfigured mode")
    args = parser.parse_args()
    if args.config:
        config_path = args.config
        try:
            with open(config_path, 'r') as cf:
                config = json.load(cf)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
            sys.exit(1)
    else:
        config = {"dataset": "", "pipeline": {"pre_training": [], "during_training": [], "post_training": []}, "docker": {}}
        dataset_name = input("Enter the dataset name: ").strip()
        config["dataset"] = dataset_name
        logger.info(f"User selected dataset: {dataset_name}")
        for stage in ["pre_training", "during_training", "post_training"]:
            print(f"\nAvailable tools for {stage}:")
            tools_list = TOOLS_DB.get("pipeline"[stage], [])
            for idx, tool in enumerate(tools_list, start=1):
                print(f"{idx}. {tool['tool_name']} - {tool.get('description', '')}")
            print("0. [Skip this stage]")
            choices = input(f"Select tool(s) for {stage} (e.g., 1 or multiple like 1,2): ").strip()
            if choices == "" or choices == "0":
                logger.info(f"No tools selected for {stage}.")
                continue
            chosen_indices = [int(x) for x in choices.split(",") if x.isdigit()]
            for i in chosen_indices:
                if 1 <= i <= len(tools_list):
                    config[stage].append(tools_list[i-1])
                    logger.info(f"Selected tool '{tools_list[i-1]['tool_name']}' for stage {stage}")
        logger.info("Configuration assembled from interactive input.")
    # Run the pipeline with the assembled config
    try:
        run_pipeline(config)
        logger.info("Pipeline execution completed.")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)
