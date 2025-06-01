# Landseer: Exploring the Machine Learning Defense Landscape

## Overview

Landseer is a modular framework to systematically explore, compose, and evaluate machine learning defenses across a range of threat models. While many existing defenses target narrow threat categories (robustness, privacy, fairness, etc.), Landseer allows researchers and practitioners to analyze how multiple defenses interact when combined across the entire ML pipeline. The framework supports seamless integration of multiple defense stages and includes an automated evaluation engine that tests system performance and robustness across a variety of attack scenarios.

## Key Features

- **Automated Pipeline Execution**: Systematically tests all combinations of defense tools
- **Docker-based Tool Integration**: Runs defense mechanisms in isolated containers
- **Multi-stage Defense Testing**: Supports pre-training, during-training, and post-training defenses
- **Comprehensive Evaluation**: Tests clean accuracy, robustness, fingerprinting resistance, and attack success rates
- **Attack Simulation**: Supports backdoor, evasion, extraction, and inference attacks
- **Intelligent Caching**: Avoids redundant computations with hash-based caching
- **Parallel Execution**: GPU-aware parallel processing of combinations
- **Result Tracking**: Detailed logging and CSV-based result reporting

### Supported Metrics
- **Clean Accuracy**: Standard model performance on clean test data
- **Robust Accuracy**: Performance under adversarial attacks (PGD)
- **Out-of-Distribution (OOD) Detection**: AUC for detecting anomalous inputs
- **Fingerprinting Resistance**: Model's resistance to ownership verification attacks
- **Backdoor Attack Success Rate (ASR)**: Success rate of backdoor triggers
- **Training Duration**: Time measurements for each tool and combination

### Attack Types
- **Backdoor**: Data poisoning attacks with trigger patterns
- **Evasion**: Adversarial examples designed to fool the model
- **Extraction**: Model stealing and membership inference attacks
- **Inference**: Privacy attacks extracting training data information

## Installation

### Prerequisites
- Python 3.11+
- Docker with GPU support (for CUDA-enabled tools)
- NVIDIA Container Toolkit (for GPU acceleration)

### Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd landseer-pipeline
   ```

2. **Install dependencies**
   ```bash
   pip install poetry
   poetry install
   ```

3. **Configure Docker access**
   ```bash
   # Ensure Docker daemon is running
   sudo systemctl start docker
   
   # Add user to docker group (optional)
   sudo usermod -aG docker $USER
   ```

4. **Set up environment variables**
   ```bash
   # For private GitHub Container Registry access
   export GHCR_TOKEN=your_github_token
   ```

## Usage

### Basic Pipeline Execution

```bash
# Run pipeline with configuration files
poetry run landseer -c configs/pipeline/test_config.yaml -a configs/attack/test_config_1.yaml

# With custom output directory
poetry run landseer -c configs/pipeline/test_config.yaml -a configs/attack/test_config_1.yaml -o ./my_results

# Disable caching (run all tools fresh)
poetry run landseer -c configs/pipeline/test_config.yaml -a configs/attack/test_config_1.yaml --no-cache

# CPU-only execution
poetry run landseer -c configs/pipeline/test_config.yaml -a configs/attack/test_config_1.yaml --no-gpu
```

### Configuration Options

**Pipeline Configuration** (`configs/pipeline/*.yaml`):
```yaml
dataset:
  name: cifar10
  link: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  format: pickle
  sha1: c58f30108f718f92721af3b95e74349a

pipeline:
  pre_training:
    tools:
      - name: feature-squeeze
        docker:
          image: ghcr.io/landseer-project/pre_squeeze:v1
          command: python main.py --bit-depth 4
    noop:
      name: noop
      docker:
        image: ghcr.io/landseer-project/pre_noop:v1
        command: python main.py
  # ... during_training and post_training sections
```

**Attack Configuration** (`configs/attack/*.yaml`):
```yaml
attacks:
  backdoor: true
  evasion: false
  extraction: false
  inference: false
  other: false
```

##  Project Structure

```
landseer-pipeline/
├── src/landseer_pipeline/
│   ├── config/                 # Configuration management
│   ├── dataset_handler/        # Dataset loading and preprocessing
│   ├── docker_handler/         # Docker container management
│   ├── evaluator/             # Model evaluation and metrics
│   ├── pipeline/              # Pipeline execution logic
│   ├── tools/                 # Tool execution framework
│   └── utils/                 # Utilities (logging, GPU, files)
├── configs/
│   ├── pipeline/              # Pipeline configuration files
│   ├── attack/                # Attack configuration files
│   └── model/                 # Model architecture definitions
├── cache/                     # Cached tool outputs
├── results/                   # Experiment results
└── logs/                      # Execution logs
```

2. **Tool interface requirements**:
   - Input: `/data` directory (dataset + previous tool outputs)
   - Output: `/output` directory (processed data/model)
   - Config: `config_model.py` (model architecture)

3. **Expected outputs**:
   - Pre-training tools: Processed dataset files (npy)
   - During-training tools: `model.pt` (trained PyTorch model)
   - Post-training tools: `model.pt` (refined model)

## Results Interpretation

### Output Files
All experiment results are stored in the `results/` directory, containing metrics and logs for each valid defense combination across configured attack scenarios.

1. **`results_combinations.csv`**: Main results for each tool combination
   ```csv
   pipeline_id,combination,pre_training,in_training,post_training,dataset_name,dataset_type,acc_train_clean,acc_test_clean,acc_robust,ood_auc,fingerprinting,asr,total_duration
   ```

2. **`results_tools.csv`**: Individual tool execution details
   ```csv
   pipeline_id,combination,stage,tool_name,cache_key,duration_sec,status,output_path
   ```

## Public Defense Docker Images

Landseer provides all defense modules as pre-built Docker images hosted at:

* `ghcr.io/landseer-project/`
