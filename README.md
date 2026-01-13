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
- **Clean Accuracy**: Standard model performance on clean train and test data
- **Robust Accuracy**: Performance under adversarial attacks (PGD)
- **Out-of-Distribution (OOD) Detection**: AUC for detecting out-of-distribution inputs
- **Fingerprint confidence score and p-value**: Confidence score and p-value of suspected stolen model to victim model's fingerptint
- **Backdoor Attack Success Rate (ASR)**: Success rate of backdoor triggers
- **Epsilon**: Privacy Budget for differentially private data or models 
- **Training Duration**: Time measurements for each tool and combination

### Attack Types
- **Backdoor**: Data poisoning attacks with trigger patterns
- **Evasion**: Adversarial examples designed to fool the model
- **Extraction**: Model stealing and membership inference attacks
- **Inference**: Privacy attacks extracting training data information

##  Project Structure

```
landseer-pipeline/
├── src/
|   ├── landseer_pipeline/
│      ├── config/                 # Configuration management
│      ├── dataset_handler/        # Dataset loading and preprocessing
│      ├── docker_handler/         # Docker container management
│      ├── evaluator/             # Model evaluation and metrics
│      ├── pipeline/              # Pipeline execution logic
│      ├── tools/                 # Tool execution framework
│      └── utils/                 # Utilities (logging, GPU, files)
|   └── landseer_ui/              
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

5. **(Optional) Set up MySQL Database for Results**
   
   For easier querying and analysis of pipeline results, you can enable MySQL storage.
   See [Database Setup Instructions](docs/DATABASE_SETUP.md) for details.
   
   Quick start with Docker:
   ```bash
   # Start MySQL container
   docker run -d --name landseer-mysql \
     -e MYSQL_ROOT_PASSWORD=rootpass \
     -e MYSQL_DATABASE=landseer_pipeline \
     -e MYSQL_USER=landseer \
     -e MYSQL_PASSWORD=landseer \
     -p 3306:3306 \
     mysql:8.0
   
   # Apply schema
   docker exec -i landseer-mysql mysql -u landseer -plandseer landseer_pipeline \
     < src/landseer_pipeline/database/schema.sql
   
   # Enable database logging
   source .env.db
   ```

## Usage

### Basic Pipeline Execution

```bash
# (Optional) Enable MySQL database logging
source .env.db

# Run pipeline with configuration files
poetry run landseer -c configs/pipeline/test_config.yaml -a configs/attack/test_config_1.yaml

# to spin up web interface
poetry run uvicorn landseer_ui.server:app --reload --host 0.0.0.0 --port 8000
```

### Configuration Options

**Pipeline Configuration** (`configs/pipeline/*.yaml`):
```yaml
dataset:
  name: cifar10                    # Required: Dataset name (cifar10, mnist, celeba, etc.)
  variant: clean                   # Optional: clean, poisoned (default: clean)
  version: "1.0"                   # Optional: Dataset version
  params:                          # Optional: Dataset-specific parameters
    subset_size: 1000
    seed: 42
    poison_fraction: 0.1

# Model Configuration  
model:
  script: /path/to/model_config.py # Required: Path to model definition script
  framework: pytorch               # Required: pytorch, tensorflow, etc.
  params:                          # Optional: Model hyperparameters
    learning_rate: 0.001
    batch_size: 32
    epochs: 100

# Pipeline Stages Configuration
pipeline:
  # Pre-training stage (data preprocessing, outlier detection, etc.)
  pre_training:
    tools:
    - name: pre_xgbod              # Tool name
      docker:
        image: ghcr.io/landseer-project/pre_xgbod:v2
        command: python3 main.py
        config_script: configs/model/config_model.py  # Optional: tool-specific model config
      auxiliar
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

3. **MySQL Database** (if enabled): Results are also stored in MySQL for SQL querying.
   See [Database Setup Instructions](docs/DATABASE_SETUP.md) for query examples.

## Public Defense Docker Images

Landseer provides all defense modules as pre-built Docker images hosted at:

* `ghcr.io/landseer-project/`
### Model Converter Images

For cross-framework interoperability, Landseer provides dedicated model converter containers:

* `ghcr.io/landseer-project/model_converter_pytorch_to_other:v1` - Converts PyTorch models to ONNX or TensorFlow format
* `ghcr.io/landseer-project/model_converter_other_to_pytorch:v1` - Converts TensorFlow or ONNX models to PyTorch format

These converters are automatically invoked when pipeline tools require different model formats, eliminating the need for heavy ML package dependencies in the main Landseer environment.