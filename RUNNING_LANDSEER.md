# Running Landseer Pipeline (Main Branch)

## Quick Start

### 1. Activate the Environment

```bash
cd /share/landseer/workspace-ayushi/landseer-main
source /share/landseer/workspace-ayushi/.shared/envs/landseer-main/bin/activate
```

### 2. Run a Pipeline

Basic command structure:
```bash
landseer --config configs/pipeline/<PIPELINE>.yaml \
         --attack-config configs/attack/<ATTACK>.yaml
```

**Example - Run TRADES pipeline:**
```bash
landseer --config configs/pipeline/trades.yaml \
         --attack-config configs/attack/test_config_1.yaml
```

## Available Pipelines

Located in `configs/pipeline/`:

- **`dnn_watermarking.yaml`** - DNN watermarking defense
- **`dp.yaml`** - Differential Privacy
- **`explain.yaml`** - Explainability techniques
- **`fairness.yaml`** - Fairness-aware training
- **`teaching.yaml`** - Knowledge distillation
- **`trades.yaml`** - TRADES adversarial training
- **`watermarknn.yaml`** - WatermarkNN defense
- **`watermarknn_tensorflow.yaml`** - TensorFlow version

## Available Attack Configs

Located in `configs/attack/`:

- **`test_config_1.yaml`** - No attacks enabled (baseline)
- **`test_config_2.yaml`** - Alternative test configuration
- **`fairness_config.yaml`** - Fairness-related scenarios

## Command Options

### Essential Options

```bash
--config, -c CONFIG              # Pipeline configuration file (required)
--attack-config, -a ATTACK       # Attack configuration file (required)
--log-level LEVEL                # DEBUG, INFO, WARNING, ERROR (default: INFO)
--dry-run                        # Validate config without running
```

### Execution Control

```bash
--combo-id COMBO_ID              # Run specific combination (e.g., comb_001)
--interactive, -i                # Interactively select which combination to run
--no-gpu                         # Disable GPU usage
--no-cache                       # Disable dataset caching
```

### Directory Options

```bash
--output-dir DIR                 # Override output directory
--data-dir DIR                   # Directory to store datasets (default: data/)
--log-dir DIR                    # Directory to store logs (default: logs/)
--results-dir DIR                # Directory to store results (default: results/)
```

### Resource Management

```bash
--max-temp MAX_TEMP              # Maximum GPU temperature threshold
--cooldown-time SECONDS          # GPU cooldown time between runs
--docker-shm-size SIZE           # Docker shared memory (e.g., 512m, 1g, 2g)
--docker-mem-limit SIZE          # Docker memory limit (e.g., 8g)
```

## Example Commands

### 1. Dry Run (Validate Configuration)
```bash
landseer --config configs/pipeline/trades.yaml \
         --attack-config configs/attack/test_config_1.yaml \
         --dry-run
```

### 2. Run with Debug Logging
```bash
landseer --config configs/pipeline/dnn_watermarking.yaml \
         --attack-config configs/attack/test_config_1.yaml \
         --log-level DEBUG
```

### 3. Run Specific Combination
```bash
landseer --config configs/pipeline/fairness.yaml \
         --attack-config configs/attack/fairness_config.yaml \
         --combo-id comb_005
```

### 4. Interactive Mode
```bash
landseer --config configs/pipeline/trades.yaml \
         --attack-config configs/attack/test_config_1.yaml \
         --interactive
```

### 5. Custom Resource Limits
```bash
landseer --config configs/pipeline/dp.yaml \
         --attack-config configs/attack/test_config_1.yaml \
         --docker-shm-size 2g \
         --docker-mem-limit 16g \
         --max-temp 80
```

## Understanding Pipeline Execution

### Pipeline Stages

Landseer executes tools in four stages:
1. **Pre-training** - Data preprocessing and augmentation
2. **During-training** - Training-time defenses
3. **Post-training** - Post-hoc model modifications
4. **Deployment** - Runtime defenses and monitoring

### Combination Generation

The system generates all possible combinations of tools across stages. For example, with:
- 3 pre-training options (including noop)
- 2 during-training options
- 2 post-training options
- 5 deployment options

Total combinations: 3 × 2 × 2 × 5 = **60 combinations**

### Parallel Execution

- Automatically detects available GPUs
- Runs combinations in parallel (one per GPU)
- Manages GPU temperature and resource limits

## Output Locations

After running, check these directories:

### Results
```
results/<PIPELINE_ID>/<TIMESTAMP>/
├── all_pipeline_combinations.csv    # All generated combinations
├── output/
│   ├── comb_000/                    # Per-combination outputs
│   ├── comb_001/
│   └── ...
└── staged_inputs/                    # Intermediate data between stages
```

### Cache
```
cache/artifact_store/                 # Cached tool outputs (reusable)
└── <HASH>/
    ├── output/                       # Tool output files
    └── .success                      # Success marker
```

### Logs
```
logs/                                 # Execution logs
run_logs/                             # Detailed run logs
```

## Monitoring Execution

### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Monitor Docker Containers
```bash
watch -n 2 'docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"'
```

### View Real-time Logs
```bash
tail -f logs/landseer_<TIMESTAMP>.log
```

## Troubleshooting

### Issue: Docker permission denied
```bash
# Add user to docker group (if not already)
sudo usermod -aG docker $USER
newgrp docker
```

### Issue: CUDA out of memory
- Reduce `--docker-mem-limit` or Docker containers running in parallel
- Use `--combo-id` to run one combination at a time
- Check GPU memory: `nvidia-smi`

### Issue: Cache corruption
```bash
# Clear artifact cache
rm -rf cache/artifact_store/*

# Clear dataset cache
rm -rf data/<dataset>/processed/
```

### Issue: Container image not found
```bash
# Pull required images manually
docker pull ghcr.io/landseer-project/pre_xgbod:v2
docker pull ghcr.io/landseer-project/in_trades:v1
# ... etc
```

## Environment Information

- **Python**: 3.14.3
- **PyTorch**: 2.11.0+cu128
- **CUDA**: Available (4 GPUs detected)
- **Virtual Environment**: `/share/landseer/workspace-ayushi/.shared/envs/landseer-main`
- **Data Directory**: Symlinked to shared dataset folder

## Getting Help

```bash
landseer --help
```

For detailed pipeline configuration syntax, see:
- [Pipeline Configuration Examples](configs/pipeline/)
- [Attack Configuration Examples](configs/attack/)
- [Documentation](docs/)

## Tips for Efficient Runs

1. **Use dry-run first** to validate your configuration
2. **Enable caching** (default) to reuse intermediate results
3. **Use interactive mode** to selectively run interesting combinations
4. **Monitor GPU temperature** especially for long runs
5. **Check results incrementally** in `results/<PIPELINE_ID>/` directory
