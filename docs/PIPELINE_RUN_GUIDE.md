# Landseer Pipeline Run Guide

This guide explains how to run a complete ML defense pipeline using Landseer, from system setup through execution to result interpretation.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Setup](#system-setup)
3. [Starting a Pipeline Run](#starting-a-pipeline-run)
4. [Monitoring Pipeline Progress](#monitoring-pipeline-progress)
5. [Interpreting Results](#interpreting-results)
6. [Advanced Options](#advanced-options)

---

## Prerequisites

Before running a pipeline, ensure you have:

- **Python 3.11+** (or compatible version)
- **Docker** with GPU support (for containerized tools)
- **5-20+ GB free disk space** (depending on dataset and tool caching)
- **NVIDIA drivers** (if using GPU acceleration)
- A configured pipeline YAML file (e.g., `configs/pipeline/trades.yaml`)
- A model config script (e.g., `configs/model/config_model.py`)

### Optional but Recommended
- MySQL database for result persistence
- MinIO for distributed artifact caching across workers
- Multiple GPU devices for parallel execution

---

## System Setup

### Step 1: Install Dependencies

**Recommended: Using pip in a writable virtualenv**

(Note: Poetry installation may encounter library compatibility issues on some systems. This guide uses pip, which is tested and works reliably.)

```bash
# Create a virtualenv in a writable location
python3 -m venv ~/landseer-env
source ~/landseer-env/bin/activate

# Navigate to Landseer directory
cd /path/to/Landseer

# Install Landseer (editable install includes all dependencies)
pip install -e .

# Verify installation
python -c "from src.backend import cli; print('✓ Backend module available')"
python -c "from src.worker import cli; print('✓ Worker module available')"
```

**Alternative: Using Poetry (if available on your system)**

If Poetry is installed and compatible:

```bash
# Install dependencies with Poetry
poetry install

# Activate Poetry environment
poetry shell
# OR prefix commands with: poetry run
```

### Step 2: Start the Backend Server

The backend manages tasks, workers, and pipeline state:

```bash
# From the Landseer project directory
cd /path/to/Landseer

# Activate your virtualenv
source ~/landseer-env/bin/activate

# Start backend server
PYTHONPATH=./src:. python -m src.backend.cli \
  --host 0.0.0.0 \
  --port 8000 \
  --config configs/pipeline/trades.yaml

# Alternative with Poetry
poetry run landseer-backend \
  --host 0.0.0.0 \
  --port 8000 \
  --config configs/pipeline/trades.yaml
```

**Expected output:**
```
============================================================
Starting Landseer Backend...
============================================================
...
Backend initialized successfully!
Starting FastAPI server...
============================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Verify backend is running:**
```bash
curl http://localhost:8000/health
# Should return: {"status":"ok","timestamp":"...","scheduler_active":true}
```

### Step 3: Start One or More Workers

Workers execute tasks on the pipeline. Start at least one (typically matches your GPU count):

```bash
# Activate virtualenv
source ~/landseer-env/bin/activate
cd /path/to/Landseer

# Terminal 1: Start worker 1 (GPU 0)
PYTHONPATH=./src:. python -m src.worker.cli \
  --backend-url http://localhost:8000 \
  --workspace /data/landseer/workers/worker_1 \
  --cache-dir /data/landseer/cache \
  --gpu 0

# Terminal 2: Start worker 2 (GPU 1, if available)
PYTHONPATH=./src:. python -m src.worker.cli \
  --backend-url http://localhost:8000 \
  --workspace /data/landseer/workers/worker_2 \
  --cache-dir /data/landseer/cache \
  --gpu 1

# CPU-only worker (no GPU)
PYTHONPATH=./src:. python -m src.worker.cli \
  --backend-url http://localhost:8000 \
  --workspace /data/landseer/workers/worker_cpu \
  --no-cache

# Alternative with Poetry
poetry run landseer-worker \
  --backend-url http://localhost:8000 \
  --workspace /data/landseer/workers/worker_1 \
  --gpu 0
```

**Expected output:**
```
============================================================
Starting Landseer Worker: worker_<UUID>
============================================================
Registered as worker: worker_<UUID>
Worker started, entering work loop...
No tasks available, waiting 5.0s...
```

### Step 4: (Optional) Start the Web Frontend

The web dashboard provides a visual interface for managing pipelines:

```bash
# Navigate to frontend directory
cd /path/to/Landseer/src/frontend

# Option A: Pre-built frontend (if available)
python -m http.server 3000 --directory dist
# Open http://localhost:3000

# Option B: Development server (requires Node.js)
npm install
npm run dev
# Open http://localhost:3000
```

---

## Starting a Pipeline Run

### Method 1: Using the REST API (Direct)

Start a pipeline run by sending a POST request to the backend API:

```bash
# Basic pipeline run (uses defaults: cache enabled, all combinations)
curl -X POST http://localhost:8000/api/pipeline-configs/trades/runs \
  -H "Content-Type: application/json" \
  -d '{
    "use_cache": true,
    "dry_run": false
  }'

# Response:
# {
#   "id": "run_20260320_183000_abe12345",
#   "pipeline_config_id": "trades",
#   "run_number": 1,
#   "status": "PENDING",
#   "created_at": "2026-03-20T18:30:00.000000",
#   ...
# }
```

**Request parameters:**
- `use_cache` (bool, default: `true`): Enable/disable artifact caching
- `combo_id` (string, optional): Run only a specific combination ID
- `dry_run` (bool, default: `false`): Validate configuration without executing
- `attack_config_path` (string, optional): Override the default attack config

**Response fields:**
- `id`: Unique run ID (use this for monitoring)
- `status`: Current run status (PENDING, RUNNING, COMPLETED, FAILED)
- `run_number`: Sequential number for this config

### Method 2: Using Python Client

```python
import httpx

# Create client
client = httpx.Client(base_url="http://localhost:8000")

# Start pipeline run
response = client.post(
    "/api/pipeline-configs/trades/runs",
    json={
        "use_cache": True,
        "dry_run": False
    }
)

run_data = response.json()
run_id = run_data["id"]
print(f"Started pipeline run: {run_id}")

# Close client
client.close()
```

### Method 3: Using the Web Frontend

1. Open http://localhost:3000 in your browser
2. Navigate to **Pipelines** section
3. Select a pipeline configuration (e.g., "trades")
4. Click **"Run Pipeline"** button
5. Configure options:
   - ☐ Use Cache (checked by default)
   - ☐ Dry Run (unchecked by default)
6. Click **"Start Run"**
7. You'll see the run ID and status on screen

---

## Monitoring Pipeline Progress

### Check Run Status

```bash
# Get status of a specific run
curl http://localhost:8000/api/pipeline-runs/run_20260320_183000_abe12345

# Get all runs for a config
curl http://localhost:8000/api/pipeline-configs/trades/runs
```

### Check Worker Status

```bash
# List all registered workers
curl http://localhost:8000/workers

# Get worker details
curl http://localhost:8000/workers/worker_a7a831c8
```

### Check Task Progress

```bash
# Get pipeline task list
curl http://localhost:8000/tasks | jq '.tasks[] | {id, status, tool_name}'

# Watch progress in real-time
while true; do
  curl -s http://localhost:8000/tasks \
    | jq '{total: .total, pending: (.tasks[] | select(.status=="PENDING") | .status) | length}'
  sleep 5
done
```

### Web Frontend Monitoring

1. **Dashboard**: Real-time overview of running tasks and worker status
2. **Tasks Page**: Detailed view of all tasks with filters
3. **Workflows**: View pipeline structure and workflow completion
4. **Metrics**: View evaluation results as they complete
5. **Workers**: List of registered workers and their status

### Python Monitoring Script

```python
import httpx
import time
import json

client = httpx.Client(base_url="http://localhost:8000")
run_id = "run_20260320_183000_abe12345"

while True:
    # Check run status
    run = client.get(f"/api/pipeline-runs/{run_id}").json()
    
    # Check progress
    progress = client.get("/tasks").json()
    total = progress["total"]
    completed = len([t for t in progress["tasks"] if t["status"] == "COMPLETED"])
    failed = len([t for t in progress["tasks"] if t["status"] == "FAILED"])
    
    print(f"[{run['status']}] Progress: {completed}/{total} completed, {failed} failed")
    
    if run["status"] in ["COMPLETED", "FAILED"]:
        print(f"Run finished with status: {run['status']}")
        break
    
    time.sleep(10)

client.close()
```

---

## Interpreting Results

### Access Results

Results are stored in multiple locations:

```
Landseer/
├── results/                    # CSV results
│   ├── results_combinations.csv
│   └── results_tools.csv
├── cache/                      # Cached artifacts and logs
└── run_logs/                   # Pipeline execution logs
```

### Results CSV Format

**`results_combinations.csv`**: Summary for each tool combination
```csv
pipeline_id,combination,pre_training,in_training,post_training,dataset_name,acc_train_clean,acc_test_clean,acc_robust,asr,total_duration
trades_run_1,combo_001,pre_xgbod,in_trades,post_magnet,cifar10,0.945,0.923,0.812,0.045,1234.5
```

**`results_tools.csv`**: Individual tool execution details
```csv
pipeline_id,combination,stage,tool_name,cache_key,duration_sec,status
trades_run_1,combo_001,pre_training,pre_xgbod,a1b2c3d4,45.3,COMPLETED
```

### Query Results via API

```bash
# Get metrics for a specific workflow
curl "http://localhost:8000/workflows/{workflow_id}/metrics"

# Get all pipeline metrics
curl "http://localhost:8000/pipelines/{pipeline_id}/metrics"
```

### Analyze Results in Python

```python
import pandas as pd

# Load results
combinations = pd.read_csv("results/results_combinations.csv")
tools = pd.read_csv("results/results_tools.csv")

# Analyze best combinations by clean accuracy
best = combinations.nlargest(5, "acc_test_clean")[
    ["combination", "pre_training", "in_training", "post_training", "acc_test_clean"]
]
print(best)

# Check most expensive tools
tool_costs = tools.groupby("tool_name")["duration_sec"].sum().sort_values(ascending=False)
print(tool_costs.head())

# Find failed tools
failed = tools[tools["status"] == "FAILED"]
print(f"Failed tools: {len(failed)}")
```

---

## Advanced Options

### Running with Specific Configurations

```bash
# Use custom pipeline config
curl -X POST http://localhost:8000/api/pipeline-configs/custom-defense/runs \
  -H "Content-Type: application/json" \
  -d '{"use_cache": true}'

# Run only a specific combination
curl -X POST http://localhost:8000/api/pipeline-configs/trades/runs \
  -H "Content-Type: application/json" \
  -d '{"combo_id": "combo_012"}'
```

### Dry Run Mode (Validation Only)

```bash
# Validate configuration without executing tasks
curl -X POST http://localhost:8000/api/pipeline-configs/trades/runs \
  -H "Content-Type: application/json" \
  -d '{"dry_run": true}'
```

### Stopping a Run

```bash
# Stop an active pipeline run
curl -X POST http://localhost:8000/api/pipeline-runs/run_20260320_183000_abe12345/stop

# Restart a stopped run
curl -X POST http://localhost:8000/api/pipeline-runs/run_20260320_183000_abe12345/restart \
  -H "Content-Type: application/json" \
  -d '{"use_cache": true}'
```

### Enable Result Persistence

To store results in a MySQL database (optional):

```bash
# Start MySQL container
docker run -d --name landseer-db \
  -e MYSQL_ROOT_PASSWORD=rootpass \
  -e MYSQL_DATABASE=landseer \
  -e MYSQL_USER=landseer \
  -e MYSQL_PASSWORD=landseer \
  -p 3306:3306 \
  mysql:8.0

# Set environment variables
export LANDSEER_DB_HOST=localhost
export LANDSEER_DB_USER=landseer
export LANDSEER_DB_PASSWORD=landseer
export LANDSEER_DB_NAME=landseer

# Backend will now automatically persist results to MySQL
```

### Distributed Caching (MinIO)

For multi-worker setups, enable MinIO artifact caching:

```bash
# Start MinIO container
docker run -d --name landseer-minio \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  -p 9000:9000 \
  -p 9001:9001 \
  minio/minio server /data

# Set MinIO endpoint (default for local)
# Workers and backend will automatically cache artifacts in MinIO
```

---

## Troubleshooting

### Backend won't start

```bash
# Check if port 8000 is already in use
lsof -i :8000

# Try a different port
PYTHONPATH=./src:. python -m src.backend.cli --port 8001

# Enable debug mode for more details
PYTHONPATH=./src:. python -m src.backend.cli --debug

# Ensure virtualenv is activated
source ~/landseer-env/bin/activate
```

### Poetry won't install (libpython error)

If Poetry fails with "libpython3.X.so.1.0: cannot open shared object file", use **pip instead**:

```bash
# Use pip in a virtualenv (works reliably)
python3 -m venv ~/landseer-env
source ~/landseer-env/bin/activate
pip install -e /path/to/Landseer
```

### Workers can't connect to backend

```bash
# Verify backend is running
curl http://localhost:8000/health

# Check network connectivity from worker machine
ping localhost

# Try explicit backend URL (use machine IP if on different machine)
PYTHONPATH=./src:. python -m src.worker.cli \
  --backend-url http://127.0.0.1:8000

# Enable worker debug mode
PYTHONPATH=./src:. python -m src.worker.cli \
  --backend-url http://localhost:8000 \
  --debug
```

### Tasks are stuck/not executing

```bash
# Check if workers are idle
curl http://localhost:8000/workers | jq '.workers[] | {worker_id, status}'

# Verify tasks are available
curl http://localhost:8000/tasks | jq '.tasks[0]'

# Check worker logs for errors in terminal output
```

### Out of disk space

Cache and results consume significant disk space. To manage:

```bash
# Clear cache (be careful: this invalidates caching benefits)
rm -rf /data/landseer/cache

# Start worker without cache
PYTHONPATH=./src:. python -m src.worker.cli \
  --backend-url http://localhost:8000 \
  --no-cache

# Move cache to larger disk
PYTHONPATH=./src:. python -m src.worker.cli \
  --backend-url http://localhost:8000 \
  --cache-dir /mnt/large_disk/cache

# Check disk usage
du -sh /data/landseer/cache
du -sh results/
```

---

## Example: Complete Workflow

```bash
# Setup: One-time installation
python3 -m venv ~/landseer-env
source ~/landseer-env/bin/activate
cd /path/to/Landseer
pip install -e .

# ==============================================================
# RUNNING THE PIPELINE (Terminal 1: Backend)
# ==============================================================
source ~/landseer-env/bin/activate
cd /path/to/Landseer
PYTHONPATH=./src:. python -m src.backend.cli --config configs/pipeline/trades.yaml

# ==============================================================
# RUNNING THE PIPELINE (Terminal 2: Worker)
# ==============================================================
source ~/landseer-env/bin/activate
cd /path/to/Landseer
PYTHONPATH=./src:. python -m src.worker.cli \
  --backend-url http://localhost:8000 \
  --workspace /data/landseer/workers/worker_1 \
  --gpu 0

# ==============================================================
# RUNNING THE PIPELINE (Terminal 3: Trigger Run)
# ==============================================================
# Start pipeline run
RUN_RESPONSE=$(curl -s -X POST http://localhost:8000/api/pipeline-configs/trades/runs \
  -H "Content-Type: application/json" \
  -d '{"use_cache": true}')

RUN_ID=$(echo $RUN_RESPONSE | jq -r '.id')
echo "Started run: $RUN_ID"

# ==============================================================
# MONITORING (Terminal 4: Watch Progress)
# ==============================================================
while true; do
  curl -s http://localhost:8000/api/pipeline-runs/$RUN_ID | \
    jq '{status: .status, created_at: .created_at, started_at: .started_at}'
  sleep 30
done

# ==============================================================
# ANALYSIS: After run completes
# ==============================================================
source ~/landseer-env/bin/activate
python3 << 'EOF'
import pandas as pd

results = pd.read_csv('results/results_combinations.csv')
print('Top 10 combinations by test accuracy:')
print(results.nlargest(10, 'acc_test_clean')[[
    'pre_training', 'in_training', 'post_training', 
    'acc_test_clean', 'total_duration'
]])

print('\nTool execution summary:')
tool_stats = pd.read_csv('results/results_tools.csv')
print(tool_stats.groupby('tool_name')['duration_sec'].agg(['count', 'sum', 'mean']))
EOF
```

---

## More Information

- [Pipeline Configuration Guide](./PIPELINE_CONFIG_GUIDE.md)
- [Database Setup Guide](./DATABASE_SETUP.md)
- [System Working Guide](./LANDSEER_SYSTEM_WORKING_GUIDE.md)
- [API Documentation](../README.md#usage)
