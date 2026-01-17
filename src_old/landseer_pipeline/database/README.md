# Landseer Pipeline Database

This module provides MySQL database support for storing and querying pipeline run results.

## Features

- **Structured Storage**: Smart table design optimized for querying pipeline runs, tool combinations, and metrics
- **ORM Models**: SQLAlchemy models for easy Python integration
- **Repository Pattern**: High-level data access methods for common operations
- **Query Helpers**: Convenient functions for analysis and reporting
- **Result Importer**: Import existing CSV/JSON results into the database

## Quick Start

### 1. Prerequisites

- MySQL 8.0 or later
- Python packages: `mysql-connector-python`, `sqlalchemy`

```bash
pip install mysql-connector-python sqlalchemy
```

### 2. Create Database

```bash
# Create the database and user
mysql -u root -p -e "
  CREATE DATABASE landseer_pipeline CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
  CREATE USER 'landseer'@'localhost' IDENTIFIED BY 'your_password';
  GRANT ALL PRIVILEGES ON landseer_pipeline.* TO 'landseer'@'localhost';
  FLUSH PRIVILEGES;
"

# Apply the schema
mysql -u landseer -p landseer_pipeline < schema.sql
```

Or use the setup script:
```bash
chmod +x setup_db.sh
./setup_db.sh --create-db --create-user --password your_password --root-password root_password
```

### 3. Configure Connection

Set environment variables:
```bash
export LANDSEER_DB_HOST=localhost
export LANDSEER_DB_PORT=3306
export LANDSEER_DB_NAME=landseer_pipeline
export LANDSEER_DB_USER=landseer
export LANDSEER_DB_PASSWORD=your_password
```

Or copy `.env.example` to `.env` and update values.

### 4. Import Existing Results

```bash
python -m landseer_pipeline.database.importer /path/to/Landseer/results
```

## Usage Examples

### Basic Connection

```python
from landseer_pipeline.database import get_db_connection

# Using environment variables
db = get_db_connection()
db.connect()

# Execute queries
results = db.fetch_all("SELECT * FROM pipeline_runs LIMIT 10")
for row in results:
    print(row['pipeline_id'], row['status'])

db.close()
```

### Using SQLAlchemy ORM

```python
from landseer_pipeline.database.models import create_session, PipelineRun, Combination

session = create_session("mysql+mysqlconnector://landseer:pass@localhost/landseer_pipeline")

# Query recent runs
runs = session.query(PipelineRun).order_by(PipelineRun.run_timestamp.desc()).limit(5).all()
for run in runs:
    print(f"{run.pipeline_id}: {run.total_combinations} combinations")

session.close()
```

### Using Repositories

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from landseer_pipeline.database.repository import PipelineRunRepository, MetricsRepository

engine = create_engine("mysql+mysqlconnector://landseer:pass@localhost/landseer_pipeline")
Session = sessionmaker(bind=engine)
session = Session()

# Get pipeline run details
run_repo = PipelineRunRepository(session)
run = run_repo.get_latest()
print(f"Latest run: {run.pipeline_id}, success rate: {run.success_rate}%")

# Compare tool performance
metrics_repo = MetricsRepository(session)
comparison = metrics_repo.compare_tools(
    tool_names=["in-trades", "in-pgd"],
    dataset_name="cifar10"
)
print(comparison)

session.close()
```

### Using Query Helper

```python
from landseer_pipeline.database.queries import QueryHelper

helper = QueryHelper("mysql+mysqlconnector://landseer:pass@localhost/landseer_pipeline")

# Find best performing combinations
best = helper.find_best_combinations(
    metric="acc_test_clean",
    dataset="cifar10",
    limit=10
)
for comb in best:
    print(f"{comb['combination_code']}: {comb['acc_test_clean']:.4f}")

# Compare tools
comparison = helper.compare_tools(
    tools=["in-trades", "in-pgd", "in-free"],
    metrics=["acc_test_clean", "pgd_accuracy"]
)
print(comparison)

# Find robust combinations
robust = helper.find_robust_combinations(
    min_clean_acc=0.75,
    min_pgd_acc=0.40,
    dataset="cifar10"
)
for r in robust:
    print(f"{r['combination_code']}: clean={r['clean_accuracy']:.4f}, pgd={r['pgd_accuracy']:.4f}")
```

## Database Schema

### Core Tables

| Table | Description |
|-------|-------------|
| `pipeline_runs` | Main pipeline execution records |
| `combinations` | Tool combinations within each run |
| `combination_metrics` | Evaluation metrics for each combination |
| `combination_tools` | Many-to-many linking combinations to tools |
| `tool_executions` | Individual tool execution records |

### Reference Tables

| Table | Description |
|-------|-------------|
| `datasets` | Dataset registry (cifar10, mnist, etc.) |
| `models` | Model configurations |
| `tools` | Available tools registry |
| `tool_categories` | Tool categorization |

### Artifact Tables

| Table | Description |
|-------|-------------|
| `artifact_nodes` | Content-addressable artifact storage |
| `artifact_files` | Files within artifact nodes |
| `output_file_provenance` | Tracks which tool produced each output |

### Pre-built Views

| View | Description |
|------|-------------|
| `v_combination_results` | Complete combination results with metrics |
| `v_combination_tools_summary` | Tools used per combination by stage |
| `v_tool_performance` | Aggregate tool performance statistics |
| `v_pipeline_summary` | Pipeline run summaries |
| `v_best_combinations` | Top combinations by robustness score |
| `v_cache_efficiency` | Cache hit rates and time savings |

## Common Queries

### Find combinations using specific tool
```sql
CALL sp_get_combinations_with_tool('in-trades', 'during_training');
```

### Compare two tools
```sql
CALL sp_compare_tool_effectiveness('in-trades', 'in-pgd', 'cifar10');
```

### Get pipeline details
```sql
CALL sp_get_pipeline_details('3ef5725f725d1dad');
```

### Find best combination for a metric
```sql
CALL sp_find_best_combination('cifar10', 'pgd_accuracy', 10);
```

### Custom analysis queries
```sql
-- Top combinations by clean accuracy
SELECT c.combination_code, cm.acc_test_clean, cm.pgd_accuracy
FROM combinations c
JOIN combination_metrics cm ON c.combination_id = cm.combination_id
WHERE c.status = 'success'
ORDER BY cm.acc_test_clean DESC
LIMIT 10;

-- Average metrics by training tool
SELECT 
    t.tool_name,
    COUNT(*) AS num_combinations,
    AVG(cm.acc_test_clean) AS avg_clean_acc,
    AVG(cm.pgd_accuracy) AS avg_pgd_acc
FROM tools t
JOIN combination_tools ct ON t.tool_id = ct.tool_id
JOIN combinations c ON ct.combination_id = c.combination_id
JOIN combination_metrics cm ON c.combination_id = cm.combination_id
WHERE ct.stage = 'during_training' AND c.status = 'success'
GROUP BY t.tool_name
ORDER BY avg_pgd_acc DESC;

-- Cache efficiency by tool
SELECT 
    t.tool_name,
    COUNT(*) AS total_runs,
    SUM(CASE WHEN te.cache_hit THEN 1 ELSE 0 END) AS cache_hits,
    AVG(te.duration_sec) AS avg_duration
FROM tools t
JOIN tool_executions te ON t.tool_id = te.tool_id
GROUP BY t.tool_name;
```

## Entity Relationship Diagram

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│    datasets     │      │     models      │      │ tool_categories │
├─────────────────┤      ├─────────────────┤      ├─────────────────┤
│ dataset_id (PK) │      │ model_id (PK)   │      │ category_id(PK) │
│ dataset_name    │      │ script_path     │      │ category_name   │
│ variant         │      │ content_hash    │      │ description     │
│ version         │      │ framework       │      └────────┬────────┘
└────────┬────────┘      └────────┬────────┘               │
         │                        │                         │
         ▼                        ▼                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        pipeline_runs                             │
├─────────────────────────────────────────────────────────────────┤
│ run_id (PK) │ pipeline_id │ dataset_id (FK) │ model_id (FK)     │
│ run_timestamp │ status │ total_combinations │ config_file_path  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐     ┌─────────────────┐
│ pipeline_attacks│    │  combinations   │     │     tools       │
├─────────────────┤    ├─────────────────┤     ├─────────────────┤
│ attack_id (PK)  │    │combination_id PK│     │ tool_id (PK)    │
│ run_id (FK)     │    │ run_id (FK)     │     │ tool_name       │
│ backdoor_enabled│    │combination_code │     │ stage           │
│ adversarial_..  │    │ status          │     │ category_id(FK) │
└─────────────────┘    └────────┬────────┘     └────────┬────────┘
                                │                       │
         ┌──────────────────────┼───────────────────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│combination_tools│    │combination_     │    │ tool_executions │
├─────────────────┤    │    metrics      │    ├─────────────────┤
│ id (PK)         │    ├─────────────────┤    │ execution_id PK │
│combination_id FK│    │ metric_id (PK)  │    │combination_id FK│
│ tool_id (FK)    │    │combination_id FK│    │ tool_id (FK)    │
│ stage           │    │acc_test_clean   │    │ cache_hit       │
│ tool_order      │    │ pgd_accuracy    │    │ duration_sec    │
└─────────────────┘    │ carlini_l2_acc  │    │ status          │
                       │ ood_auc         │    └─────────────────┘
                       │ mia_auc         │
                       └─────────────────┘
```

## Integration with Pipeline

To automatically store results during pipeline execution, add database recording to your pipeline code:

```python
from landseer_pipeline.database import get_db_connection
from landseer_pipeline.database.repository import (
    DatasetRepository, PipelineRunRepository, 
    CombinationRepository, MetricsRepository
)

# At pipeline start
db = get_db_connection()
db.connect()
session = db.session  # or use SQLAlchemy session

# Create run record
run_repo = PipelineRunRepository(session)
dataset_repo = DatasetRepository(session)

dataset = dataset_repo.get_or_create(name="cifar10", variant="clean")
run = run_repo.create(
    pipeline_id=pipeline_hash,
    run_timestamp=datetime.now(),
    dataset=dataset,
    config_file_path=config_path
)

# After each combination
comb_repo = CombinationRepository(session)
metrics_repo = MetricsRepository(session)

combination = comb_repo.create(
    run=run,
    combination_code=comb_code,
    combination_index=idx,
    tools_by_stage=tools_dict
)

# After evaluation
metrics_repo.create_or_update(
    combination_id=combination.combination_id,
    acc_test_clean=results['accuracy'],
    pgd_accuracy=results['pgd_acc'],
    # ... other metrics
)

session.commit()
```
