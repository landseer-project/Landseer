# Landseer Database Setup Instructions

This guide explains how to set up MySQL database storage for Landseer pipeline results, enabling powerful SQL queries for analysis and evaluation.

## Overview

By default, Landseer stores results in CSV files. Optionally, you can enable MySQL storage for:
- **Structured querying**: Use SQL to find best combinations, compare tools, analyze trends
- **Cross-run analysis**: Query across multiple pipeline runs
- **Pre-built views**: Ready-made views for common analysis patterns
- **Stored procedures**: One-call functions for complex queries

## Quick Start (Docker)

### 1. Start MySQL Container

```bash
docker run -d --name landseer-mysql \
  -e MYSQL_ROOT_PASSWORD=rootpass \
  -e MYSQL_DATABASE=landseer_pipeline \
  -e MYSQL_USER=landseer \
  -e MYSQL_PASSWORD=landseer \
  -p 3306:3306 \
  mysql:8.0
```

### 2. Wait for MySQL to Initialize

```bash
# Wait ~10 seconds, then verify
docker exec landseer-mysql mysqladmin ping -h localhost -u root -prootpass --silent
```

### 3. Apply Database Schema

```bash
docker exec -i landseer-mysql mysql -u landseer -plandseer landseer_pipeline \
  < src/landseer_pipeline/database/schema.sql
```

### 4. Verify Tables Created

```bash
docker exec landseer-mysql mysql -u landseer -plandseer landseer_pipeline \
  -e "SHOW TABLES;"
```

Expected output:
```
Tables_in_landseer_pipeline
artifact_files
artifact_nodes
combination_metrics
combination_tools
combinations
datasets
models
output_file_provenance
pipeline_attacks
pipeline_runs
tool_categories
tool_executions
tools
v_best_combinations
v_cache_efficiency
v_combination_results
v_combination_tools_summary
v_pipeline_summary
v_tool_performance
```

### 5. Enable Database Logging

Before running Landseer, source the environment file:

```bash
source .env.db
```

Or set variables manually:
```bash
export LANDSEER_DB_HOST=localhost
export LANDSEER_DB_PORT=3306
export LANDSEER_DB_NAME=landseer_pipeline
export LANDSEER_DB_USER=landseer
export LANDSEER_DB_PASSWORD=landseer
```

### 6. Run Pipeline

```bash
source .env.db
poetry run landseer -c configs/pipeline/your_config.yaml -a configs/attack/test_config_1.yaml
```

Results will be stored in **both** CSV files and MySQL database.

---

## Managing the MySQL Container

### Stop MySQL
```bash
docker stop landseer-mysql
```

### Start MySQL (after stop)
```bash
docker start landseer-mysql
```

### Remove MySQL (deletes all data!)
```bash
docker rm -f landseer-mysql
```

### View MySQL logs
```bash
docker logs landseer-mysql
```

### Connect to MySQL shell
```bash
docker exec -it landseer-mysql mysql -u landseer -plandseer landseer_pipeline
```

---

## Querying Results

### Using MySQL CLI

```bash
# Connect to database
docker exec -it landseer-mysql mysql -u landseer -plandseer landseer_pipeline
```

### Example Queries

#### List Recent Pipeline Runs
```sql
SELECT pipeline_id, run_timestamp, status, 
       total_combinations, successful_combinations, failed_combinations
FROM pipeline_runs
ORDER BY run_timestamp DESC
LIMIT 10;
```

#### Find Best Performing Combinations
```sql
SELECT c.combination_code, cm.acc_test_clean, cm.pgd_accuracy, cm.ood_auc
FROM combinations c
JOIN combination_metrics cm ON c.combination_id = cm.combination_id
WHERE c.status = 'success'
ORDER BY cm.acc_test_clean DESC
LIMIT 10;
```

#### Compare Training Tools
```sql
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
```

#### Find Robust Combinations (good clean + adversarial accuracy)
```sql
SELECT c.combination_code, 
       cm.acc_test_clean, 
       cm.pgd_accuracy,
       (cm.acc_test_clean + cm.pgd_accuracy) / 2 AS robustness_score
FROM combinations c
JOIN combination_metrics cm ON c.combination_id = cm.combination_id
WHERE c.status = 'success'
  AND cm.acc_test_clean >= 0.70
  AND cm.pgd_accuracy >= 0.30
ORDER BY robustness_score DESC
LIMIT 20;
```

#### Cache Efficiency Report
```sql
SELECT * FROM v_cache_efficiency;
```

#### Tool Performance Summary
```sql
SELECT * FROM v_tool_performance;
```

### Using Pre-built Views

The database includes several views for common queries:

| View | Description |
|------|-------------|
| `v_combination_results` | Full results with all metrics per combination |
| `v_combination_tools_summary` | Tools used by each combination, grouped by stage |
| `v_pipeline_summary` | Summary of each pipeline run |
| `v_best_combinations` | Top combinations ranked by robustness score |
| `v_tool_performance` | Aggregate statistics per tool |
| `v_cache_efficiency` | Cache hit rates and time savings |

Example:
```sql
SELECT * FROM v_best_combinations LIMIT 10;
```

### Using Stored Procedures

```sql
-- Find combinations using a specific tool
CALL sp_get_combinations_with_tool('in-trades', 'during_training');

-- Compare two tools
CALL sp_compare_tool_effectiveness('in-trades', 'in-pgd', 'cifar10');

-- Get full pipeline details
CALL sp_get_pipeline_details('3ef5725f725d1dad');

-- Find best combinations for a metric
CALL sp_find_best_combination('cifar10', 'pgd_accuracy', 10);
```

---

## Using Python API

### Query Helper

```python
from landseer_pipeline.database.queries import QueryHelper

helper = QueryHelper("mysql+mysqlconnector://landseer:landseer@localhost/landseer_pipeline")

# Find best combinations
best = helper.find_best_combinations(
    metric="pgd_accuracy",
    dataset="cifar10",
    limit=10
)
for comb in best:
    print(f"{comb['combination_code']}: {comb['pgd_accuracy']:.4f}")

# Compare tools
comparison = helper.compare_tools(
    tools=["in-trades", "in-pgd", "in-free"],
    metrics=["acc_test_clean", "pgd_accuracy"]
)
print(comparison)

# Get run summary
summary = helper.get_run_summary(run_id=1)
print(summary)
```

### Import Existing Results

If you have existing CSV results from previous runs:

```bash
python -m landseer_pipeline.database.importer /path/to/Landseer/results
```

Or in Python:
```python
from landseer_pipeline.database.importer import ResultImporter

importer = ResultImporter("mysql+mysqlconnector://landseer:landseer@localhost/landseer_pipeline")
importer.import_all("/path/to/Landseer/results")
```

---

## Database Schema Overview

### Core Tables

| Table | Description |
|-------|-------------|
| `pipeline_runs` | Main execution records with status, timestamps, config paths |
| `combinations` | Tool combinations per run with execution status |
| `combination_metrics` | All evaluation metrics (accuracy, robustness, privacy, etc.) |
| `combination_tools` | Links combinations to tools (many-to-many) |
| `tool_executions` | Individual tool execution logs with cache info |

### Reference Tables

| Table | Description |
|-------|-------------|
| `datasets` | Dataset registry (cifar10, mnist, etc.) |
| `models` | Model configurations with content hashes |
| `tools` | Available tools by stage |
| `tool_categories` | Tool categorization (adversarial, outlier, etc.) |

### Metrics Stored

| Metric | Column | Description |
|--------|--------|-------------|
| Clean Train Accuracy | `acc_train_clean` | Training set accuracy |
| Clean Test Accuracy | `acc_test_clean` | Test set accuracy |
| PGD Accuracy | `pgd_accuracy` | Accuracy under PGD attack |
| Carlini-Wagner Accuracy | `carlini_l2_accuracy` | Accuracy under C&W L2 attack |
| OOD AUC | `ood_auc` | Out-of-distribution detection AUC |
| Fingerprinting Score | `fingerprinting_score` | Model fingerprinting confidence |
| Attack Success Rate | `attack_success_rate` | Backdoor attack success rate |
| Privacy Epsilon | `privacy_epsilon` | Differential privacy budget |
| MIA AUC | `mia_auc` | Membership inference attack AUC |
| Watermark Accuracy | `watermark_accuracy` | Watermark detection accuracy |

---

## Troubleshooting

### "Database logging not enabled" in logs

Ensure environment variables are set:
```bash
echo $LANDSEER_DB_HOST  # Should show "localhost"
source .env.db
```

### "Connection refused" errors

Check MySQL container is running:
```bash
docker ps | grep landseer-mysql
```

If not running:
```bash
docker start landseer-mysql
```

### "Access denied" errors

Verify credentials:
```bash
docker exec landseer-mysql mysql -u landseer -plandseer -e "SELECT 1;"
```

### Reset database (delete all data)

```bash
docker rm -f landseer-mysql
# Then re-run the Quick Start steps
```

---

## Alternative: Native MySQL Installation

If you prefer to install MySQL natively instead of Docker:

### Arch Linux
```bash
sudo pacman -S mariadb
sudo mariadb-install-db --user=mysql --basedir=/usr --datadir=/var/lib/mysql
sudo systemctl start mariadb
sudo mysql -u root

# In MySQL shell:
CREATE DATABASE landseer_pipeline CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'landseer'@'localhost' IDENTIFIED BY 'landseer';
GRANT ALL PRIVILEGES ON landseer_pipeline.* TO 'landseer'@'localhost';
FLUSH PRIVILEGES;
EXIT;

# Apply schema
mysql -u landseer -plandseer landseer_pipeline < src/landseer_pipeline/database/schema.sql
```

### Ubuntu/Debian
```bash
sudo apt-get install mysql-server mysql-client
sudo systemctl start mysql
# Then follow the same MySQL commands as above
```

---

## Files Reference

| File | Description |
|------|-------------|
| `.env.db` | Environment variables for database connection |
| `src/landseer_pipeline/database/schema.sql` | Full database schema |
| `src/landseer_pipeline/database/models.py` | SQLAlchemy ORM models |
| `src/landseer_pipeline/database/repository.py` | Data access layer |
| `src/landseer_pipeline/database/queries.py` | High-level query helpers |
| `src/landseer_pipeline/database/importer.py` | Import existing CSV results |
| `src/landseer_pipeline/database/README.md` | Developer documentation |
