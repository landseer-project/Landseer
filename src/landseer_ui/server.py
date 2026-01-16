import json
import yaml
import csv
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from typing import Dict, Any, List, Optional
from datetime import datetime

app = FastAPI(title="Landseer Pipeline UI", version="0.2")

BASE_RESULTS_ROOT = Path("results")
ARTIFACT_STORE = Path("artifact_store")
LOGS_DIR = Path("logs")


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def _load_yaml(path: Path):
    try:
        return yaml.safe_load(path.read_text())
    except Exception:
        return None

def _extract_pipeline_metadata(pipeline_id: str, timestamp: str, results_dir: Path):
    """Extract pipeline configuration metadata from logs and files."""
    log_path = LOGS_DIR / f"pipeline_{pipeline_id}_{timestamp}.log"
    metadata = {
        "pipeline_id": pipeline_id,
        "timestamp": timestamp,
        "config_file": None,
        "attack_config": None,
        "dataset": None,
        "model_script": None,
        "framework": None,
        "tools_by_stage": {},
        "total_combinations": 0,
        "success_count": 0,
        "failure_count": 0
    }
    
    # Parse log file for configuration information
    if log_path.exists():
        try:
            with open(log_path, 'r') as f:
                content = f.read()
                
                # Extract config files
                config_match = re.search(r'Config file\s+:\s+(.+)', content)
                if config_match:
                    metadata["config_file"] = Path(config_match.group(1)).name
                
                attack_match = re.search(r'Attack config file\s+:\s+(.+)', content)
                if attack_match:
                    metadata["attack_config"] = Path(attack_match.group(1)).name
                
                # Extract dataset info
                dataset_match = re.search(r'Name\s+:\s+(\w+)', content)
                if dataset_match:
                    metadata["dataset"] = dataset_match.group(1)
                
                # Extract model info
                script_match = re.search(r'Script\s+:\s+(.+)', content)
                if script_match:
                    metadata["model_script"] = Path(script_match.group(1)).name
                
                framework_match = re.search(r'Framework\s+:\s+(\w+)', content)
                if framework_match:
                    metadata["framework"] = framework_match.group(1)
                
                # Extract tool stages
                stage_lines = re.findall(r'(\w+)\s+:\s+(.+)', content)
                for stage, tools in stage_lines:
                    if stage in ['pre_training', 'during_training', 'post_training', 'deployment']:
                        metadata["tools_by_stage"][stage] = tools.strip()
                
                # Extract combination count
                combo_match = re.search(r'Generated (\d+) total combinations', content)
                if combo_match:
                    metadata["total_combinations"] = int(combo_match.group(1))
        except Exception as e:
            print(f"Error parsing log {log_path}: {e}")
    
    # Count successes/failures from CSV
    csv_path = results_dir / "results_combinations.csv"
    if csv_path.exists():
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    status = row.get('combination_status', '').lower()
                    if status == 'success':
                        metadata["success_count"] += 1
                    elif status == 'failure':
                        metadata["failure_count"] += 1
        except Exception as e:
            print(f"Error reading CSV {csv_path}: {e}")
    
    return metadata

@app.get("/api/status")
async def status():
    return {"ok": True}

def _discover_runs():
    """Return list of run metadata with pipeline configs and tool info."""
    runs = []
    if not BASE_RESULTS_ROOT.exists():
        return runs
    
    for pipeline_dir in BASE_RESULTS_ROOT.iterdir():
        if not pipeline_dir.is_dir():
            continue
        for ts_dir in pipeline_dir.iterdir():
            if not ts_dir.is_dir():
                continue
            results_dir = ts_dir
            if (results_dir / "results_tools.csv").exists() or (results_dir / "results_combinations.csv").exists():
                metadata = _extract_pipeline_metadata(pipeline_dir.name, ts_dir.name, results_dir)
                metadata["results_path"] = str(results_dir)
                runs.append(metadata)
    
    return runs

def _latest_run_dir():
    runs = _discover_runs()
    if runs:
        return Path(runs[0]["results_path"])
    return None

@app.get("/api/runs")
async def list_runs(sort_by: str = Query("timestamp", description="Sort by: timestamp, config, dataset, tools")):
    runs = _discover_runs()
    
    # Apply sorting
    if sort_by == "config":
        runs.sort(key=lambda x: x.get("config_file") or "", reverse=False)
    elif sort_by == "dataset":
        runs.sort(key=lambda x: x.get("dataset") or "", reverse=False)
    elif sort_by == "tools":
        runs.sort(key=lambda x: str(x.get("tools_by_stage") or {}), reverse=False)
    else:  # default: timestamp
        runs.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return runs

@app.get("/api/pipeline_configs")
async def list_pipeline_configs():
    """List available pipeline configuration files."""
    configs = []
    config_root = Path("configs/pipeline")
    if config_root.exists():
        for config_file in config_root.rglob("*.yaml"):
            rel_path = config_file.relative_to(config_root)
            config_data = _load_yaml(config_file)
            configs.append({
                "name": str(rel_path),
                "path": str(config_file),
                "dataset": config_data.get("dataset", {}).get("name") if config_data else None,
                "framework": config_data.get("model", {}).get("framework") if config_data else None
            })
    return configs

@app.get("/api/combinations")
async def list_combinations(run: str | None = Query(None, description="Optional run spec pipeline_id:timestamp")):
    # Determine target run directory
    target_dir = None
    if run:
        try:
            pid, ts = run.split(":", 1)
            candidate = BASE_RESULTS_ROOT / pid / ts
            if candidate.exists():
                target_dir = candidate
        except ValueError:
            pass
    if target_dir is None:
        target_dir = _latest_run_dir()
    if target_dir is None:
        return []
    
    combos_csv = target_dir / "results_combinations.csv"
    mapping_file = target_dir / "artifact_mappings.json"
    rows = []
    
    if combos_csv.exists():
        with open(combos_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse tool lists from CSV (they're stored as string representations of lists)
                parsed_row = dict(row)
                for stage in ['pre_training', 'in_training', 'post_training', 'deployment']:
                    if stage in parsed_row:
                        try:
                            # Try to parse as Python list literal
                            tools = eval(parsed_row[stage]) if parsed_row[stage] else []
                            parsed_row[f"{stage}_tools"] = tools
                        except:
                            parsed_row[f"{stage}_tools"] = [parsed_row[stage]] if parsed_row[stage] else []
                rows.append(parsed_row)
    
    # Fallback: if no data rows (only header) use artifact_mappings.json keys
    if (not rows) and mapping_file.exists():
        data = _load_json(mapping_file) or {}
        for combo_id in data.keys():
            rows.append({
                "combination": combo_id,
                "combination_status": "unknown"
            })
    return rows

@app.get("/api/combination/{combination_id}")
async def get_combination(combination_id: str, run: str | None = Query(None, description="Optional run spec pipeline_id:timestamp")):
    # Resolve run directory
    target_dir = None
    if run:
        try:
            pid, ts = run.split(":", 1)
            candidate = BASE_RESULTS_ROOT / pid / ts
            if candidate.exists():
                target_dir = candidate
        except ValueError:
            pass
    if target_dir is None:
        target_dir = _latest_run_dir()
    if target_dir is None:
        raise HTTPException(404, detail="No runs available")
    
    mapping_file = target_dir / "artifact_mappings.json"
    data = _load_json(mapping_file) or {}
    combo = data.get(combination_id)
    if not combo:
        raise HTTPException(404, detail="Combination not found")
    return combo

@app.get("/api/run/{pipeline_id}/{timestamp}")
async def get_run_details(pipeline_id: str, timestamp: str):
    """Get detailed information about a specific run."""
    results_dir = BASE_RESULTS_ROOT / pipeline_id / timestamp
    if not results_dir.exists():
        raise HTTPException(404, detail="Run not found")
    
    metadata = _extract_pipeline_metadata(pipeline_id, timestamp, results_dir)
    
    # Add combination details
    combos_csv = results_dir / "results_combinations.csv"
    combinations = []
    if combos_csv.exists():
        with open(combos_csv) as f:
            reader = csv.DictReader(f)
            combinations = list(reader)
    
    metadata["combinations"] = combinations
    return metadata

@app.get("/api/artifact/{node_hash}/manifest")
async def get_artifact_manifest(node_hash: str):
    manifest = ARTIFACT_STORE / node_hash / "manifest.json"
    if not manifest.exists():
        raise HTTPException(404, detail="Manifest not found")
    return _load_json(manifest) or {}

@app.get("/api/artifact/{node_hash}/file/{path:path}")
async def get_artifact_file(node_hash: str, path: str):
    file_path = ARTIFACT_STORE / node_hash / "output" / path
    if not file_path.exists():
        raise HTTPException(404, detail="File not found")
    return FileResponse(file_path)

@app.get("/api/log/{combination_id}/{stage_tool}")
async def get_tool_log(combination_id: str, stage_tool: str, run: str | None = Query(None)):
    target_dir = None
    if run:
        try:
            pid, ts = run.split(":", 1)
            candidate = BASE_RESULTS_ROOT / pid / ts
            if candidate.exists():
                target_dir = candidate
        except ValueError:
            pass
    if target_dir is None:
        target_dir = _latest_run_dir()
    if target_dir is None:
        raise HTTPException(404, detail="No runs available")
    log_file = target_dir / "tool_logs" / combination_id / f"{stage_tool}.log"
    if not log_file.exists():
        raise HTTPException(404, detail="Log not found")
    return FileResponse(log_file)

@app.get("/api/log/{combination_id}/{stage_tool}/content")
async def get_tool_log_content(combination_id: str, stage_tool: str, run: str | None = Query(None)):
    """Get log content as JSON for display in UI"""
    target_dir = None
    if run:
        try:
            pid, ts = run.split(":", 1)
            candidate = BASE_RESULTS_ROOT / pid / ts
            if candidate.exists():
                target_dir = candidate
        except ValueError:
            pass
    if target_dir is None:
        target_dir = _latest_run_dir()
    if target_dir is None:
        raise HTTPException(404, detail="No runs available")
    
    log_file = target_dir / "tool_logs" / combination_id / f"{stage_tool}.log"
    if not log_file.exists():
        raise HTTPException(404, detail="Log not found")
    
    try:
        content = log_file.read_text(encoding='utf-8', errors='replace')
        # Limit content size for UI display
        if len(content) > 50000:  # 50KB limit
            content = content[-50000:] + "\n... (truncated to last 50KB)"
        
        return {
            "combination": combination_id,
            "tool": stage_tool,
            "content": content,
            "size": log_file.stat().st_size,
            "path": str(log_file)
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Error reading log file: {str(e)}")

@app.get("/api/logs/{combination_id}")
async def get_combination_logs(combination_id: str, run: str | None = Query(None)):
    """Get all available logs for a combination"""
    target_dir = None
    if run:
        try:
            pid, ts = run.split(":", 1)
            candidate = BASE_RESULTS_ROOT / pid / ts
            if candidate.exists():
                target_dir = candidate
        except ValueError:
            pass
    if target_dir is None:
        target_dir = _latest_run_dir()
    if target_dir is None:
        raise HTTPException(404, detail="No runs available")
    
    logs_dir = target_dir / "tool_logs" / combination_id
    if not logs_dir.exists():
        return {"logs": []}
    
    logs = []
    for log_file in logs_dir.glob("*.log"):
        try:
            stat = log_file.stat()
            logs.append({
                "name": log_file.stem,
                "path": str(log_file),
                "size": stat.st_size,
                "modified": stat.st_mtime
            })
        except Exception:
            continue
    
    # Sort by modification time, most recent first
    logs.sort(key=lambda x: x["modified"], reverse=True)
    
    return {"logs": logs}


@app.get("/api/metrics/{run_id}")
async def get_metrics(run_id: str):
    """Get metrics for a specific pipeline run."""
    try:
        # Parse run_id format: pipeline_id:timestamp
        if ':' in run_id:
            pipeline_id, timestamp = run_id.split(':', 1)
        else:
            raise HTTPException(status_code=400, detail="Invalid run_id format. Expected 'pipeline_id:timestamp'")
        
        results_dir = BASE_RESULTS_ROOT / pipeline_id / timestamp
        if not results_dir.exists():
            raise HTTPException(status_code=404, detail="Pipeline run not found")
        
        # Load combination metrics
        combinations_csv = results_dir / "results_combinations.csv"
        tools_csv = results_dir / "results_tools.csv"
        
        metrics_data = {
            "pipeline_id": pipeline_id,
            "timestamp": timestamp,
            "combinations": [],
            "tools": [],
            "summary": {}
        }
        
        # Parse combinations CSV
        if combinations_csv.exists():
            with open(combinations_csv, 'r') as f:
                reader = csv.DictReader(f)
                combinations = list(reader)
                metrics_data["combinations"] = combinations
                
                # Calculate summary statistics
                total_combinations = len(combinations)
                successful_combinations = len([c for c in combinations if c.get("combination_status") == "success"])
                failed_combinations = total_combinations - successful_combinations
                
                # Calculate average metrics for successful combinations
                successful_combos = [c for c in combinations if c.get("combination_status") == "success"]
                if successful_combos:
                    def safe_float(value):
                        try:
                            return float(value) if value and value != '-1' else None
                        except:
                            return None
                    
                    # Calculate averages for key metrics
                    acc_train_values = [safe_float(c.get("acc_train_clean")) for c in successful_combos]
                    acc_test_values = [safe_float(c.get("acc_test_clean")) for c in successful_combos]
                    pgd_acc_values = [safe_float(c.get("pgd_acc")) for c in successful_combos]
                    carlini_acc_values = [safe_float(c.get("carlini_acc")) for c in successful_combos]
                    
                    acc_train_values = [v for v in acc_train_values if v is not None]
                    acc_test_values = [v for v in acc_test_values if v is not None]
                    pgd_acc_values = [v for v in pgd_acc_values if v is not None]
                    carlini_acc_values = [v for v in carlini_acc_values if v is not None]
                    
                    metrics_data["summary"] = {
                        "total_combinations": total_combinations,
                        "successful_combinations": successful_combinations,
                        "failed_combinations": failed_combinations,
                        "success_rate": successful_combinations / total_combinations if total_combinations > 0 else 0,
                        "avg_train_accuracy": sum(acc_train_values) / len(acc_train_values) if acc_train_values else None,
                        "avg_test_accuracy": sum(acc_test_values) / len(acc_test_values) if acc_test_values else None,
                        "avg_pgd_accuracy": sum(pgd_acc_values) / len(pgd_acc_values) if pgd_acc_values else None,
                        "avg_carlini_accuracy": sum(carlini_acc_values) / len(carlini_acc_values) if carlini_acc_values else None,
                        "total_duration": sum([float(c.get("total_duration", 0)) for c in combinations if c.get("total_duration")])
                    }
        
        # Parse tools CSV  
        if tools_csv.exists():
            with open(tools_csv, 'r') as f:
                reader = csv.DictReader(f)
                metrics_data["tools"] = list(reader)
        
        return metrics_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading metrics: {str(e)}")


@app.get("/api/metrics/comparison")
async def compare_metrics(run_ids: List[str] = Query(...)):
    """Compare metrics across multiple pipeline runs."""
    try:
        comparison_data = {
            "runs": [],
            "comparison_summary": {}
        }
        
        for run_id in run_ids:
            try:
                metrics = await get_metrics(run_id)
                comparison_data["runs"].append({
                    "run_id": run_id,
                    "metrics": metrics
                })
            except Exception as e:
                comparison_data["runs"].append({
                    "run_id": run_id,
                    "error": str(e)
                })
        
        # Generate comparison summary
        successful_runs = [r for r in comparison_data["runs"] if "error" not in r]
        if len(successful_runs) >= 2:
            summaries = [r["metrics"]["summary"] for r in successful_runs]
            
            comparison_data["comparison_summary"] = {
                "best_test_accuracy": {
                    "run_id": max(successful_runs, key=lambda x: x["metrics"]["summary"].get("avg_test_accuracy", 0))["run_id"],
                    "value": max([s.get("avg_test_accuracy", 0) for s in summaries if s.get("avg_test_accuracy") is not None])
                },
                "best_success_rate": {
                    "run_id": max(successful_runs, key=lambda x: x["metrics"]["summary"].get("success_rate", 0))["run_id"],
                    "value": max([s.get("success_rate", 0) for s in summaries])
                },
                "fastest_run": {
                    "run_id": min(successful_runs, key=lambda x: x["metrics"]["summary"].get("total_duration", float('inf')))["run_id"],
                    "value": min([s.get("total_duration", float('inf')) for s in summaries if s.get("total_duration") is not None])
                }
            }
        
        return comparison_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing metrics: {str(e)}")


@app.delete("/api/run/{pipeline_id}/{timestamp}")
async def delete_run(pipeline_id: str, timestamp: str):
    """Delete a specific pipeline run and all its associated data."""
    import shutil
    
    try:
        # Convert timestamp format if needed (remove dashes)
        clean_timestamp = timestamp.replace('-', '')
        
        # Construct paths to delete
        run_dir = BASE_RESULTS_ROOT / pipeline_id / clean_timestamp
        log_file = LOGS_DIR / f"pipeline_{pipeline_id}_{clean_timestamp}.log"
        
        deleted_items = []
        
        # Delete run directory if it exists
        if run_dir.exists():
            shutil.rmtree(run_dir)
            deleted_items.append(f"Run directory: {run_dir}")
        
        # Delete log file if it exists
        if log_file.exists():
            log_file.unlink()
            deleted_items.append(f"Log file: {log_file}")
        
        # Check if pipeline directory is now empty and delete if so
        pipeline_dir = BASE_RESULTS_ROOT / pipeline_id
        if pipeline_dir.exists() and not any(pipeline_dir.iterdir()):
            pipeline_dir.rmdir()
            deleted_items.append(f"Empty pipeline directory: {pipeline_dir}")
        
        if not deleted_items:
            raise HTTPException(status_code=404, detail=f"Run {pipeline_id}:{timestamp} not found")
        
        return {
            "message": f"Successfully deleted run {pipeline_id}:{timestamp}",
            "deleted_items": deleted_items
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting run: {str(e)}")


@app.get("/api/pipeline-log/{pipeline_id}/{timestamp}")
async def get_pipeline_log(pipeline_id: str, timestamp: str):
    """Get the main pipeline log file for a specific run."""
    try:
        # Convert timestamp format if needed (remove dashes)
        clean_timestamp = timestamp.replace('-', '')
        
        log_file = LOGS_DIR / f"pipeline_{pipeline_id}_{clean_timestamp}.log"
        
        if not log_file.exists():
            raise HTTPException(status_code=404, detail="Pipeline log file not found")
        
        # Read the log file
        log_content = log_file.read_text(encoding='utf-8', errors='replace')
        
        return {
            "pipeline_id": pipeline_id,
            "timestamp": timestamp,
            "log_file": str(log_file),
            "content": log_content,
            "size": len(log_content)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading pipeline log: {str(e)}")


# Simple static front-end (can be expanded later)
INDEX_HTML = (Path(__file__).parent / "index.html")

@app.get("/")
async def root():
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return {"message": "Landseer UI API"}
