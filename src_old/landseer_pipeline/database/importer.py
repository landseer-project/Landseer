"""
Result Importer for Landseer Pipeline Database

Imports existing CSV/JSON results from pipeline runs into the MySQL database.
"""

import os
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import re

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import (
    Base, Dataset, Model, Tool, PipelineRun, PipelineAttacks,
    Combination, CombinationTools, CombinationMetrics, ToolExecution,
    ArtifactNode, ArtifactFile, OutputFileProvenance,
    StageEnum, RunStatusEnum, CombinationStatusEnum, ExecutionStatusEnum
)
from .repository import (
    DatasetRepository, ModelRepository, ToolRepository,
    PipelineRunRepository, CombinationRepository, MetricsRepository,
    ToolExecutionRepository, ArtifactRepository
)

logger = logging.getLogger(__name__)


class ResultImporter:
    """
    Imports pipeline results from filesystem into MySQL database.
    
    Usage:
        importer = ResultImporter("mysql+mysqlconnector://user:pass@host/db")
        importer.import_run("/path/to/results/pipeline_id/timestamp")
        # or
        importer.import_all("/path/to/results")
    """
    
    def __init__(self, database_url: str):
        """
        Initialize the importer.
        
        Args:
            database_url: SQLAlchemy database URL
        """
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Ensure tables exist
        Base.metadata.create_all(bind=self.engine)
    
    def _parse_tools_list(self, tools_str: str) -> List[str]:
        """Parse a string representation of tools list."""
        if not tools_str or tools_str == '[]':
            return []
        
        # Remove brackets and quotes
        cleaned = tools_str.strip("[]").replace("'", "").replace('"', '')
        if not cleaned:
            return []
        
        return [t.strip() for t in cleaned.split(',') if t.strip()]
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime."""
        # Try different formats
        formats = [
            "%Y%m%d%H%M%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Cannot parse timestamp: {timestamp_str}")
    
    def _ensure_tools_exist(
        self,
        session,
        tools_by_stage: Dict[str, List[str]]
    ) -> Dict[str, Tool]:
        """Ensure all tools exist in database, return mapping."""
        tool_repo = ToolRepository(session)
        tool_map = {}
        
        for stage_name, tool_names in tools_by_stage.items():
            stage = StageEnum(stage_name)
            for tool_name in tool_names:
                if tool_name not in tool_map:
                    tool = tool_repo.get_or_create(tool_name, stage)
                    tool_map[tool_name] = tool
        
        return tool_map
    
    def import_run(
        self,
        run_path: str,
        pipeline_id: Optional[str] = None
    ) -> Optional[int]:
        """
        Import a single pipeline run.
        
        Args:
            run_path: Path to the run directory (e.g., results/pipeline_id/timestamp)
            pipeline_id: Override pipeline ID (otherwise extracted from path)
        
        Returns:
            run_id if successful, None otherwise
        """
        run_path = Path(run_path)
        
        if not run_path.exists():
            logger.error(f"Run path does not exist: {run_path}")
            return None
        
        # Extract pipeline_id and timestamp from path
        if pipeline_id is None:
            pipeline_id = run_path.parent.name
        timestamp_str = run_path.name
        
        try:
            run_timestamp = self._parse_timestamp(timestamp_str)
        except ValueError as e:
            logger.error(f"Cannot parse timestamp from path: {e}")
            return None
        
        # Check for results files
        results_csv = run_path / "results_combinations.csv"
        if not results_csv.exists():
            logger.warning(f"No results_combinations.csv found in {run_path}")
            return None
        
        session = self.SessionLocal()
        try:
            # Create repositories
            dataset_repo = DatasetRepository(session)
            model_repo = ModelRepository(session)
            run_repo = PipelineRunRepository(session)
            comb_repo = CombinationRepository(session)
            metrics_repo = MetricsRepository(session)
            tool_repo = ToolRepository(session)
            
            # Read results to extract dataset info
            with open(results_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if not rows:
                logger.warning(f"Empty results file: {results_csv}")
                return None
            
            # Get dataset info from first row
            first_row = rows[0]
            dataset_name = first_row.get('dataset', 'unknown')
            dataset_variant = first_row.get('variant', 'clean')
            
            # Get or create dataset
            dataset = dataset_repo.get_or_create(
                name=dataset_name,
                variant=dataset_variant
            )
            
            # Check if run already exists
            existing = run_repo.get_by_pipeline_id(pipeline_id)
            for run in existing:
                if run.run_timestamp == run_timestamp:
                    logger.info(f"Run already exists: {pipeline_id}/{timestamp_str}")
                    return run.run_id
            
            # Create pipeline run
            status = RunStatusEnum.COMPLETED
            success_marker = run_path / ".success"
            failed_marker = run_path / ".failed"
            if failed_marker.exists():
                status = RunStatusEnum.FAILED
            elif not success_marker.exists():
                status = RunStatusEnum.PARTIAL
            
            # Count results
            successful = sum(1 for r in rows if r.get('combination_status') == 'success')
            failed = sum(1 for r in rows if r.get('combination_status') == 'failure')
            
            pipeline_run = run_repo.create(
                pipeline_id=pipeline_id,
                run_timestamp=run_timestamp,
                dataset=dataset,
                total_combinations=len(rows),
                successful_combinations=successful,
                failed_combinations=failed,
                status=status
            )
            
            # Import attack config if exists
            attack_config_path = run_path.parent.parent / "configs" / "attack"
            # (Would need to parse attack config YAML if needed)
            
            # Collect all tools first
            all_tools_by_stage = {
                'pre_training': set(),
                'during_training': set(),
                'post_training': set(),
                'deployment': set(),
            }
            
            for row in rows:
                for stage in all_tools_by_stage.keys():
                    tools = self._parse_tools_list(row.get(stage, '[]'))
                    all_tools_by_stage[stage].update(tools)
            
            # Ensure all tools exist
            for stage_name, tool_names in all_tools_by_stage.items():
                stage = StageEnum(stage_name)
                for tool_name in tool_names:
                    tool_repo.get_or_create(tool_name, stage)
            
            session.flush()
            
            # Import combinations and metrics
            for idx, row in enumerate(rows):
                comb_code = row.get('combination_id', f'comb_{idx:03d}')
                
                tools_by_stage = {
                    'pre_training': self._parse_tools_list(row.get('pre_training', '[]')),
                    'during_training': self._parse_tools_list(row.get('during_training', '[]')),
                    'post_training': self._parse_tools_list(row.get('post_training', '[]')),
                    'deployment': self._parse_tools_list(row.get('deployment', '[]')),
                }
                
                comb_status = CombinationStatusEnum.SUCCESS
                if row.get('combination_status') == 'failure':
                    comb_status = CombinationStatusEnum.FAILURE
                elif row.get('combination_status') == 'skipped':
                    comb_status = CombinationStatusEnum.SKIPPED
                
                duration = None
                if row.get('total_duration'):
                    try:
                        duration = float(row['total_duration'])
                    except (ValueError, TypeError):
                        pass
                
                combination = comb_repo.create(
                    run=pipeline_run,
                    combination_code=comb_code,
                    combination_index=idx,
                    tools_by_stage=tools_by_stage,
                    status=comb_status,
                    total_duration_sec=duration
                )
                
                # Import metrics
                metrics = {}
                metric_mappings = {
                    'acc_train_clean': 'acc_train_clean',
                    'acc_test_clean': 'acc_test_clean',
                    'pgd_acc': 'pgd_accuracy',
                    'pgd_accuracy': 'pgd_accuracy',
                    'carlini_acc': 'carlini_l2_accuracy',
                    'carlini_l2_accuracy': 'carlini_l2_accuracy',
                    'ood_auc': 'ood_auc',
                    'fingerprinting': 'fingerprinting_score',
                    'fingerprinting_score': 'fingerprinting_score',
                    'asr': 'attack_success_rate',
                    'attack_success_rate': 'attack_success_rate',
                    'privacy_epsilon': 'privacy_epsilon',
                    'dp_accuracy': 'dp_accuracy',
                    'watermark_accuracy': 'watermark_accuracy',
                    'mia_auc': 'mia_auc',
                    'eps_estimate': 'eps_estimate',
                }
                
                for csv_key, db_key in metric_mappings.items():
                    if csv_key in row and row[csv_key]:
                        try:
                            val = float(row[csv_key])
                            if val != -1:  # -1 means not computed
                                metrics[db_key] = val
                        except (ValueError, TypeError):
                            pass
                
                if metrics:
                    metrics_repo.create_or_update(
                        combination_id=combination.combination_id,
                        **metrics
                    )
            
            # Import tool execution details if available
            tools_csv = run_path / "results_tools.csv"
            if tools_csv.exists():
                self._import_tool_executions(session, tools_csv, pipeline_run.run_id)
            
            # Import artifact mappings if available
            artifact_mappings = run_path / "artifact_mappings.json"
            if artifact_mappings.exists():
                self._import_artifact_mappings(session, artifact_mappings, pipeline_run.run_id)
            
            session.commit()
            logger.info(f"Successfully imported run: {pipeline_id}/{timestamp_str} (id={pipeline_run.run_id})")
            return pipeline_run.run_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to import run {run_path}: {e}", exc_info=True)
            return None
        finally:
            session.close()
    
    def _import_tool_executions(
        self,
        session,
        tools_csv: Path,
        run_id: int
    ) -> None:
        """Import tool execution records."""
        exec_repo = ToolExecutionRepository(session)
        tool_repo = ToolRepository(session)
        comb_repo = CombinationRepository(session)
        
        with open(tools_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                comb_code = row.get('combination_id')
                tool_name = row.get('tool_name')
                stage_str = row.get('stage')
                
                if not all([comb_code, tool_name, stage_str]):
                    continue
                
                # Get combination
                comb = comb_repo.get_by_code(run_id, comb_code)
                if not comb:
                    continue
                
                # Get tool
                tool = tool_repo.get_by_name(tool_name)
                if not tool:
                    continue
                
                try:
                    stage = StageEnum(stage_str)
                except ValueError:
                    continue
                
                status = ExecutionStatusEnum.SUCCESS
                if row.get('status') == 'failure':
                    status = ExecutionStatusEnum.FAILURE
                elif row.get('cache_hit', 'false').lower() == 'true':
                    status = ExecutionStatusEnum.CACHED
                
                duration = None
                if row.get('duration_sec'):
                    try:
                        duration = float(row['duration_sec'])
                    except (ValueError, TypeError):
                        pass
                
                exec_repo.create(
                    combination_id=comb.combination_id,
                    tool_id=tool.tool_id,
                    stage=stage,
                    cache_key=row.get('cache_key'),
                    cache_hit=row.get('cache_hit', 'false').lower() == 'true',
                    status=status,
                    duration_sec=duration,
                    output_path=row.get('output_path')
                )
    
    def _import_artifact_mappings(
        self,
        session,
        mappings_path: Path,
        run_id: int
    ) -> None:
        """Import artifact mappings."""
        artifact_repo = ArtifactRepository(session)
        comb_repo = CombinationRepository(session)
        
        with open(mappings_path, 'r') as f:
            mappings = json.load(f)
        
        for comb_code, stages in mappings.items():
            comb = comb_repo.get_by_code(run_id, comb_code)
            if not comb:
                continue
            
            for stage_name, tool_entries in stages.items():
                if not isinstance(tool_entries, list):
                    continue
                
                for entry in tool_entries:
                    node_hash = entry.get('node_hash')
                    if not node_hash:
                        continue
                    
                    try:
                        stage = StageEnum(stage_name)
                    except ValueError:
                        continue
                    
                    artifact_repo.get_or_create(
                        node_hash=node_hash,
                        tool_name=entry.get('tool'),
                        stage=stage
                    )
    
    def import_all(self, results_dir: str) -> Dict[str, int]:
        """
        Import all pipeline runs from a results directory.
        
        Args:
            results_dir: Path to the results directory
        
        Returns:
            Dictionary mapping pipeline_id/timestamp to run_id
        """
        results_path = Path(results_dir)
        imported = {}
        
        if not results_path.exists():
            logger.error(f"Results directory does not exist: {results_dir}")
            return imported
        
        # Iterate over pipeline directories
        for pipeline_dir in results_path.iterdir():
            if not pipeline_dir.is_dir():
                continue
            
            pipeline_id = pipeline_dir.name
            
            # Iterate over timestamp directories
            for run_dir in pipeline_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                
                run_id = self.import_run(str(run_dir), pipeline_id)
                if run_id:
                    key = f"{pipeline_id}/{run_dir.name}"
                    imported[key] = run_id
        
        logger.info(f"Imported {len(imported)} pipeline runs")
        return imported


def main():
    """CLI entry point for result importer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Import pipeline results to MySQL database")
    parser.add_argument("results_path", help="Path to results directory or specific run")
    parser.add_argument("--database-url", "-d", 
                       default="mysql+mysqlconnector://landseer:landseer@localhost/landseer_pipeline",
                       help="Database connection URL")
    parser.add_argument("--single-run", "-s", action="store_true",
                       help="Import a single run instead of all runs")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    importer = ResultImporter(args.database_url)
    
    if args.single_run:
        run_id = importer.import_run(args.results_path)
        if run_id:
            print(f"Successfully imported run with ID: {run_id}")
        else:
            print("Failed to import run")
            exit(1)
    else:
        imported = importer.import_all(args.results_path)
        print(f"Imported {len(imported)} runs:")
        for key, run_id in imported.items():
            print(f"  {key}: {run_id}")


if __name__ == "__main__":
    main()
