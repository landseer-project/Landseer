"""
Database-enabled Result Logger for Landseer Pipeline

This extends the ResultLogger to also store results in MySQL database
for easier querying and analysis.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from ..utils.result_logger import ResultLogger

logger = logging.getLogger(__name__)

# Database imports - optional, graceful fallback if not available
try:
    from .db_connection import DatabaseConfig, DatabaseConnection
    from .models import (
        Base, Dataset, Model, Tool, PipelineRun, Combination,
        CombinationMetrics, ToolExecution, StageEnum,
        RunStatusEnum, CombinationStatusEnum, ExecutionStatusEnum,
        create_session
    )
    from .repository import (
        DatasetRepository, ModelRepository, ToolRepository,
        PipelineRunRepository, CombinationRepository, MetricsRepository,
        ToolExecutionRepository
    )
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    DB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Database module not available: {e}. Results will only be saved to CSV.")
    DB_AVAILABLE = False


class DatabaseResultLogger(ResultLogger):
    """
    Extended ResultLogger that writes to both CSV files and MySQL database.
    
    Falls back to CSV-only if database is not configured or available.
    
    Usage:
        logger = DatabaseResultLogger(
            results_dir="/path/to/results",
            pipeline_id="abc123",
            dataset_name="cifar10",
            dataset_variant="clean",
            config_file_path="/path/to/config.yaml"
        )
        
        # Start the run
        logger.start_run()
        
        # Log combinations and tools as before
        logger.log_combination(...)
        logger.log_tool(...)
        
        # Finish the run
        logger.finish_run(status="completed")
    """
    
    def __init__(
        self,
        results_dir,
        pipeline_id: str,
        dataset_name: str = "unknown",
        dataset_variant: str = "clean",
        config_file_path: Optional[str] = None,
        attack_config_path: Optional[str] = None,
        model_script_path: Optional[str] = None,
        model_content_hash: Optional[str] = None,
        database_url: Optional[str] = None,
        enable_database: bool = True
    ):
        """
        Initialize the database-enabled result logger.
        
        Args:
            results_dir: Directory for CSV result files
            pipeline_id: Unique pipeline identifier (hash)
            dataset_name: Name of dataset (e.g., "cifar10")
            dataset_variant: Dataset variant ("clean" or "poisoned")
            config_file_path: Path to pipeline config file
            attack_config_path: Path to attack config file
            model_script_path: Path to model script
            model_content_hash: Hash of model script content
            database_url: SQLAlchemy database URL (or use env vars)
            enable_database: Whether to enable database logging
        """
        # Initialize parent CSV logger
        super().__init__(results_dir, pipeline_id)
        
        self.dataset_name = dataset_name
        self.dataset_variant = dataset_variant
        self.config_file_path = config_file_path
        self.attack_config_path = attack_config_path
        self.model_script_path = model_script_path
        self.model_content_hash = model_content_hash
        
        # Database state
        self.db_enabled = enable_database and DB_AVAILABLE
        self.session = None
        self.run_record: Optional[PipelineRun] = None
        self.combination_records: Dict[str, Combination] = {}
        self.tool_cache: Dict[str, Tool] = {}
        
        # Build database URL from environment if not provided
        if self.db_enabled and database_url is None:
            database_url = self._build_database_url()
        
        self.database_url = database_url
        
        if self.db_enabled and self.database_url:
            try:
                self._init_database()
                logger.info(f"Database logging enabled for pipeline {pipeline_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize database, falling back to CSV only: {e}")
                self.db_enabled = False
    
    def _build_database_url(self) -> Optional[str]:
        """Build database URL from environment variables."""
        host = os.getenv("LANDSEER_DB_HOST")
        if not host:
            return None
        
        port = os.getenv("LANDSEER_DB_PORT", "3306")
        user = os.getenv("LANDSEER_DB_USER", "landseer")
        password = os.getenv("LANDSEER_DB_PASSWORD", "")
        database = os.getenv("LANDSEER_DB_NAME", "landseer_pipeline")
        
        return f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    
    def _init_database(self):
        """Initialize database connection and session."""
        engine = create_engine(self.database_url, echo=False, pool_pre_ping=True)
        
        # Ensure tables exist
        Base.metadata.create_all(bind=engine)
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        self.session = SessionLocal()
    
    def _get_or_create_tool(self, tool_name: str, stage: str) -> Optional[Tool]:
        """Get or create a tool record."""
        if not self.db_enabled or not self.session:
            return None
        
        cache_key = f"{tool_name}:{stage}"
        if cache_key in self.tool_cache:
            return self.tool_cache[cache_key]
        
        try:
            tool_repo = ToolRepository(self.session)
            stage_enum = StageEnum(stage)
            tool = tool_repo.get_or_create(tool_name, stage_enum)
            self.tool_cache[cache_key] = tool
            return tool
        except Exception as e:
            logger.warning(f"Failed to get/create tool {tool_name}: {e}")
            return None
    
    def start_run(self, total_combinations: int = 0):
        """
        Start a new pipeline run record in the database.
        
        Args:
            total_combinations: Expected number of combinations
        """
        if not self.db_enabled or not self.session:
            return
        
        try:
            # Get or create dataset
            dataset_repo = DatasetRepository(self.session)
            dataset = dataset_repo.get_or_create(
                name=self.dataset_name,
                variant=self.dataset_variant
            )
            
            # Get or create model if we have the info
            model = None
            if self.model_script_path and self.model_content_hash:
                model_repo = ModelRepository(self.session)
                model = model_repo.get_or_create(
                    script_path=self.model_script_path,
                    content_hash=self.model_content_hash
                )
            
            # Create pipeline run
            run_repo = PipelineRunRepository(self.session)
            self.run_record = run_repo.create(
                pipeline_id=self.pipeline_id,
                run_timestamp=datetime.now(),
                dataset=dataset,
                model=model,
                config_file_path=self.config_file_path,
                attack_config_path=self.attack_config_path,
                total_combinations=total_combinations
            )
            
            self.session.commit()
            logger.info(f"Created database run record: {self.run_record.run_id}")
            
        except Exception as e:
            logger.error(f"Failed to create run record: {e}")
            self.session.rollback()
    
    def log_tool(self, combination, stage, tool_name, cache_key, output_path, duration, status):
        """Log a tool execution to both CSV and database."""
        # Always log to CSV
        super().log_tool(combination, stage, tool_name, cache_key, output_path, duration, status)
        
        # Log to database
        if not self.db_enabled or not self.session or not self.run_record:
            return
        
        try:
            # Get the combination record
            comb_record = self.combination_records.get(combination)
            if not comb_record:
                return
            
            # Get the tool record
            tool = self._get_or_create_tool(tool_name, stage)
            if not tool:
                return
            
            # Map status
            exec_status = ExecutionStatusEnum.SUCCESS
            if status == "failure" or status == "failed":
                exec_status = ExecutionStatusEnum.FAILURE
            elif status == "cached":
                exec_status = ExecutionStatusEnum.CACHED
            
            # Create execution record
            exec_repo = ToolExecutionRepository(self.session)
            exec_repo.create(
                combination_id=comb_record.combination_id,
                tool_id=tool.tool_id,
                stage=StageEnum(stage),
                cache_key=cache_key,
                cache_hit=(status == "cached"),
                status=exec_status,
                duration_sec=duration,
                output_path=str(output_path) if output_path else None
            )
            
            self.session.flush()
            
        except Exception as e:
            logger.warning(f"Failed to log tool execution to database: {e}")
    
    def log_combination(self, combination, tools_by_stage, dataset_name, dataset_type, acc, duration, status: str):
        """Log a combination result to both CSV and database."""
        # Always log to CSV
        super().log_combination(combination, tools_by_stage, dataset_name, dataset_type, acc, duration, status)
        
        # Log to database
        if not self.db_enabled or not self.session or not self.run_record:
            return
        
        try:
            # Extract tool names by stage
            def extract_names(tools):
                if not tools:
                    return []
                return [tool.name if hasattr(tool, 'name') else str(tool) for tool in tools]
            
            tools_dict = {
                'pre_training': extract_names(tools_by_stage.get("pre_training", [])),
                'during_training': extract_names(tools_by_stage.get("during_training", [])),
                'post_training': extract_names(tools_by_stage.get("post_training", [])),
                'deployment': extract_names(tools_by_stage.get("deployment", [])),
            }
            
            # Ensure all tools exist in database
            for stage, tool_names in tools_dict.items():
                for tool_name in tool_names:
                    self._get_or_create_tool(tool_name, stage)
            
            # Map status
            comb_status = CombinationStatusEnum.SUCCESS
            if status == "failure" or status == "failed":
                comb_status = CombinationStatusEnum.FAILURE
            elif status == "skipped":
                comb_status = CombinationStatusEnum.SKIPPED
            
            # Parse combination index from code (e.g., "comb_010" -> 10)
            try:
                comb_index = int(combination.split("_")[1])
            except (IndexError, ValueError):
                comb_index = len(self.combination_records)
            
            # Create combination record
            comb_repo = CombinationRepository(self.session)
            comb_record = comb_repo.create(
                run=self.run_record,
                combination_code=combination,
                combination_index=comb_index,
                tools_by_stage=tools_dict,
                status=comb_status,
                total_duration_sec=duration
            )
            
            self.combination_records[combination] = comb_record
            
            # Create metrics record
            metrics_repo = MetricsRepository(self.session)
            
            # Helper to convert -1 to None
            def clean_metric(val):
                if val == -1 or val is None:
                    return None
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None
            
            metrics_repo.create_or_update(
                combination_id=comb_record.combination_id,
                acc_train_clean=clean_metric(acc.get("clean_train_accuracy")),
                acc_test_clean=clean_metric(acc.get("clean_test_accuracy")),
                pgd_accuracy=clean_metric(acc.get("pgd_acc")),
                carlini_l2_accuracy=clean_metric(acc.get("carlini_l2_accuracy")),
                ood_auc=clean_metric(acc.get("ood_auc")),
                fingerprinting_score=clean_metric(acc.get("fingerprinting")),
                attack_success_rate=clean_metric(acc.get("backdoor_asr")),
                privacy_epsilon=clean_metric(acc.get("privacy_epsilon")),
                dp_accuracy=clean_metric(acc.get("dp_accuracy")),
                watermark_accuracy=clean_metric(acc.get("watermark_accuracy")),
                mia_auc=clean_metric(acc.get("mia_auc")),
                eps_estimate=clean_metric(acc.get("eps_estimate")),
            )
            
            self.session.commit()
            
            # Update run counts
            if comb_status == CombinationStatusEnum.SUCCESS:
                self.run_record.successful_combinations = (
                    self.run_record.successful_combinations or 0
                ) + 1
            elif comb_status == CombinationStatusEnum.FAILURE:
                self.run_record.failed_combinations = (
                    self.run_record.failed_combinations or 0
                ) + 1
            
            self.session.flush()
            
        except Exception as e:
            logger.error(f"Failed to log combination to database: {e}")
            self.session.rollback()
    
    def finish_run(self, status: str = "completed", error_message: Optional[str] = None):
        """
        Finish the pipeline run and update the database record.
        
        Args:
            status: Final status ("completed", "failed", "partial")
            error_message: Optional error message if failed
        """
        if not self.db_enabled or not self.session or not self.run_record:
            return
        
        try:
            # Map status
            if status == "completed":
                run_status = RunStatusEnum.COMPLETED
            elif status == "failed":
                run_status = RunStatusEnum.FAILED
            else:
                run_status = RunStatusEnum.PARTIAL
            
            # Update run record
            self.run_record.status = run_status
            self.run_record.completed_at = datetime.now()
            if error_message:
                self.run_record.error_message = error_message
            
            # Calculate total duration if we have start and end times
            if self.run_record.created_at and self.run_record.completed_at:
                delta = self.run_record.completed_at - self.run_record.created_at
                self.run_record.total_duration_sec = delta.total_seconds()
            
            self.session.commit()
            logger.info(f"Pipeline run {self.run_record.run_id} finished with status: {status}")
            
        except Exception as e:
            logger.error(f"Failed to finish run record: {e}")
            self.session.rollback()
    
    def close(self):
        """Close the database session."""
        if self.session:
            try:
                self.session.close()
            except Exception as e:
                logger.warning(f"Error closing database session: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.finish_run(status="failed", error_message=str(exc_val))
        self.close()


def create_result_logger(
    results_dir,
    pipeline_id: str,
    dataset_name: str = "unknown",
    dataset_variant: str = "clean",
    config_file_path: Optional[str] = None,
    enable_database: bool = True,
    **kwargs
) -> ResultLogger:
    """
    Factory function to create the appropriate result logger.
    
    Returns DatabaseResultLogger if database is available and enabled,
    otherwise returns the basic ResultLogger.
    
    Args:
        results_dir: Directory for result files
        pipeline_id: Pipeline identifier
        dataset_name: Dataset name
        dataset_variant: Dataset variant
        config_file_path: Path to config file
        enable_database: Whether to enable database logging
        **kwargs: Additional arguments for DatabaseResultLogger
    
    Returns:
        ResultLogger instance
    """
    if enable_database and DB_AVAILABLE:
        # Check if database is configured
        db_url = kwargs.get('database_url') or os.getenv("LANDSEER_DB_HOST")
        if db_url:
            try:
                return DatabaseResultLogger(
                    results_dir=results_dir,
                    pipeline_id=pipeline_id,
                    dataset_name=dataset_name,
                    dataset_variant=dataset_variant,
                    config_file_path=config_file_path,
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"Failed to create DatabaseResultLogger: {e}")
    
    # Fallback to basic logger
    return ResultLogger(results_dir, pipeline_id)
