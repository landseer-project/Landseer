"""
Database service for the Landseer backend.

Provides integration between the in-memory scheduler and the persistent database.
This allows the system to:
- Persist task state for crash recovery
- Track historical execution data
- Support multiple backend instances
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..common import get_logger
from ..pipeline.pipeline import Pipeline
from ..pipeline.workflow import Workflow
from ..pipeline.tasks import Task, TaskStatus as PipelineTaskStatus
from .scheduler import Scheduler

# Import database modules
try:
    from ..db import (
        Database,
        DatabaseConfig,
        init_database,
        get_database,
        get_session,
        session_scope,
        TaskRepository,
        WorkerRepository,
        WorkflowRepository,
        ExperimentRunRepository,
        TaskModel,
        WorkerModel,
        WorkflowModel,
        ExperimentRunModel,
        DBTaskStatus,
        WorkerStatus,
    )
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

logger = get_logger(__name__)


class DatabaseService:
    """
    Service for database operations in the backend.
    
    Provides methods to:
    - Sync pipeline state to database
    - Load pipeline state from database
    - Track worker registration and heartbeats
    - Store task execution results
    """
    
    def __init__(self, config: Optional["DatabaseConfig"] = None, enabled: bool = True):
        """
        Initialize database service.
        
        Args:
            config: Database configuration
            enabled: Whether to enable database persistence
        """
        self.enabled = enabled and DB_AVAILABLE
        self.config = config
        self._db: Optional[Database] = None
        
        if not DB_AVAILABLE:
            logger.warning("Database module not available. Persistence disabled.")
        elif not enabled:
            logger.info("Database persistence disabled by configuration.")
    
    def initialize(self) -> bool:
        """
        Initialize the database connection.
        
        Returns:
            True if initialization successful
        """
        if not self.enabled:
            return False
        
        try:
            self._db = init_database(self.config, create_tables=True)
            logger.info("Database service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.enabled = False
            return False
    
    def is_available(self) -> bool:
        """Check if database is available."""
        return self.enabled and self._db is not None
    
    def get_session(self):
        """
        Get a database session.
        
        Returns:
            Database session (use with session_scope() context manager for proper cleanup)
        """
        if not self.is_available():
            raise RuntimeError("Database service is not available")
        return get_session()
    
    # =========================================================================
    # Pipeline Sync Operations
    # =========================================================================
    
    def sync_pipeline_to_db(self, pipeline: Pipeline, run_id: Optional[str] = None) -> Optional[str]:
        """
        Sync an in-memory pipeline's workflows/tasks onto an experiment run.

        ``run_id`` (or ``pipeline.id``) must already exist as an ``ExperimentRun``.
        """
        if not self.is_available():
            return None

        try:
            from ..db.models import (
                TaskType as DBTaskType,
                ToolModel,
                ToolType,
                ToolStage,
                ToolDefenseType,
                WorkflowStatus,
            )

            with session_scope() as session:
                run_repo = ExperimentRunRepository(session)
                workflow_repo = WorkflowRepository(session)
                task_repo = TaskRepository(session)

                effective_run_id = (
                    run_id
                    or getattr(pipeline, "run_id", None)
                    or pipeline.id
                )
                db_run = run_repo.get_by_id(effective_run_id)
                if not db_run:
                    logger.error(
                        "Cannot sync pipeline %s: experiment run %s not found",
                        pipeline.id,
                        effective_run_id,
                    )
                    return None

                def _ensure_tool(task: Task) -> str:
                    tool = task.tool
                    tool_id = getattr(tool, "id", None) or tool.name
                    existing = session.get(ToolModel, tool_id)
                    if existing:
                        return tool_id

                    stage_raw = (
                        getattr(tool, "defense_stage", None)
                        or getattr(tool, "stage", None)
                        or "other"
                    )
                    if hasattr(stage_raw, "value"):
                        stage_raw = stage_raw.value
                    try:
                        stage = ToolStage(str(stage_raw))
                    except ValueError:
                        stage = ToolStage.OTHER

                    if getattr(tool, "is_baseline", False):
                        tool_type = ToolType.BASELINE
                    elif str(getattr(task.task_type, "value", task.task_type)) == "evaluation":
                        tool_type = ToolType.EVAL
                    else:
                        tool_type = ToolType.DEFENSE

                    session.add(
                        ToolModel(
                            id=tool_id,
                            tool_name=tool.name,
                            tool_type=tool_type,
                            tool_defense_type=ToolDefenseType.NONE,
                            tool_stage=stage,
                            tool_image=tool.container.image,
                            tool_parameters=getattr(tool, "parameters", None) or {},
                        )
                    )
                    session.flush()
                    return tool_id

                def _db_task_type(task: Task) -> DBTaskType:
                    token = str(getattr(task.task_type, "value", task.task_type))
                    if token == "evaluation":
                        return DBTaskType.EVALUATION
                    return DBTaskType.TOOL

                # Tasks first
                task_map: Dict[str, Task] = {}
                for workflow in pipeline.workflows:
                    for task in workflow.tasks:
                        task_map[task.id] = task

                for task in task_map.values():
                    tool_id = _ensure_tool(task)
                    db_task = task_repo.get_by_id(task.id)
                    if not db_task:
                        task_repo.create(
                            {
                                "id": task.id,
                                "task_type": _db_task_type(task),
                                "tool_id": tool_id,
                                "config": task.config or {},
                                "priority": task.priority,
                                "status": self._convert_status(task.status),
                                "task_hash": task.get_hash(),
                                "counter": task.counter,
                                "experiment_run_id": effective_run_id,
                            }
                        )
                    else:
                        db_task.experiment_run_id = effective_run_id
                        db_task.status = self._convert_status(task.status)

                # Workflows + ordered membership
                for workflow in pipeline.workflows:
                    db_workflow = workflow_repo.get_by_id(workflow.id)
                    if not db_workflow:
                        db_workflow = workflow_repo.create(
                            {
                                "id": workflow.id,
                                "experiment_run_id": effective_run_id,
                                "status": WorkflowStatus.PENDING,
                            }
                        )
                    else:
                        db_workflow.experiment_run_id = effective_run_id

                    ordered = []
                    for t in workflow.tasks:
                        db_t = task_repo.get_by_id(t.id)
                        if db_t:
                            ordered.append(db_t)
                    db_workflow.set_ordered_tasks(session, ordered)

                # Dependencies
                for task in task_map.values():
                    db_task = task_repo.get_by_id(task.id)
                    if not db_task:
                        continue
                    for dep in task.dependencies:
                        db_dep = task_repo.get_by_id(dep.id)
                        if db_dep and db_dep not in db_task.dependencies:
                            db_task.dependencies.append(db_dep)

                logger.info(
                    "Synced pipeline %s to experiment run %s",
                    pipeline.id,
                    effective_run_id,
                )
                return effective_run_id

        except Exception as e:
            logger.error(f"Failed to sync pipeline to database: {e}", exc_info=True)
            return None
    
    def sync_task_status(
        self,
        task_id: str,
        status: PipelineTaskStatus,
        error_message: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
        worker_id: Optional[str] = None
    ) -> bool:
        """
        Sync task status to database.
        
        Args:
            task_id: Task ID
            status: New task status
            error_message: Error message if failed
            execution_time_ms: Execution time in milliseconds
            worker_id: Worker that executed the task
            
        Returns:
            True if successful
        """
        if not self.is_available():
            return False
        
        try:
            with session_scope() as session:
                task_repo = TaskRepository(session)
                task = task_repo.update_status(
                    task_id=task_id,
                    status=self._convert_status(status),
                    error_message=error_message,
                    execution_time_ms=execution_time_ms
                )
                
                if task and worker_id:
                    task.assigned_worker_id = worker_id
                
                return task is not None
                
        except Exception as e:
            logger.error(f"Failed to sync task status: {e}")
            return False
    
    def save_evaluation_result(
        self,
        workflow_id: str,
        evaluation_task_id: str,
        result_data: Dict[str, Any],
        pipeline_id: Optional[str] = None,
        run_id: Optional[str] = None,
        experiment_run_id: Optional[str] = None,
        # Legacy kwargs ignored (tool identity comes from evaluation_task → tools)
        evaluator_name: Optional[str] = None,
        evaluator_image: Optional[str] = None,
        evaluator_tool_id: Optional[str] = None,
    ) -> bool:
        """
        Save or update an evaluation result for a workflow.

        Tool name/image and experiment run are not stored on this row —
        resolve via ``evaluation_task → tools`` and ``workflow → experiment_run``.
        """
        if not self.is_available():
            logger.warning(
                f"Database not available, cannot save evaluation result for "
                f"{workflow_id}/{evaluation_task_id}"
            )
            return False
        try:
            from ..db.models import EvaluationResultModel

            result_id = f"{workflow_id}_{evaluation_task_id}"
            with session_scope() as session:
                er = EvaluationResultModel.from_evaluation_result(
                    result_id=result_id,
                    workflow_id=workflow_id,
                    evaluation_task_id=evaluation_task_id,
                    result_data=result_data,
                )
                session.merge(er)

            metrics_count = len(result_data.get("metrics", {}))
            logger.info(
                f"Saved evaluation result {result_id} "
                f"(metrics={metrics_count}, "
                f"skipped={result_data.get('skipped', False)})"
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to save evaluation result for "
                f"{workflow_id}/{evaluation_task_id}: {e}",
                exc_info=True,
            )
            return False
    
    # =========================================================================
    # Worker Operations
    # =========================================================================
    
    def register_worker(
        self,
        worker_id: str,
        hostname: str,
        capabilities: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a worker in the database.
        
        Args:
            worker_id: Unique worker ID
            hostname: Worker hostname
            capabilities: Worker capabilities
            
        Returns:
            True if successful
        """
        if not self.is_available():
            return False
        
        try:
            with session_scope() as session:
                worker_repo = WorkerRepository(session)
                existing = worker_repo.get_by_id(worker_id)
                
                if existing:
                    # Update existing worker
                    existing.hostname = hostname
                    existing.capabilities = capabilities or {}
                    existing.status = WorkerStatus.IDLE
                    existing.last_heartbeat = datetime.utcnow()
                else:
                    # Create new worker
                    worker_repo.create({
                        "id": worker_id,
                        "hostname": hostname,
                        "capabilities": capabilities or {},
                        "status": WorkerStatus.IDLE
                    })
                
                logger.debug(f"Registered worker: {worker_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register worker: {e}")
            return False
    
    def update_worker_heartbeat(
        self,
        worker_id: str,
        status: Optional[str] = None
    ) -> bool:
        """
        Update worker heartbeat in database.
        
        Args:
            worker_id: Worker ID
            status: Optional status update
            
        Returns:
            True if successful
        """
        if not self.is_available():
            return False
        
        try:
            with session_scope() as session:
                worker_repo = WorkerRepository(session)
                worker_status = None
                if status:
                    worker_status = WorkerStatus(status)
                
                worker = worker_repo.update_heartbeat(worker_id, worker_status)
                return worker is not None
                
        except Exception as e:
            logger.error(f"Failed to update worker heartbeat: {e}")
            return False
    
    def assign_task_to_worker(
        self,
        worker_id: str,
        task_id: str
    ) -> bool:
        """
        Record task assignment to worker.
        
        Args:
            worker_id: Worker ID
            task_id: Task ID
            
        Returns:
            True if successful
        """
        if not self.is_available():
            return False
        
        try:
            with session_scope() as session:
                worker_repo = WorkerRepository(session)
                task_repo = TaskRepository(session)
                
                # Update worker
                worker_repo.assign_task(worker_id, task_id)
                
                # Update task
                task_repo.assign_to_worker(task_id, worker_id)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to assign task to worker: {e}")
            return False
    
    def complete_worker_task(
        self,
        worker_id: str,
        task_id: str,
        success: bool,
        execution_time_ms: int = 0,
        workflow_id: Optional[str] = None
    ) -> bool:
        """
        Record task completion by worker.
        
        Args:
            worker_id: Worker ID
            task_id: Task ID
            success: Whether task succeeded
            execution_time_ms: Execution time
            workflow_id: Workflow ID for locality tracking
            
        Returns:
            True if successful
        """
        if not self.is_available():
            return False
        
        try:
            with session_scope() as session:
                worker_repo = WorkerRepository(session)
                worker_repo.complete_task(
                    worker_id=worker_id,
                    success=success,
                    execution_time_ms=execution_time_ms,
                    workflow_id=workflow_id
                )
                return True
                
        except Exception as e:
            logger.error(f"Failed to complete worker task: {e}")
            return False
    
    def get_all_workers(self) -> List[Dict[str, Any]]:
        """Get all workers from database."""
        if not self.is_available():
            return []
        
        try:
            with session_scope() as session:
                worker_repo = WorkerRepository(session)
                workers = worker_repo.get_all()
                return [w.to_dict() for w in workers]
        except Exception as e:
            logger.error(f"Failed to get workers: {e}")
            return []
    
    # =========================================================================
    # Query Operations
    # =========================================================================
    
    def get_task_progress(self, pipeline_id: Optional[str] = None) -> Dict[str, int]:
        """Get task progress from database."""
        if not self.is_available():
            return {"pending": 0, "running": 0, "completed": 0, "failed": 0, "total": 0}
        
        try:
            with session_scope() as session:
                task_repo = TaskRepository(session)
                return task_repo.get_progress(pipeline_id)
        except Exception as e:
            logger.error(f"Failed to get task progress: {e}")
            return {"pending": 0, "running": 0, "completed": 0, "failed": 0, "total": 0}
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics from database."""
        if not self.is_available():
            return {"total": 0, "idle": 0, "busy": 0, "offline": 0, "active": 0}
        
        try:
            with session_scope() as session:
                worker_repo = WorkerRepository(session)
                return worker_repo.get_stats()
        except Exception as e:
            logger.error(f"Failed to get worker stats: {e}")
            return {"total": 0, "idle": 0, "busy": 0, "offline": 0, "active": 0}
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _convert_status(self, status: PipelineTaskStatus):
        """Convert pipeline task status to database task status."""
        if not DB_AVAILABLE or DBTaskStatus is None:
            return None
        
        mapping = {
            PipelineTaskStatus.PENDING: DBTaskStatus.PENDING,
            PipelineTaskStatus.RUNNING: DBTaskStatus.RUNNING,
            PipelineTaskStatus.COMPLETED: DBTaskStatus.COMPLETED,
            PipelineTaskStatus.FAILED: DBTaskStatus.FAILED,
            PipelineTaskStatus.CANCELLED: DBTaskStatus.CANCELLED,
        }
        return mapping.get(status, DBTaskStatus.PENDING)


# Global database service
_db_service: Optional[DatabaseService] = None


def get_db_service() -> DatabaseService:
    """Get the global database service."""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service


def init_db_service(
    config: Optional[DatabaseConfig] = None,
    enabled: bool = True
) -> DatabaseService:
    """Initialize the global database service."""
    global _db_service
    _db_service = DatabaseService(config, enabled)
    _db_service.initialize()
    return _db_service
