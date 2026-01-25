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
        PipelineRepository,
        TaskModel,
        WorkerModel,
        WorkflowModel,
        PipelineModel,
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
    
    # =========================================================================
    # Pipeline Sync Operations
    # =========================================================================
    
    def sync_pipeline_to_db(self, pipeline: Pipeline) -> Optional[str]:
        """
        Sync a pipeline and all its workflows/tasks to the database.
        
        Args:
            pipeline: Pipeline to sync
            
        Returns:
            Pipeline ID if successful, None otherwise
        """
        if not self.is_available():
            return None
        
        try:
            with session_scope() as session:
                pipeline_repo = PipelineRepository(session)
                workflow_repo = WorkflowRepository(session)
                task_repo = TaskRepository(session)
                
                # Create or update pipeline
                db_pipeline = pipeline_repo.get_by_id(pipeline.id)
                if not db_pipeline:
                    db_pipeline = pipeline_repo.create({
                        "id": pipeline.id,
                        "name": pipeline.name,
                        "config": pipeline.config,
                        "dataset_config": pipeline.dataset,
                        "model_config": pipeline.model,
                        "status": "pending"
                    })
                
                # Collect all unique tasks
                task_map: Dict[str, Task] = {}
                for workflow in pipeline.workflows:
                    for task in workflow.tasks:
                        task_map[task.id] = task
                
                # Create tasks
                for task in task_map.values():
                    db_task = task_repo.get_by_id(task.id)
                    if not db_task:
                        task_repo.create({
                            "id": task.id,
                            "tool_name": task.tool.name,
                            "tool_image": task.tool.container.image,
                            "tool_command": task.tool.container.command,
                            "tool_is_baseline": task.tool.is_baseline,
                            "config": task.config,
                            "priority": task.priority,
                            "status": self._convert_status(task.status),
                            "task_type": task.task_type.value,
                            "task_hash": task.get_hash(),
                            "counter": task.counter,
                            "pipeline_id": pipeline.id
                        })
                
                # Create workflows and link tasks
                for workflow in pipeline.workflows:
                    db_workflow = workflow_repo.get_by_id(workflow.id)
                    if not db_workflow:
                        db_workflow = workflow_repo.create({
                            "id": workflow.id,
                            "name": workflow.name,
                            "pipeline_id": pipeline.id,
                            "status": "pending"
                        })
                    
                    # Link tasks to workflow
                    for task in workflow.tasks:
                        db_task = task_repo.get_by_id(task.id)
                        if db_task and db_workflow not in db_task.workflows:
                            db_task.workflows.append(db_workflow)
                
                # Set task dependencies
                for task in task_map.values():
                    db_task = task_repo.get_by_id(task.id)
                    if db_task:
                        for dep in task.dependencies:
                            db_dep = task_repo.get_by_id(dep.id)
                            if db_dep and db_dep not in db_task.dependencies:
                                db_task.dependencies.append(db_dep)
                
                logger.info(f"Synced pipeline {pipeline.id} to database")
                return pipeline.id
                
        except Exception as e:
            logger.error(f"Failed to sync pipeline to database: {e}")
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
