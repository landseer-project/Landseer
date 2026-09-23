"""In-memory scheduler state shared by API routes."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from fastapi import HTTPException

from src.common import get_logger
from src.pipeline.tasks import TaskStatus
from src.pipeline.pipeline import Pipeline
from src.backend.scheduler import Scheduler, PriorityScheduler

try:
    from src.backend.db_service import get_db_service, DatabaseService
    DB_SERVICE_AVAILABLE = True
except ImportError:
    DB_SERVICE_AVAILABLE = False
    DatabaseService = None  # type: ignore
    get_db_service = None  # type: ignore

logger = get_logger(__name__)

class SchedulerState:
    """
    Holds the global scheduler state for the API.
    
    This is initialized when the API starts and provides access to:
    - The scheduler instance
    - The pipeline being executed
    - Task execution metadata (timing, errors, etc.)
    - Registered workers
    - Database service for persistence
    """
    
    def __init__(self):
        self.scheduler: Optional[Scheduler] = None
        self.pipeline: Optional[Pipeline] = None
        self.started_at: Optional[datetime] = None
        self.task_metadata: Dict[str, Dict[str, Any]] = {}
        self.workers: Dict[str, Dict[str, Any]] = {}
        self._worker_counter: int = 0
        self._custom_tools: Dict[str, Dict[str, Any]] = {}
        self._db_service: Optional["DatabaseService"] = None
    
    @property
    def db_service(self) -> Optional["DatabaseService"]:
        """Get database service."""
        if self._db_service is None and DB_SERVICE_AVAILABLE:
            try:
                self._db_service = get_db_service()
            except RuntimeError:
                pass
        return self._db_service
    
    def initialize(self, pipeline: Pipeline, scheduler_type: str = "priority") -> None:
        """
        Initialize the scheduler with a pipeline.
        
        Args:
            pipeline: Pipeline instance to schedule
            scheduler_type: Type of scheduler to use ('priority' is default)
        """
        self.pipeline = pipeline
        
        if scheduler_type == "priority":
            t_sched_ctor_start = time.time()
            self.scheduler = PriorityScheduler(pipeline)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        self.started_at = datetime.now()
        logger.info(f"Scheduler initialized with pipeline: {pipeline.name}")
    
    def is_initialized(self) -> bool:
        """Check if scheduler is initialized."""
        return self.scheduler is not None
    
    def store_task_metadata(
        self,
        task_id: str,
        error_message: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
        result: Optional[Dict[str, Any]] = None,
        worker_id: Optional[str] = None,
        status: Optional[TaskStatus] = None
    ) -> None:
        """Store execution metadata for a task."""
        self.task_metadata[task_id] = {
            "error_message": error_message,
            "execution_time_ms": execution_time_ms,
            "result": result,
            "worker_id": worker_id,
            "updated_at": datetime.now().isoformat()
        }
        
        # Persist task status to database
        if self.db_service and status:
            self.db_service.sync_task_status(
                task_id=task_id,
                status=status,
                error_message=error_message,
                execution_time_ms=execution_time_ms,
                worker_id=worker_id
            )
    
    def register_worker(
        self,
        hostname: str,
        worker_id: Optional[str] = None,
        capabilities: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a new worker and return its ID."""
        if worker_id is None:
            self._worker_counter += 1
            worker_id = f"worker_{self._worker_counter}"
        
        now = datetime.now().isoformat()
        self.workers[worker_id] = {
            "worker_id": worker_id,
            "hostname": hostname,
            "status": "idle",
            "registered_at": now,
            "last_heartbeat": now,
            "current_task_id": None,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "capabilities": capabilities or {}
        }
        
        # Persist to database
        if self.db_service:
            self.db_service.register_worker(worker_id, hostname, capabilities)
        
        logger.info(f"Worker registered: {worker_id} ({hostname})")
        return worker_id
    
    def update_worker_heartbeat(self, worker_id: str, status: Optional[str] = None) -> bool:
        """Update worker heartbeat timestamp."""
        if worker_id not in self.workers:
            return False
        self.workers[worker_id]["last_heartbeat"] = datetime.now().isoformat()
        if status:
            self.workers[worker_id]["status"] = status
            if status == "idle" and self.workers[worker_id].get("current_task_id"):
                stale_task_id = self.workers[worker_id]["current_task_id"]
                stale_task = (
                    self.scheduler._find_task_by_id(stale_task_id)
                    if self.scheduler is not None
                    else None
                )
                if stale_task and stale_task.status == TaskStatus.RUNNING:
                    # Worker claims to be idle while still owning a running task.
                    # This most commonly happens after claim/report timeouts and can deadlock progress.
                    stale_task.status = TaskStatus.PENDING
                    self.workers[worker_id]["current_task_id"] = None
                    logger.warning(
                        f"Reclaimed stale running task {stale_task_id} from idle heartbeat worker {worker_id}"
                    )
                elif stale_task is None or stale_task.status != TaskStatus.RUNNING:
                    # Orphan pointer: task finished or unknown to scheduler — drop so UI/API stay consistent.
                    self.workers[worker_id]["current_task_id"] = None
                    logger.info(
                        f"Cleared orphan current_task_id {stale_task_id} on idle heartbeat "
                        f"for worker {worker_id}"
                    )
        
        # Persist to database
        if self.db_service:
            self.db_service.update_worker_heartbeat(worker_id, status)
        return True
    
    def assign_task_to_worker(self, worker_id: str, task_id: str) -> None:
        """Mark a worker as busy with a task."""
        if worker_id in self.workers:
            self.workers[worker_id]["status"] = "busy"
            self.workers[worker_id]["current_task_id"] = task_id
            self.workers[worker_id]["last_heartbeat"] = datetime.now().isoformat()
            
            # Persist to database
            if self.db_service:
                self.db_service.assign_task_to_worker(worker_id, task_id)

    def find_worker_assigned_to_task(self, task_id: str) -> Optional[str]:
        """Return the worker currently assigned to task_id, if any."""
        for worker_id, worker in self.workers.items():
            if worker.get("current_task_id") == task_id:
                return worker_id
        return None
    
    def complete_worker_task(self, worker_id: str, success: bool, task_id: Optional[str] = None, execution_time_ms: int = 0) -> None:
        """Mark worker task as complete."""
        if worker_id in self.workers:
            current_task_id = task_id or self.workers[worker_id].get("current_task_id")
            self.workers[worker_id]["status"] = "idle"
            self.workers[worker_id]["current_task_id"] = None
            self.workers[worker_id]["last_heartbeat"] = datetime.now().isoformat()
            if success:
                self.workers[worker_id]["tasks_completed"] += 1
            else:
                self.workers[worker_id]["tasks_failed"] += 1
            
            # Persist to database
            if self.db_service and current_task_id:
                self.db_service.complete_worker_task(
                    worker_id=worker_id,
                    task_id=current_task_id,
                    success=success,
                    execution_time_ms=execution_time_ms
                )
    
    def reclaim_stale_workers(self, stale_timeout_seconds: float = 90.0) -> dict:
        """
        Detect workers whose heartbeat has gone silent, reset their RUNNING tasks
        to PENDING, and mark those workers offline.

        Called automatically by the background reclaim loop every 60 s.
        Also called manually via POST /scheduler/reclaim-stale.

        stale_timeout_seconds: seconds since last heartbeat before a worker is
        considered dead. Default 90 s = 3× the 30 s worker heartbeat interval.
        """
        now = datetime.now()
        reclaimed_tasks: List[str] = []
        stale_worker_ids: List[str] = []

        for worker_id, worker in self.workers.items():
            last_hb_str = worker.get("last_heartbeat")
            if not last_hb_str:
                continue
            elapsed = (now - datetime.fromisoformat(last_hb_str)).total_seconds()
            if elapsed < stale_timeout_seconds:
                continue  # worker is alive

            stale_worker_ids.append(worker_id)
            task_id = worker.get("current_task_id")
            if task_id and self.scheduler:
                task = self.scheduler._find_task_by_id(task_id)
                if task and task.status == TaskStatus.RUNNING:
                    task.status = TaskStatus.PENDING
                    reclaimed_tasks.append(task_id)
                    logger.info(f"Reclaimed task {task_id} from stale worker {worker_id} "
                                f"(no heartbeat for {elapsed:.0f}s)")

            worker["status"] = "offline"
            worker["current_task_id"] = None

        if stale_worker_ids:
            logger.warning(f"Stale workers: {stale_worker_ids} | "
                           f"Reclaimed tasks: {reclaimed_tasks}")

        return {"stale_workers": stale_worker_ids, "reclaimed_tasks": reclaimed_tasks}

    def add_tool(
        self,
        name: str,
        image: str,
        command: str,
        runtime: Optional[str] = None,
        is_baseline: bool = False,
        defense_stage: Optional[str] = None,
        key: Optional[str] = None,
    ) -> None:
        """Add a custom tool to the registry."""
        registry_key = key or name
        self._custom_tools[registry_key] = {
            "name": name,
            "container": {
                "image": image,
                "command": command,
                "runtime": runtime
            },
            "is_baseline": is_baseline,
            "defense_stage": defense_stage,
        }
        logger.info(f"Tool added: key={registry_key}, name={name}")
    
    def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get all tools (from registry + custom)."""
        from src.pipeline.tools import get_all_tools
        tools = get_all_tools()
        # Convert to dict format and merge with custom tools
        result = {}
        for name, tool in tools.items():
            result[name] = {
                "name": tool.name,
                "container": {
                    "image": tool.container.image,
                    "command": tool.container.command,
                    "runtime": tool.container.runtime
                },
                "is_baseline": tool.is_baseline,
                "defense_stage": tool.defense_stage,
            }
        result.update(self._custom_tools)
        return result


# Global scheduler state
_scheduler_state = SchedulerState()


def get_scheduler_state() -> SchedulerState:
    """Dependency injection for scheduler state."""
    return _scheduler_state


def get_scheduler() -> Scheduler:
    """Get the active scheduler, raising an error if not initialized."""
    if not _scheduler_state.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="Scheduler not initialized. Please initialize with a pipeline first."
        )
    return _scheduler_state.scheduler
