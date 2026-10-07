"""
Repository classes for database operations.

Provides CRUD operations and queries for Landseer entities.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, or_, func
from sqlalchemy.orm import Session

from ..common import get_logger
from .models import (
    TaskModel,
    WorkerModel,
    WorkflowModel,
    ExperimentModel,
    ExperimentRunModel,
    ExperimentRunStatus,
    ArtifactModel,
    TaskStatus,
    WorkerStatus,
    TaskType,
    ToolModel,
    ToolType,
    ToolStage,
    ToolDefenseType,
)

logger = get_logger(__name__)


class TaskRepository:
    """Repository for Task operations."""
    
    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session
    
    def create(self, task_data: Dict[str, Any]) -> TaskModel:
        """Create a new task."""
        task = TaskModel(**task_data)
        self.session.add(task)
        self.session.flush()
        logger.debug(f"Created task: {task.id}")
        return task
    
    def get_by_id(self, task_id: str) -> Optional[TaskModel]:
        """Get task by ID."""
        return self.session.query(TaskModel).filter(TaskModel.id == task_id).first()
    
    def get_all(self, pipeline_id: Optional[str] = None) -> List[TaskModel]:
        """Get all tasks, optionally filtered by experiment run (legacy: pipeline_id)."""
        query = self.session.query(TaskModel)
        if pipeline_id:
            query = query.filter(TaskModel.experiment_run_id == pipeline_id)
        return query.all()
    
    def get_by_status(
        self,
        status: TaskStatus,
        pipeline_id: Optional[str] = None
    ) -> List[TaskModel]:
        """Get tasks by status."""
        query = self.session.query(TaskModel).filter(TaskModel.status == status)
        if pipeline_id:
            query = query.filter(TaskModel.experiment_run_id == pipeline_id)
        return query.order_by(TaskModel.priority.asc()).all()
    
    def get_ready_tasks(self, pipeline_id: Optional[str] = None) -> List[TaskModel]:
        """
        Get tasks that are ready to execute.
        
        A task is ready if:
        - It's in PENDING status
        - All its dependencies are COMPLETED
        """
        query = self.session.query(TaskModel).filter(
            TaskModel.status == TaskStatus.PENDING
        )
        if pipeline_id:
            query = query.filter(TaskModel.experiment_run_id == pipeline_id)
        
        tasks = query.order_by(TaskModel.priority.asc()).all()
        
        # Filter to only tasks with all dependencies completed
        ready_tasks = []
        for task in tasks:
            deps_completed = all(
                dep.status == TaskStatus.COMPLETED
                for dep in task.dependencies
            )
            if deps_completed:
                ready_tasks.append(task)
        
        return ready_tasks
    
    def get_by_worker(self, worker_id: str) -> List[TaskModel]:
        """Get tasks assigned to a specific worker."""
        return self.session.query(TaskModel).filter(
            TaskModel.assigned_worker_id == worker_id
        ).all()
    
    def get_by_cache_key(self, cache_key: str) -> Optional[TaskModel]:
        """Get task by cache key (task_hash / artifact id)."""
        return self.session.query(TaskModel).filter(
            or_(
                TaskModel.task_hash == cache_key,
                TaskModel.artifact_id == cache_key,
            )
        ).first()
    
    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        error_message: Optional[str] = None,
        execution_time_ms: Optional[int] = None
    ) -> Optional[TaskModel]:
        """Update task status."""
        task = self.get_by_id(task_id)
        if not task:
            return None
        
        task.status = status
        task.error_message = error_message
        task.execution_time_ms = execution_time_ms
        task.updated_at = datetime.utcnow()
        
        if status == TaskStatus.RUNNING:
            task.started_at = datetime.utcnow()
            task.attempts += 1
        elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            task.completed_at = datetime.utcnow()
        
        self.session.flush()
        logger.debug(f"Updated task {task_id} status to {status.value}")
        return task
    
    def assign_to_worker(
        self,
        task_id: str,
        worker_id: str
    ) -> Optional[TaskModel]:
        """Assign task to a worker."""
        task = self.get_by_id(task_id)
        if not task:
            return None
        
        task.assigned_worker_id = worker_id
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        task.attempts += 1
        task.updated_at = datetime.utcnow()
        
        self.session.flush()
        logger.debug(f"Assigned task {task_id} to worker {worker_id}")
        return task
    
    def set_cache_key(
        self,
        task_id: str,
        cache_key: str,
        artifact_path: Optional[str] = None
    ) -> Optional[TaskModel]:
        """Set cache key (task_hash) and optional artifact id for a task."""
        task = self.get_by_id(task_id)
        if not task:
            return None
        
        task.task_hash = cache_key
        if artifact_path is not None:
            # Callers historically passed a path; prefer treating as artifact id/key.
            task.artifact_id = artifact_path
        else:
            task.artifact_id = cache_key
        task.updated_at = datetime.utcnow()
        
        self.session.flush()
        return task
    
    def get_progress(self, pipeline_id: Optional[str] = None) -> Dict[str, int]:
        """Get task progress statistics."""
        query = self.session.query(
            TaskModel.status,
            func.count(TaskModel.id)
        ).group_by(TaskModel.status)
        
        if pipeline_id:
            query = query.filter(TaskModel.experiment_run_id == pipeline_id)
        
        results = query.all()
        
        progress = {
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "total": 0
        }
        
        for status, count in results:
            if status:
                progress[status.value] = count
                progress["total"] += count
        
        return progress
    
    def delete(self, task_id: str) -> bool:
        """Delete a task."""
        task = self.get_by_id(task_id)
        if not task:
            return False
        
        self.session.delete(task)
        self.session.flush()
        logger.debug(f"Deleted task: {task_id}")
        return True
    
    def bulk_create(self, tasks_data: List[Dict[str, Any]]) -> List[TaskModel]:
        """Create multiple tasks at once."""
        tasks = [TaskModel(**data) for data in tasks_data]
        self.session.add_all(tasks)
        self.session.flush()
        logger.debug(f"Created {len(tasks)} tasks")
        return tasks


class WorkerRepository:
    """Repository for Worker operations."""
    
    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session
    
    def create(self, worker_data: Dict[str, Any]) -> WorkerModel:
        """Register a new worker."""
        worker = WorkerModel(**worker_data)
        self.session.add(worker)
        self.session.flush()
        logger.debug(f"Registered worker: {worker.id}")
        return worker
    
    def get_by_id(self, worker_id: str) -> Optional[WorkerModel]:
        """Get worker by ID."""
        return self.session.query(WorkerModel).filter(
            WorkerModel.id == worker_id
        ).first()
    
    def get_all(self) -> List[WorkerModel]:
        """Get all workers."""
        return self.session.query(WorkerModel).all()
    
    def get_active(self, stale_timeout: int = 120) -> List[WorkerModel]:
        """Get workers that have sent heartbeat within timeout."""
        cutoff = datetime.utcnow()
        return self.session.query(WorkerModel).filter(
            WorkerModel.status != WorkerStatus.OFFLINE
        ).all()
    
    def get_idle(self) -> List[WorkerModel]:
        """Get idle workers."""
        return self.session.query(WorkerModel).filter(
            WorkerModel.status == WorkerStatus.IDLE
        ).all()
    
    def update_heartbeat(
        self,
        worker_id: str,
        status: Optional[WorkerStatus] = None
    ) -> Optional[WorkerModel]:
        """Update worker heartbeat timestamp."""
        worker = self.get_by_id(worker_id)
        if not worker:
            return None
        
        worker.last_heartbeat = datetime.utcnow()
        if status:
            worker.status = status
        
        self.session.flush()
        return worker
    
    def assign_task(
        self,
        worker_id: str,
        task_id: str
    ) -> Optional[WorkerModel]:
        """Assign a task to worker."""
        worker = self.get_by_id(worker_id)
        if not worker:
            return None
        
        worker.current_task_id = task_id
        worker.status = WorkerStatus.BUSY
        worker.last_heartbeat = datetime.utcnow()
        
        self.session.flush()
        logger.debug(f"Worker {worker_id} assigned task {task_id}")
        return worker
    
    def complete_task(
        self,
        worker_id: str,
        success: bool,
        execution_time_ms: int = 0,
        workflow_id: Optional[str] = None
    ) -> Optional[WorkerModel]:
        """Mark worker's current task as complete."""
        worker = self.get_by_id(worker_id)
        if not worker:
            return None
        
        worker.current_task_id = None
        worker.status = WorkerStatus.IDLE
        worker.last_heartbeat = datetime.utcnow()
        worker.total_execution_time_ms += execution_time_ms
        
        if success:
            worker.tasks_completed += 1
        else:
            worker.tasks_failed += 1
        
        # Track workflow for locality
        if workflow_id:
            from .models import WorkflowModel

            wf = self.session.get(WorkflowModel, workflow_id)
            if wf is not None and wf not in worker.handled_workflows:
                worker.handled_workflows.append(wf)
        
        self.session.flush()
        return worker
    
    def mark_offline(self, worker_id: str) -> Optional[WorkerModel]:
        """Mark worker as offline."""
        worker = self.get_by_id(worker_id)
        if not worker:
            return None
        
        worker.status = WorkerStatus.OFFLINE
        worker.current_task_id = None
        
        self.session.flush()
        logger.info(f"Worker {worker_id} marked offline")
        return worker
    
    def mark_stale_workers_offline(self, timeout_seconds: int = 120) -> int:
        """Mark workers with stale heartbeat as offline."""
        cutoff = datetime.utcnow()
        
        stale_workers = self.session.query(WorkerModel).filter(
            and_(
                WorkerModel.status != WorkerStatus.OFFLINE,
                WorkerModel.last_heartbeat < cutoff
            )
        ).all()
        
        count = 0
        for worker in stale_workers:
            if worker.is_stale(timeout_seconds):
                worker.status = WorkerStatus.OFFLINE
                worker.current_task_id = None
                count += 1
        
        if count > 0:
            self.session.flush()
            logger.info(f"Marked {count} stale workers as offline")
        
        return count
    
    def delete(self, worker_id: str) -> bool:
        """Delete a worker."""
        worker = self.get_by_id(worker_id)
        if not worker:
            return False
        
        self.session.delete(worker)
        self.session.flush()
        logger.debug(f"Deleted worker: {worker_id}")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        total = self.session.query(func.count(WorkerModel.id)).scalar()
        idle = self.session.query(func.count(WorkerModel.id)).filter(
            WorkerModel.status == WorkerStatus.IDLE
        ).scalar()
        busy = self.session.query(func.count(WorkerModel.id)).filter(
            WorkerModel.status == WorkerStatus.BUSY
        ).scalar()
        offline = self.session.query(func.count(WorkerModel.id)).filter(
            WorkerModel.status == WorkerStatus.OFFLINE
        ).scalar()
        
        return {
            "total": total,
            "idle": idle,
            "busy": busy,
            "offline": offline,
            "active": idle + busy
        }


class WorkflowRepository:
    """Repository for Workflow operations."""
    
    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session
    
    def create(self, workflow_data: Dict[str, Any]) -> WorkflowModel:
        """Create a new workflow."""
        workflow = WorkflowModel(**workflow_data)
        self.session.add(workflow)
        self.session.flush()
        logger.debug(f"Created workflow: {workflow.id}")
        return workflow
    
    def get_by_id(self, workflow_id: str) -> Optional[WorkflowModel]:
        """Get workflow by ID."""
        return self.session.query(WorkflowModel).filter(
            WorkflowModel.id == workflow_id
        ).first()
    
    def get_by_name(
        self,
        name: str,
        pipeline_id: Optional[str] = None
    ) -> Optional[WorkflowModel]:
        """Get workflow by id-as-name (name column removed); filters by experiment run."""
        query = self.session.query(WorkflowModel).filter(WorkflowModel.id == name)
        if pipeline_id:
            query = query.filter(WorkflowModel.experiment_run_id == pipeline_id)
        return query.first()
    
    def get_all(self, pipeline_id: Optional[str] = None) -> List[WorkflowModel]:
        """Get all workflows, optionally filtered by experiment run (legacy: pipeline_id)."""
        query = self.session.query(WorkflowModel)
        if pipeline_id:
            query = query.filter(WorkflowModel.experiment_run_id == pipeline_id)
        return query.all()
    
    def update_status(self, workflow_id: str) -> Optional[WorkflowModel]:
        """Update workflow status based on task completion."""
        from .models import WorkflowStatus

        workflow = self.get_by_id(workflow_id)
        if not workflow:
            return None
        
        tasks = list(workflow.tasks or [])
        
        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
        running = sum(1 for t in tasks if t.status == TaskStatus.RUNNING)
        
        total = len(tasks)
        if failed > 0:
            workflow.status = WorkflowStatus.FAILED
        elif total > 0 and completed == total:
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.utcnow()
        elif running > 0 or completed > 0:
            workflow.status = WorkflowStatus.RUNNING
            if not workflow.started_at:
                workflow.started_at = datetime.utcnow()
        else:
            workflow.status = WorkflowStatus.PENDING
        
        self.session.flush()
        return workflow
    
    def delete(self, workflow_id: str) -> bool:
        """Delete a workflow."""
        workflow = self.get_by_id(workflow_id)
        if not workflow:
            return False
        
        self.session.delete(workflow)
        self.session.flush()
        logger.debug(f"Deleted workflow: {workflow_id}")
        return True
    
    def bulk_create(
        self,
        workflows_data: List[Dict[str, Any]]
    ) -> List[WorkflowModel]:
        """Create multiple workflows at once."""
        workflows = [WorkflowModel(**data) for data in workflows_data]
        self.session.add_all(workflows)
        self.session.flush()
        logger.debug(f"Created {len(workflows)} workflows")
        return workflows


class ExperimentRepository:
    """Repository for Experiment operations."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, experiment_data: Dict[str, Any]) -> ExperimentModel:
        data = dict(experiment_data)
        if "name" in data and "experiment_name" not in data:
            data["experiment_name"] = data.pop("name")
        experiment = ExperimentModel(**data)
        self.session.add(experiment)
        self.session.flush()
        logger.debug(f"Created experiment: {experiment.id}")
        return experiment

    def get_by_id(self, experiment_id: str) -> Optional[ExperimentModel]:
        return self.session.query(ExperimentModel).filter(
            ExperimentModel.id == experiment_id
        ).first()

    def get_by_path(self, config_path: str) -> Optional[ExperimentModel]:
        return self.session.query(ExperimentModel).filter(
            ExperimentModel.config_path == config_path
        ).first()

    def get_all(self) -> List[ExperimentModel]:
        return self.session.query(ExperimentModel).order_by(
            ExperimentModel.experiment_name
        ).all()

    def update(
        self, experiment_id: str, updates: Dict[str, Any]
    ) -> Optional[ExperimentModel]:
        experiment = self.get_by_id(experiment_id)
        if not experiment:
            return None
        for key, value in updates.items():
            if key == "name":
                key = "experiment_name"
            if hasattr(experiment, key):
                setattr(experiment, key, value)
        experiment.updated_at = datetime.utcnow()
        self.session.flush()
        return experiment

    def delete(self, experiment_id: str) -> bool:
        experiment = self.get_by_id(experiment_id)
        if not experiment:
            return False
        self.session.delete(experiment)
        self.session.flush()
        logger.debug(f"Deleted experiment: {experiment_id}")
        return True


class ExperimentRunRepository:
    """Repository for ExperimentRun operations."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, run_data: Dict[str, Any]) -> ExperimentRunModel:
        run = ExperimentRunModel(**run_data)
        self.session.add(run)
        self.session.flush()
        logger.debug(f"Created experiment run: {run.id}")
        return run

    def get_by_id(self, run_id: str) -> Optional[ExperimentRunModel]:
        return self.session.query(ExperimentRunModel).filter(
            ExperimentRunModel.id == run_id
        ).first()

    def get_by_experiment_id(self, experiment_id: str) -> List[ExperimentRunModel]:
        return (
            self.session.query(ExperimentRunModel)
            .filter(ExperimentRunModel.experiment_id == experiment_id)
            .order_by(ExperimentRunModel.run_number.desc())
            .all()
        )

    def get_active_runs_for_experiment(
        self, experiment_id: str
    ) -> List[ExperimentRunModel]:
        return (
            self.session.query(ExperimentRunModel)
            .filter(
                ExperimentRunModel.experiment_id == experiment_id,
                ExperimentRunModel.status.in_(
                    [
                        ExperimentRunStatus.PENDING,
                        ExperimentRunStatus.RUNNING,
                        ExperimentRunStatus.STOPPING,
                    ]
                ),
            )
            .all()
        )

    def get_next_run_number(self, experiment_id: str) -> int:
        result = (
            self.session.query(func.max(ExperimentRunModel.run_number))
            .filter(ExperimentRunModel.experiment_id == experiment_id)
            .scalar()
        )
        return (result or 0) + 1

    def update_status(
        self,
        run_id: str,
        status: ExperimentRunStatus,
        error_message: Optional[str] = None,
    ) -> Optional[ExperimentRunModel]:
        run = self.get_by_id(run_id)
        if not run:
            return None

        run.status = status
        if error_message:
            run.error_message = error_message

        if status == ExperimentRunStatus.RUNNING and not run.started_at:
            run.started_at = datetime.utcnow()
        elif status in (
            ExperimentRunStatus.COMPLETED,
            ExperimentRunStatus.FAILED,
            ExperimentRunStatus.CANCELLED,
        ):
            if not run.completed_at:
                run.completed_at = datetime.utcnow()

        self.session.flush()
        return run

    def get_all(
        self, experiment_id: Optional[str] = None
    ) -> List[ExperimentRunModel]:
        query = self.session.query(ExperimentRunModel)
        if experiment_id:
            query = query.filter(ExperimentRunModel.experiment_id == experiment_id)
        return query.order_by(ExperimentRunModel.created_at.desc()).all()

    def delete(self, run_id: str) -> bool:
        run = self.get_by_id(run_id)
        if not run:
            return False
        self.session.delete(run)
        self.session.flush()
        logger.debug(f"Deleted experiment run: {run_id}")
        return True

    def update_status_from_workflows(
        self, run_id: str
    ) -> Optional[ExperimentRunModel]:
        """Derive run status from its workflows."""
        run = self.get_by_id(run_id)
        if not run:
            return None

        completed = failed = running = 0
        for workflow in run.workflows:
            status_val = (
                workflow.status.value
                if hasattr(workflow.status, "value")
                else str(workflow.status)
            )
            if status_val == "completed":
                completed += 1
            elif status_val == "failed":
                failed += 1
            elif status_val == "running":
                running += 1

        total = len(run.workflows)
        if failed > 0:
            run.status = ExperimentRunStatus.FAILED
        elif total > 0 and completed == total:
            run.status = ExperimentRunStatus.COMPLETED
            if not run.completed_at:
                run.completed_at = datetime.utcnow()
        elif running > 0 or completed > 0:
            run.status = ExperimentRunStatus.RUNNING
            if not run.started_at:
                run.started_at = datetime.utcnow()
        else:
            run.status = ExperimentRunStatus.PENDING

        self.session.flush()
        return run


class ArtifactRepository:
    """Repository for Artifact operations."""
    
    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session
    
    def create(self, artifact_data: Dict[str, Any]) -> ArtifactModel:
        """Create a new artifact record.

        Accepts optional ``parent_artifact_ids`` (list of cache keys) and wires
        the ``parents`` relationship.
        """
        parent_ids = list(artifact_data.pop("parent_artifact_ids", None) or [])
        artifact = ArtifactModel(**artifact_data)
        if parent_ids:
            parents = (
                self.session.query(ArtifactModel)
                .filter(ArtifactModel.id.in_(parent_ids))
                .all()
            )
            by_id = {p.id: p for p in parents}
            # Preserve caller order
            artifact.parents = [by_id[pid] for pid in parent_ids if pid in by_id]
        self.session.add(artifact)
        self.session.flush()
        logger.debug(f"Created artifact: {artifact.id}")
        return artifact
    
    def get_by_id(self, artifact_id: str) -> Optional[ArtifactModel]:
        """Get artifact by ID (cache key)."""
        return self.session.query(ArtifactModel).filter(
            ArtifactModel.id == artifact_id
        ).first()
    
    def get_by_task_id(self, task_id: str) -> List[ArtifactModel]:
        """Get artifacts for a task."""
        return self.session.query(ArtifactModel).filter(
            ArtifactModel.task_id == task_id
        ).all()
    
    def update_last_accessed(self, artifact_id: str) -> Optional[ArtifactModel]:
        """Update artifact last accessed timestamp."""
        artifact = self.get_by_id(artifact_id)
        if not artifact:
            return None
        
        artifact.last_accessed_at = datetime.utcnow()
        self.session.flush()
        return artifact
    
    def delete(self, artifact_id: str) -> bool:
        """Delete an artifact record."""
        artifact = self.get_by_id(artifact_id)
        if not artifact:
            return False
        
        self.session.delete(artifact)
        self.session.flush()
        logger.debug(f"Deleted artifact: {artifact_id}")
        return True
    
    def get_total_size(self) -> int:
        """Get total size of all artifacts in bytes."""
        result = self.session.query(
            func.sum(ArtifactModel.size_bytes)
        ).scalar()
        return result or 0
    
    def get_by_run_id(self, run_id: str) -> List[ArtifactModel]:
        """Get all artifacts for an experiment run (alias: experiment_run_id)."""
        return self.session.query(ArtifactModel).filter(
            ArtifactModel.experiment_run_id == run_id
        ).all()
    
    def delete_by_run_id(self, run_id: str) -> int:
        """Delete all artifacts created by a specific run. Returns count deleted."""
        artifacts = self.get_by_run_id(run_id)
        count = len(artifacts)
        for artifact in artifacts:
            self.session.delete(artifact)
        self.session.flush()
        logger.info(f"Deleted {count} artifacts for run {run_id}")
        return count
