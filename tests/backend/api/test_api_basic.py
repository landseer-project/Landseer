"""
Basic API endpoint tests.

Tests for:
- Health checks
- Root endpoint
- Basic info endpoints
- Error handling for uninitialized scheduler
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock

from src.backend.api import app, get_scheduler_state, SchedulerState
from src.pipeline.tasks import Task, TaskStatus, TaskType, TaskFactory, clear_task_registry
from src.pipeline.tools import ToolDefinition, ContainerConfig
from src.pipeline.workflow import Workflow, WorkflowFactory
from src.pipeline.pipeline import DefenseEvaluationPipeline
from src.backend.scheduler.priority_scheduler import PriorityScheduler


# ============================================================================
# Fixtures
# ============================================================================


# Fixtures are in conftest.py


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_tool():
    """Create a sample tool."""
    return ToolDefinition(
        name="test_tool",
        container=ContainerConfig(
            image="test/image:latest",
            command="python main.py"
        )
    )


@pytest.fixture
def initialized_pipeline(sample_tool):
    """Create an initialized pipeline with scheduler."""
    from .conftest import create_task_with_id
    
    task = create_task_with_id(
        task_id="task_1",
        tool=sample_tool,
        task_type=TaskType.PRE_TRAINING,
        config={"param": "value"}
    )
    
    workflow = WorkflowFactory.create_workflow(
        name="test_workflow",
        tasks=[task]
    )
    
    pipeline = DefenseEvaluationPipeline(
        name="test_pipeline",
        workflows=[workflow]
    )
    
    from src.backend.api import _scheduler_state
    _scheduler_state.initialize(pipeline)
    
    return pipeline


# ============================================================================
# Test: Root and Health Endpoints
# ============================================================================


class TestRootAndHealth:
    """Tests for root and health check endpoints."""
    
    def test_root_endpoint(self, client):
        """Root endpoint should return API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Landseer Scheduler API"
        assert "version" in data
        assert "docs" in data
        assert "health" in data
    
    def test_health_check_uninitialized(self, client):
        """Health check should work even when scheduler is not initialized."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert data["scheduler_active"] is False
    
    def test_health_check_initialized(self, client, initialized_pipeline):
        """Health check should show scheduler as active when initialized."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["scheduler_active"] is True


# ============================================================================
# Test: Pipeline Info Endpoints
# ============================================================================


class TestPipelineInfo:
    """Tests for pipeline information endpoints."""
    
    def test_get_pipeline_info_uninitialized(self, client):
        """Getting pipeline info should fail when scheduler not initialized."""
        response = client.get("/info/pipeline")
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"].lower()
    
    def test_get_pipeline_info_initialized(self, client, initialized_pipeline):
        """Getting pipeline info should return correct data."""
        response = client.get("/info/pipeline")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == initialized_pipeline.id
        assert data["name"] == initialized_pipeline.name
        assert data["workflow_count"] == len(initialized_pipeline.workflows)
        assert data["task_count"] == 1
    
    def test_get_workflows_uninitialized(self, client):
        """Getting workflows should fail when scheduler not initialized."""
        response = client.get("/info/workflows")
        assert response.status_code == 503
    
    def test_get_workflows_initialized(self, client, initialized_pipeline):
        """Getting workflows should return all workflows."""
        response = client.get("/info/workflows")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["workflows"]) == 1
        assert data["workflows"][0]["id"] == initialized_pipeline.workflows[0].id
        assert data["workflows"][0]["name"] == initialized_pipeline.workflows[0].name


# ============================================================================
# Test: Task Management Endpoints - Basic
# ============================================================================


class TestTaskManagementBasic:
    """Basic tests for task management endpoints."""
    
    def test_get_next_task_uninitialized(self, client):
        """Getting next task should fail when scheduler not initialized."""
        response = client.get("/tasks/next")
        assert response.status_code == 503
    
    def test_get_next_task_no_tasks_ready(self, client, initialized_pipeline):
        """Getting next task when none ready should return appropriate message."""
        # Mark the only task as running
        from src.backend.api import _scheduler_state
        task = _scheduler_state.scheduler.get_all_tasks()[0]
        task.status = TaskStatus.RUNNING
        
        response = client.get("/tasks/next")
        assert response.status_code == 200
        data = response.json()
        assert data["has_task"] is False
        assert "No tasks ready" in data["message"] or "running" in data["message"].lower()
    
    def test_get_next_task_success(self, client, initialized_pipeline):
        """Getting next task should return a ready task."""
        response = client.get("/tasks/next")
        assert response.status_code == 200
        data = response.json()
        assert data["has_task"] is True
        assert data["task"] is not None
        assert data["task"]["id"] == "task_1"
        assert data["task"]["status"] == "running"  # Status updated to running
    
    def test_get_all_tasks_uninitialized(self, client):
        """Getting all tasks should fail when scheduler not initialized."""
        response = client.get("/tasks")
        assert response.status_code == 503
    
    def test_get_all_tasks_initialized(self, client, initialized_pipeline):
        """Getting all tasks should return all tasks."""
        response = client.get("/tasks")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["id"] == "task_1"
    
    def test_get_all_tasks_filtered_by_status(self, client, initialized_pipeline):
        """Getting tasks filtered by status should return only matching tasks."""
        # Mark task as completed
        from src.backend.api import _scheduler_state
        task = _scheduler_state.scheduler.get_all_tasks()[0]
        task.status = TaskStatus.COMPLETED
        
        # Get completed tasks
        response = client.get("/tasks?status=completed")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["tasks"][0]["status"] == "completed"
        
        # Get pending tasks (should be empty)
        response = client.get("/tasks?status=pending")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
    
    def test_get_all_tasks_invalid_status(self, client, initialized_pipeline):
        """Getting tasks with invalid status should return 400."""
        response = client.get("/tasks?status=invalid_status")
        assert response.status_code == 400
        assert "Invalid status" in response.json()["detail"]
    
    def test_get_task_by_id_uninitialized(self, client):
        """Getting task by ID should fail when scheduler not initialized."""
        response = client.get("/tasks/task_1")
        assert response.status_code == 503
    
    def test_get_task_by_id_not_found(self, client, initialized_pipeline):
        """Getting non-existent task should return 404."""
        response = client.get("/tasks/nonexistent_task")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_get_task_by_id_success(self, client, initialized_pipeline):
        """Getting task by ID should return task details."""
        response = client.get("/tasks/task_1")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "task_1"
        assert data["tool"]["name"] == "test_tool"


# ============================================================================
# Test: Task Status Updates
# ============================================================================


class TestTaskStatusUpdates:
    """Tests for task status update endpoint."""
    
    def test_update_task_status_uninitialized(self, client):
        """Updating task status should fail when scheduler not initialized."""
        response = client.put(
            "/tasks/status",
            json={
                "task_id": "task_1",
                "status": "completed"
            }
        )
        assert response.status_code == 503
    
    def test_update_task_status_invalid_status(self, client, initialized_pipeline):
        """Updating with invalid status should return 400."""
        response = client.put(
            "/tasks/status",
            json={
                "task_id": "task_1",
                "status": "invalid_status"
            }
        )
        assert response.status_code == 400
        assert "Invalid status" in response.json()["detail"]
    
    def test_update_task_status_pending_not_allowed(self, client, initialized_pipeline):
        """Updating to pending status should not be allowed."""
        response = client.put(
            "/tasks/status",
            json={
                "task_id": "task_1",
                "status": "pending"
            }
        )
        assert response.status_code == 400
    
    def test_update_task_status_running_not_allowed(self, client, initialized_pipeline):
        """Updating to running status should not be allowed."""
        response = client.put(
            "/tasks/status",
            json={
                "task_id": "task_1",
                "status": "running"
            }
        )
        assert response.status_code == 400
    
    def test_update_task_status_task_not_found(self, client, initialized_pipeline):
        """Updating non-existent task should return 404."""
        response = client.put(
            "/tasks/status",
            json={
                "task_id": "nonexistent_task",
                "status": "completed"
            }
        )
        assert response.status_code == 404
    
    def test_update_task_status_completed(self, client, initialized_pipeline):
        """Updating task to completed should succeed."""
        # First get a task (sets it to running)
        client.get("/tasks/next")
        
        response = client.put(
            "/tasks/status",
            json={
                "task_id": "task_1",
                "status": "completed",
                "execution_time_ms": 5000
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["task_id"] == "task_1"
        assert data["new_status"] == "completed"
        
        # Verify task status was updated
        task_response = client.get("/tasks/task_1")
        assert task_response.json()["status"] == "completed"
        assert task_response.json()["execution_time_ms"] == 5000
    
    def test_update_task_status_failed(self, client, initialized_pipeline):
        """Updating task to failed should succeed and store error message."""
        # First get a task
        client.get("/tasks/next")
        
        response = client.put(
            "/tasks/status",
            json={
                "task_id": "task_1",
                "status": "failed",
                "error_message": "Task execution failed",
                "execution_time_ms": 1000
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["new_status"] == "failed"
        
        # Verify error message was stored
        task_response = client.get("/tasks/task_1")
        assert task_response.json()["status"] == "failed"
        assert task_response.json()["error_message"] == "Task execution failed"
    
    def test_update_task_status_with_result_metadata(self, client, initialized_pipeline):
        """Updating task status with result metadata should store it."""
        client.get("/tasks/next")
        
        result_metadata = {
            "cache_hit": True,
            "cache_key": "abc123",
            "logs": "Task completed successfully"
        }
        
        response = client.put(
            "/tasks/status",
            json={
                "task_id": "task_1",
                "status": "completed",
                "result": result_metadata
            }
        )
        assert response.status_code == 200
        
        # Verify metadata was stored
        task_response = client.get("/tasks/task_1")
        assert task_response.json()["cache_hit"] is True
        assert task_response.json()["cache_key"] == "abc123"


# ============================================================================
# Test: Progress Endpoints
# ============================================================================


class TestProgressEndpoints:
    """Tests for progress and statistics endpoints."""
    
    def test_get_progress_uninitialized(self, client):
        """Getting progress should fail when scheduler not initialized."""
        response = client.get("/progress")
        assert response.status_code == 503
    
    def test_get_progress_initialized(self, client, initialized_pipeline):
        """Getting progress should return correct statistics."""
        response = client.get("/progress")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["pending"] == 1
        assert data["running"] == 0
        assert data["completed"] == 0
        assert data["failed"] == 0
        assert data["progress_percent"] == 0.0
        assert data["is_complete"] is False
    
    def test_get_progress_after_completion(self, client, initialized_pipeline):
        """Progress should reflect completed tasks."""
        # Complete the task
        client.get("/tasks/next")
        client.put(
            "/tasks/status",
            json={"task_id": "task_1", "status": "completed"}
        )
        
        response = client.get("/progress")
        assert response.status_code == 200
        data = response.json()
        assert data["completed"] == 1
        assert data["pending"] == 0
        assert data["progress_percent"] == 100.0
        assert data["is_complete"] is True
    
    def test_get_ready_tasks(self, client, initialized_pipeline):
        """Getting ready tasks should return tasks ready to execute."""
        response = client.get("/progress/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["tasks"][0]["id"] == "task_1"
    
    def test_get_blocked_tasks(self, client, initialized_pipeline):
        """Getting blocked tasks should return tasks with failed dependencies."""
        # Create a task with a dependency
        from src.backend.api import _scheduler_state
        from src.pipeline.tasks import TaskFactory
        
        task1 = _scheduler_state.scheduler.get_all_tasks()[0]
        task1.status = TaskStatus.FAILED
        
        # Create a dependent task
        tool = ToolDefinition(
            name="dependent_tool",
            container=ContainerConfig(image="test/image:latest", command="python main.py")
        )
        task2 = TaskFactory.create_task(
            task_type=TaskType.POST_TRAINING,
            tool=tool,
            config={},
            dependencies=[task1]
        )
        
        workflow = WorkflowFactory.create_workflow(
            name="test_workflow_2",
            tasks=[task1, task2]
        )
        _scheduler_state.pipeline.workflows.append(workflow)
        _scheduler_state.scheduler._all_tasks.append(task2)
        
        response = client.get("/progress/blocked")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["tasks"][0]["id"] == task2.id


# ============================================================================
# Test: Scheduler Management
# ============================================================================


class TestSchedulerManagement:
    """Tests for scheduler management endpoints."""
    
    def test_get_scheduler_status_uninitialized(self, client):
        """Getting scheduler status should work even when not initialized."""
        response = client.get("/scheduler/status")
        assert response.status_code == 200
        data = response.json()
        assert data["initialized"] is False
    
    def test_get_scheduler_status_initialized(self, client, initialized_pipeline):
        """Getting scheduler status should return correct information."""
        response = client.get("/scheduler/status")
        assert response.status_code == 200
        data = response.json()
        assert data["initialized"] is True
        assert data["pipeline_name"] == "test_pipeline"
        assert "started_at" in data
    
    def test_reset_scheduler_uninitialized(self, client):
        """Resetting scheduler should fail when not initialized."""
        response = client.post("/scheduler/reset")
        assert response.status_code == 503
    
    def test_reset_scheduler_initialized(self, client, initialized_pipeline):
        """Resetting scheduler should reset all tasks to pending."""
        # Complete a task
        client.get("/tasks/next")
        client.put(
            "/tasks/status",
            json={"task_id": "task_1", "status": "completed"}
        )
        
        # Verify it's completed
        task_response = client.get("/tasks/task_1")
        assert task_response.json()["status"] == "completed"
        
        # Reset scheduler
        response = client.post("/scheduler/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Verify task is back to pending
        task_response = client.get("/tasks/task_1")
        assert task_response.json()["status"] == "pending"
    
    def test_get_scheduler_next_preview(self, client, initialized_pipeline):
        """Previewing next task should not change task status."""
        response = client.get("/scheduler/next")
        assert response.status_code == 200
        data = response.json()
        assert data["has_next"] is True
        assert data["next_task"]["id"] == "task_1"
        
        # Task should still be pending (not changed to running)
        task_response = client.get("/tasks/task_1")
        assert task_response.json()["status"] == "pending"


# ============================================================================
# Test: Worker Management
# ============================================================================


class TestWorkerManagement:
    """Tests for worker management endpoints."""
    
    def test_register_worker(self, client):
        """Registering a worker should return worker info."""
        response = client.post(
            "/workers/register",
            json={
                "hostname": "worker1.example.com",
                "capabilities": {"gpu": True, "gpu_id": 0}
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "worker_id" in data
        assert data["hostname"] == "worker1.example.com"
        assert data["status"] == "idle"
        assert data["capabilities"]["gpu"] is True
    
    def test_register_worker_with_custom_id(self, client):
        """Registering worker with custom ID should use that ID."""
        response = client.post(
            "/workers/register",
            json={
                "worker_id": "custom_worker_123",
                "hostname": "worker1.example.com"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["worker_id"] == "custom_worker_123"
    
    def test_list_workers_empty(self, client):
        """Listing workers when none registered should return empty list."""
        response = client.get("/workers")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["active"] == 0
        assert len(data["workers"]) == 0
    
    def test_list_workers_with_registered(self, client):
        """Listing workers should return all registered workers."""
        # Register two workers
        client.post(
            "/workers/register",
            json={"hostname": "worker1.example.com"}
        )
        client.post(
            "/workers/register",
            json={"hostname": "worker2.example.com"}
        )
        
        response = client.get("/workers")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert data["active"] == 2
        assert len(data["workers"]) == 2
    
    def test_get_worker_not_found(self, client):
        """Getting non-existent worker should return 404."""
        response = client.get("/workers/nonexistent_worker")
        assert response.status_code == 404
    
    def test_get_worker_success(self, client):
        """Getting worker should return worker information."""
        # Register a worker
        register_response = client.post(
            "/workers/register",
            json={"hostname": "worker1.example.com"}
        )
        worker_id = register_response.json()["worker_id"]
        
        # Get the worker
        response = client.get(f"/workers/{worker_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["worker_id"] == worker_id
        assert data["hostname"] == "worker1.example.com"
    
    def test_worker_heartbeat_not_found(self, client):
        """Heartbeat for non-existent worker should return 404."""
        response = client.post(
            "/workers/nonexistent_worker/heartbeat",
            json={"worker_id": "nonexistent_worker"}
        )
        assert response.status_code == 404
    
    def test_worker_heartbeat_success(self, client):
        """Worker heartbeat should update last_heartbeat timestamp."""
        # Register a worker
        register_response = client.post(
            "/workers/register",
            json={"hostname": "worker1.example.com"}
        )
        worker_id = register_response.json()["worker_id"]
        
        # Get initial heartbeat
        worker_response = client.get(f"/workers/{worker_id}")
        initial_heartbeat = worker_response.json()["last_heartbeat"]
        
        # Send heartbeat
        import time
        time.sleep(0.1)  # Small delay to ensure timestamp difference
        response = client.post(
            f"/workers/{worker_id}/heartbeat",
            json={"worker_id": worker_id, "status": "busy"}
        )
        assert response.status_code == 200
        
        # Verify heartbeat was updated
        worker_response = client.get(f"/workers/{worker_id}")
        new_heartbeat = worker_response.json()["last_heartbeat"]
        assert new_heartbeat != initial_heartbeat
        assert worker_response.json()["status"] == "busy"
    
    def test_worker_claim_task_not_found(self, client, initialized_pipeline):
        """Claiming task for non-existent worker should return 404."""
        response = client.post("/workers/nonexistent_worker/claim")
        assert response.status_code == 404
    
    def test_worker_claim_task_success(self, client, initialized_pipeline):
        """Worker claiming task should associate task with worker."""
        # Register a worker
        register_response = client.post(
            "/workers/register",
            json={"hostname": "worker1.example.com"}
        )
        worker_id = register_response.json()["worker_id"]
        
        # Claim a task
        response = client.post(f"/workers/{worker_id}/claim")
        assert response.status_code == 200
        data = response.json()
        assert data["has_task"] is True
        assert data["task"]["id"] == "task_1"
        
        # Verify worker has the task assigned
        worker_response = client.get(f"/workers/{worker_id}/task")
        assert worker_response.json()["has_task"] is True
        assert worker_response.json()["task"]["id"] == "task_1"
        
        # Verify worker status is busy
        worker_info = client.get(f"/workers/{worker_id}")
        assert worker_info.json()["status"] == "busy"
        assert worker_info.json()["current_task_id"] == "task_1"
