"""
Edge cases and integration tests for the backend API.

Tests for:
- Boundary conditions
- Race conditions
- State consistency
- Tool management
- Dataset endpoints
- Statistics endpoints
- Error recovery
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.backend.api import app
from src.pipeline.tasks import Task, TaskStatus, TaskType, clear_task_registry
from src.pipeline.tools import ToolDefinition, ContainerConfig, init_tool_registry
from src.pipeline.workflow import Workflow, WorkflowFactory
from src.pipeline.pipeline import DefenseEvaluationPipeline


# ============================================================================
# Fixtures
# ============================================================================


# Fixtures are in conftest.py


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def initialized_pipeline():
    """Create an initialized pipeline."""
    from .conftest import create_task_with_id
    
    tool = ToolDefinition(
        name="test_tool",
        container=ContainerConfig(image="test/image:latest", command="python main.py")
    )
    
    task = create_task_with_id(
        task_id="task_1",
        tool=tool,
        task_type=TaskType.PRE_TRAINING
    )
    
    workflow = WorkflowFactory.create_workflow(name="test_workflow", tasks=[task])
    pipeline = DefenseEvaluationPipeline(name="test_pipeline", workflows=[workflow])
    
    from src.backend.api import _scheduler_state
    _scheduler_state.initialize(pipeline)
    
    return pipeline


# ============================================================================
# Test: Boundary Conditions
# ============================================================================


class TestBoundaryConditions:
    """Tests for boundary conditions and edge cases."""
    
    def test_empty_pipeline(self, client):
        """API should handle empty pipeline gracefully."""
        clear_task_registry()
        
        # Create pipeline with no workflows
        pipeline = DefenseEvaluationPipeline(name="empty_pipeline", workflows=[])
        
        from src.backend.api import _scheduler_state
        _scheduler_state.initialize(pipeline)
        
        # Getting next task should return no task
        response = client.get("/tasks/next")
        assert response.status_code == 200
        assert response.json()["has_task"] is False
        
        # Progress should show zero tasks
        response = client.get("/progress")
        assert response.status_code == 200
        assert response.json()["total"] == 0
    
    def test_single_task_pipeline(self, client, initialized_pipeline):
        """API should handle pipeline with single task."""
        response = client.get("/tasks/next")
        assert response.status_code == 200
        assert response.json()["has_task"] is True
        
        # After completing, should have no more tasks
        client.put("/tasks/status", json={"task_id": "task_1", "status": "completed"})
        
        response = client.get("/tasks/next")
        assert response.status_code == 200
        assert response.json()["has_task"] is False
        assert "All tasks completed" in response.json()["message"]
    
    def test_large_number_of_tasks(self, client):
        """API should handle pipelines with many tasks."""
        clear_task_registry()
        
        tool = ToolDefinition(
            name="test_tool",
            container=ContainerConfig(image="test/image:latest", command="python main.py")
        )
        from .conftest import create_task_with_id
        
        tasks = []
        for i in range(100):
            task = create_task_with_id(
                task_id=f"task_{i}",
                tool=tool,
                task_type=TaskType.PRE_TRAINING,
                dependencies=[],
                priority=100 - i,
            )
            tasks.append(task)
        
        workflow = WorkflowFactory.create_workflow(name="large_workflow", tasks=tasks)
        pipeline = DefenseEvaluationPipeline(name="large_pipeline", workflows=[workflow])
        
        from src.backend.api import _scheduler_state
        _scheduler_state.initialize(pipeline)
        
        # Should be able to get tasks
        response = client.get("/tasks")
        assert response.status_code == 200
        assert response.json()["total"] == 100
        
        # Should be able to get next task
        response = client.get("/tasks/next")
        assert response.status_code == 200
        assert response.json()["has_task"] is True
    
    def test_task_with_many_dependencies(self, client):
        """API should handle tasks with many dependencies."""
        clear_task_registry()
        
        tool = ToolDefinition(
            name="test_tool",
            container=ContainerConfig(image="test/image:latest", command="python main.py")
        )
        
        from .conftest import create_task_with_id
        
        # Create 10 tasks in a chain
        tasks = []
        for i in range(10):
            task = create_task_with_id(
                task_id=f"task_{i}",
                tool=tool,
                task_type=TaskType.PRE_TRAINING,
                dependencies=tasks.copy() if tasks else [],
                priority=100 - i,
            )
            task.workflows = {"workflow_1"}
            tasks.append(task)
        
        workflow = WorkflowFactory.create_workflow(name="chain_workflow", tasks=tasks)
        pipeline = DefenseEvaluationPipeline(name="chain_pipeline", workflows=[workflow])
        
        from src.backend.api import _scheduler_state
        _scheduler_state.initialize(pipeline)
        
        # Last task should have 9 dependencies
        last_task = tasks[-1]
        response = client.get(f"/tasks/{last_task.id}")
        assert response.status_code == 200
        assert len(response.json()["dependency_ids"]) == 9


# ============================================================================
# Test: State Consistency
# ============================================================================


class TestStateConsistency:
    """Tests for state consistency across endpoints."""
    
    def test_task_status_consistency(self, client, initialized_pipeline):
        """Task status should be consistent across all endpoints."""
        # Get task via /tasks/next (sets to running)
        response = client.get("/tasks/next")
        task_id = response.json()["task"]["id"]
        
        # Verify status is running in all endpoints
        response = client.get(f"/tasks/{task_id}")
        assert response.json()["status"] == "running"
        
        response = client.get("/tasks?status=running")
        assert response.json()["total"] == 1
        assert response.json()["tasks"][0]["status"] == "running"
        
        response = client.get("/progress")
        assert response.json()["running"] == 1
        assert response.json()["pending"] == 0
    
    def test_worker_task_assignment_consistency(self, client, initialized_pipeline):
        """Worker task assignment should be consistent."""
        # Register worker and claim task
        worker_response = client.post(
            "/workers/register",
            json={"hostname": "worker1.example.com"}
        )
        worker_id = worker_response.json()["worker_id"]
        
        client.post(f"/workers/{worker_id}/claim")
        
        # Verify consistency
        worker_info = client.get(f"/workers/{worker_id}")
        task_id = worker_info.json()["current_task_id"]
        
        worker_task = client.get(f"/workers/{worker_id}/task")
        assert worker_task.json()["task"]["id"] == task_id
        
        task_info = client.get(f"/tasks/{task_id}")
        assert task_info.status_code == 200
        assert task_info.json()["status"] == "running"
    
    def test_progress_consistency_after_updates(self, client, initialized_pipeline):
        """Progress should be consistent after status updates."""
        # Initial state
        response = client.get("/progress")
        assert response.json()["pending"] == 1
        assert response.json()["completed"] == 0
        
        # Complete task
        client.get("/tasks/next")
        client.put("/tasks/status", json={"task_id": "task_1", "status": "completed"})
        
        # Progress should be updated
        response = client.get("/progress")
        assert response.json()["pending"] == 0
        assert response.json()["completed"] == 1
        assert response.json()["is_complete"] is True


# ============================================================================
# Test: Tool Management
# ============================================================================


class TestToolManagement:
    """Tests for tool management endpoints."""
    
    def test_list_tools(self, client):
        """Listing tools should return all available tools."""
        # Initialize tool registry
        with patch('src.backend.api._scheduler_state.get_all_tools') as mock_get:
            mock_get.return_value = {
                "tool1": {
                    "name": "tool1",
                    "container": {
                        "image": "test/image:latest",
                        "command": "python main.py"
                    },
                    "is_baseline": False
                }
            }
            
            response = client.get("/tools")
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert len(data["tools"]) == 1
    
    def test_get_tool_not_found(self, client):
        """Getting non-existent tool should return 404."""
        response = client.get("/tools/nonexistent_tool")
        assert response.status_code == 404
    
    def test_add_tool(self, client):
        """Adding a tool should add it to the registry."""
        response = client.post(
            "/tools",
            json={
                "name": "new_tool",
                "image": "test/new_tool:latest",
                "command": "python run.py",
                "is_baseline": False
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "new_tool"
        assert data["container"]["image"] == "test/new_tool:latest"
        
        # Tool should be retrievable
        response = client.get("/tools/new_tool")
        assert response.status_code == 200
        assert response.json()["name"] == "new_tool"
    
    def test_add_tool_duplicate_name(self, client):
        """Adding tool with duplicate name should overwrite or reject."""
        # Add tool
        client.post(
            "/tools",
            json={
                "name": "duplicate_tool",
                "image": "test/tool:latest",
                "command": "python main.py"
            }
        )
        
        # Try to add again with different image
        response = client.post(
            "/tools",
            json={
                "name": "duplicate_tool",
                "image": "test/different:latest",
                "command": "python main.py"
            }
        )
        # Should either accept (overwrite) or reject
        assert response.status_code in [200, 400, 409]


# ============================================================================
# Test: Dataset Endpoints
# ============================================================================


class TestDatasetEndpoints:
    """Tests for dataset information endpoints."""
    
    def test_get_dataset_info_no_context(self, client):
        """Getting dataset info without context should return unavailable."""
        with patch('src.backend.api.get_backend_context', return_value=None):
            response = client.get("/dataset")
            assert response.status_code == 200
            data = response.json()
            assert data["available"] is False
    
    def test_get_dataset_info_with_context(self, client):
        """Getting dataset info with context should return dataset info."""
        from src.backend.initialization import BackendContext
        
        mock_context = MagicMock()
        mock_context.dataset_info = {
            "name": "cifar10",
            "variant": "clean",
            "train_samples": 50000,
            "test_samples": 10000,
            "output_dir": "/tmp/dataset",
            "minio_key": "datasets/cifar10_clean"
        }
        mock_context.pipeline = MagicMock()
        mock_context.pipeline.model = {"script": "configs/model/config_model.py"}
        mock_context.pipeline.dataset = {"name": "cifar10", "variant": "clean"}
        mock_context.store = MagicMock()
        mock_context.store.is_available = True
        
        with patch('src.backend.initialization.get_backend_context', return_value=mock_context):
            response = client.get("/dataset")
            assert response.status_code == 200
            data = response.json()
            assert data["available"] is True
            assert data["name"] == "cifar10"
            assert data["variant"] == "clean"
            assert data["train_samples"] == 50000
    
    def test_get_dataset_download_url_no_dataset(self, client):
        """Getting download URL without dataset should return 404."""
        with patch('src.backend.api.get_backend_context', return_value=None):
            response = client.get("/dataset/download-url")
            assert response.status_code == 404


# ============================================================================
# Test: Statistics Endpoints
# ============================================================================


class TestStatisticsEndpoints:
    """Tests for statistics and system information endpoints."""
    
    def test_get_database_stats_no_db(self, client):
        """Getting database stats without DB should return unavailable."""
        response = client.get("/stats/database")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is False
    
    def test_get_store_stats_no_store(self, client):
        """Getting store stats without store should return unavailable."""
        with patch('src.backend.api.get_backend_context', return_value=None):
            response = client.get("/stats/store")
            assert response.status_code == 200
            data = response.json()
            assert data["available"] is False
    
    def test_get_system_stats(self, client, initialized_pipeline):
        """Getting system stats should return system information."""
        response = client.get("/stats/system")
        assert response.status_code == 200
        data = response.json()
        assert "scheduler_active" in data
        assert "database_available" in data
        assert "store_available" in data
        assert "workers_registered" in data
        assert data["scheduler_active"] is True


# ============================================================================
# Test: Error Recovery
# ============================================================================


class TestErrorRecovery:
    """Tests for error recovery and resilience."""
    
    def test_recover_from_failed_task(self, client, initialized_pipeline):
        """System should recover gracefully from failed tasks."""
        # Fail a task
        client.get("/tasks/next")
        client.put(
            "/tasks/status",
            json={
                "task_id": "task_1",
                "status": "failed",
                "error_message": "Task failed"
            }
        )
        
        # System should still be functional
        response = client.get("/health")
        assert response.status_code == 200
        
        response = client.get("/progress")
        assert response.status_code == 200
        assert response.json()["failed"] == 1
    
    def test_recover_from_invalid_state(self, client, initialized_pipeline):
        """System should handle invalid state transitions gracefully."""
        # Try to update task that's not running
        response = client.put(
            "/tasks/status",
            json={
                "task_id": "task_1",
                "status": "completed"
            }
        )
        # Should either succeed (if allowed) or fail gracefully
        assert response.status_code in [200, 400, 422]
        assert response.status_code != 500
    
    def test_handle_missing_task_metadata(self, client, initialized_pipeline):
        """System should handle missing task metadata gracefully."""
        # Get task (creates metadata entry)
        client.get("/tasks/next")
        
        # Manually clear metadata
        from src.backend.api import _scheduler_state
        _scheduler_state.task_metadata.clear()
        
        # Getting task should still work
        response = client.get("/tasks/task_1")
        assert response.status_code == 200
        # Metadata fields should be None
        assert response.json()["error_message"] is None


# ============================================================================
# Test: Priority and Scheduling Edge Cases
# ============================================================================


class TestPriorityEdgeCases:
    """Tests for priority calculation and scheduling edge cases."""
    
    def test_get_task_priority_uninitialized(self, client):
        """Getting task priority should fail when scheduler not initialized."""
        response = client.get("/tasks/task_1/priority")
        assert response.status_code == 503
    
    def test_get_task_priority_not_found(self, client, initialized_pipeline):
        """Getting priority for non-existent task should return 404."""
        response = client.get("/tasks/nonexistent_task/priority")
        assert response.status_code == 404
    
    def test_get_task_priority_success(self, client, initialized_pipeline):
        """Getting task priority should return priority information."""
        response = client.get("/tasks/task_1/priority")
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task_1"
        assert "priority" in data
        assert "dependency_level" in data
        assert "usage_counter" in data
        assert data["dependency_level"] == 0  # No dependencies
        assert data["usage_counter"] == 1
    
    def test_get_priority_levels(self, client, initialized_pipeline):
        """Getting priority levels should group tasks by depth."""
        response = client.get("/progress/levels")
        assert response.status_code == 200
        data = response.json()
        assert "levels" in data
        assert "0" in data["levels"]  # JSON object keys are strings
        assert len(data["levels"]["0"]) == 1


# ============================================================================
# Test: Task Logs Endpoint
# ============================================================================


class TestTaskLogs:
    """Tests for task logs endpoint."""
    
    def test_get_task_logs_uninitialized(self, client):
        """Getting task logs should fail when scheduler not initialized."""
        response = client.get("/tasks/task_1/logs")
        assert response.status_code == 503
    
    def test_get_task_logs_not_found(self, client, initialized_pipeline):
        """Getting logs for non-existent task should return 404."""
        response = client.get("/tasks/nonexistent_task/logs")
        assert response.status_code == 404
    
    def test_get_task_logs_no_metadata(self, client, initialized_pipeline):
        """Getting logs for task without metadata should return empty logs."""
        response = client.get("/tasks/task_1/logs")
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task_1"
        assert data["logs"] is None
    
    def test_get_task_logs_with_metadata(self, client, initialized_pipeline):
        """Getting logs for task with metadata should return logs."""
        # Complete task with result containing logs
        client.get("/tasks/next")
        client.put(
            "/tasks/status",
            json={
                "task_id": "task_1",
                "status": "completed",
                "result": {
                    "logs": "Task execution output",
                    "stdout": "Standard output",
                    "stderr": "Standard error"
                }
            }
        )
        
        response = client.get("/tasks/task_1/logs")
        assert response.status_code == 200
        data = response.json()
        assert data["logs"] == "Task execution output"
        assert "execution_time_ms" in data
