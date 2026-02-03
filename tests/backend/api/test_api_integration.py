"""
Integration tests for the backend API.

Tests for:
- End-to-end workflows
- Multiple workers coordination
- Task lifecycle management
- Pipeline execution flow
- Error propagation
- State persistence
"""

import pytest
import time
import threading
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.backend.api import app
from src.pipeline.tasks import Task, TaskStatus, TaskType, clear_task_registry
from src.pipeline.tools import ToolDefinition, ContainerConfig
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
def dependency_pipeline():
    """Create a pipeline with task dependencies."""
    import uuid
    from .conftest import create_task_with_id
    
    # Use unique pipeline ID for each test
    pipeline_id = f"pipeline_{uuid.uuid4().hex[:8]}"
    
    tool = ToolDefinition(
        name="test_tool",
        container=ContainerConfig(image="test/image:latest", command="python main.py")
    )
    
    # Task A (no dependencies)
    task_a = create_task_with_id(
        task_id="task_a",
        tool=tool,
        task_type=TaskType.PRE_TRAINING,
        priority=100,
        pipeline_id=pipeline_id
    )
    task_a.workflows = {"workflow_1"}
    
    # Task B (depends on A)
    task_b = create_task_with_id(
        task_id="task_b",
        tool=tool,
        task_type=TaskType.POST_TRAINING,
        dependencies=[task_a],
        priority=90,
        pipeline_id=pipeline_id
    )
    task_b.workflows = {"workflow_1"}
    
    # Task C (depends on B)
    task_c = create_task_with_id(
        task_id="task_c",
        tool=tool,
        task_type=TaskType.DEPLOYMENT,
        dependencies=[task_b],
        priority=80,
        pipeline_id=pipeline_id
    )
    task_c.workflows = {"workflow_1"}
    
    workflow = WorkflowFactory.create_workflow(
        name="dependency_workflow",
        tasks=[task_a, task_b, task_c]
    )
    
    pipeline = DefenseEvaluationPipeline(
        name="dependency_pipeline",
        workflows=[workflow]
    )
    pipeline.id = pipeline_id
    
    from src.backend.api import _scheduler_state
    _scheduler_state.initialize(pipeline)
    
    return pipeline


# ============================================================================
# Test: End-to-End Workflow Execution
# ============================================================================


class TestEndToEndWorkflow:
    """Tests for complete workflow execution."""
    
    def test_complete_task_lifecycle(self, client, dependency_pipeline):
        """Complete task lifecycle: claim -> execute -> complete."""
        # Register worker
        worker_response = client.post(
            "/workers/register",
            json={"hostname": "worker1.example.com"}
        )
        worker_id = worker_response.json()["worker_id"]
        
        # Claim task A
        claim_response = client.post(f"/workers/{worker_id}/claim")
        assert claim_response.json()["has_task"] is True
        task_a_id = claim_response.json()["task"]["id"]
        assert task_a_id == "task_a"
        
        # Verify task is running
        task_response = client.get(f"/tasks/{task_a_id}")
        assert task_response.json()["status"] == "running"
        
        # Complete task A
        complete_response = client.put(
            "/tasks/status",
            json={
                "task_id": task_a_id,
                "status": "completed",
                "execution_time_ms": 1000
            }
        )
        assert complete_response.json()["success"] is True
        
        # Verify task A is completed
        task_response = client.get(f"/tasks/{task_a_id}")
        assert task_response.json()["status"] == "completed"
        
        # Task B should now be ready
        claim_response = client.post(f"/workers/{worker_id}/claim")
        assert claim_response.json()["has_task"] is True
        assert claim_response.json()["task"]["id"] == "task_b"
    
    def test_dependency_chain_execution(self, client, dependency_pipeline):
        """Tasks should execute in dependency order."""
        # Register worker
        worker_response = client.post(
            "/workers/register",
            json={"hostname": "worker1.example.com"}
        )
        worker_id = worker_response.json()["worker_id"]
        
        # Task A should be available first
        claim_response = client.post(f"/workers/{worker_id}/claim")
        assert claim_response.json()["task"]["id"] == "task_a"
        
        # Complete A
        client.put(
            "/tasks/status",
            json={"task_id": "task_a", "status": "completed"}
        )
        
        # Task B should be available next
        claim_response = client.post(f"/workers/{worker_id}/claim")
        assert claim_response.json()["task"]["id"] == "task_b"
        
        # Complete B
        client.put(
            "/tasks/status",
            json={"task_id": "task_b", "status": "completed"}
        )
        
        # Task C should be available last
        claim_response = client.post(f"/workers/{worker_id}/claim")
        assert claim_response.json()["task"]["id"] == "task_c"
    
    def test_failed_task_blocks_dependents(self, client, dependency_pipeline):
        """Failed task should block dependent tasks."""
        # Register worker
        worker_response = client.post(
            "/workers/register",
            json={"hostname": "worker1.example.com"}
        )
        worker_id = worker_response.json()["worker_id"]
        
        # Complete task A
        claim_response = client.post(f"/workers/{worker_id}/claim")
        client.put(
            "/tasks/status",
            json={"task_id": "task_a", "status": "completed"}
        )
        
        # Fail task B
        claim_response = client.post(f"/workers/{worker_id}/claim")
        client.put(
            "/tasks/status",
            json={
                "task_id": "task_b",
                "status": "failed",
                "error_message": "Task B failed"
            }
        )
        
        # Task C should not be available (blocked by failed B)
        claim_response = client.post(f"/workers/{worker_id}/claim")
        assert claim_response.json()["has_task"] is False
        assert "blocked" in claim_response.json()["message"].lower() or \
               "no tasks" in claim_response.json()["message"].lower()
        
        # Verify task C is blocked
        blocked_response = client.get("/progress/blocked")
        assert blocked_response.json()["total"] == 1
        assert blocked_response.json()["tasks"][0]["id"] == "task_c"


# ============================================================================
# Test: Multiple Workers Coordination
# ============================================================================


class TestMultipleWorkers:
    """Tests for multiple workers coordinating."""
    
    def test_multiple_workers_claim_different_tasks(self, client, dependency_pipeline):
        """Multiple workers should be able to claim different tasks."""
        # Register two workers
        worker1_response = client.post(
            "/workers/register",
            json={"hostname": "worker1.example.com"}
        )
        worker1_id = worker1_response.json()["worker_id"]
        
        worker2_response = client.post(
            "/workers/register",
            json={"hostname": "worker2.example.com"}
        )
        worker2_id = worker2_response.json()["worker_id"]
        
        # Both should be able to claim tasks (if available)
        claim1 = client.post(f"/workers/{worker1_id}/claim")
        assert claim1.json()["has_task"] is True
        task1_id = claim1.json()["task"]["id"]
        
        # Complete task 1
        client.put(
            "/tasks/status",
            json={"task_id": task1_id, "status": "completed"}
        )
        
        # Worker 2 should be able to claim next task
        claim2 = client.post(f"/workers/{worker2_id}/claim")
        assert claim2.json()["has_task"] is True
        task2_id = claim2.json()["task"]["id"]
        
        # Tasks should be different
        assert task1_id != task2_id
    
    def test_worker_heartbeat_tracking(self, client):
        """Worker heartbeats should be tracked correctly."""
        # Register worker
        worker_response = client.post(
            "/workers/register",
            json={"hostname": "worker1.example.com"}
        )
        worker_id = worker_response.json()["worker_id"]
        
        # Get initial heartbeat
        worker_info = client.get(f"/workers/{worker_id}")
        initial_heartbeat = worker_info.json()["last_heartbeat"]
        
        # Wait a bit and send heartbeat
        time.sleep(0.1)
        client.post(
            f"/workers/{worker_id}/heartbeat",
            json={"worker_id": worker_id}
        )
        
        # Verify heartbeat updated
        worker_info = client.get(f"/workers/{worker_id}")
        new_heartbeat = worker_info.json()["last_heartbeat"]
        assert new_heartbeat != initial_heartbeat
    
    def test_worker_task_completion_tracking(self, client, dependency_pipeline):
        """Worker task completion should be tracked."""
        # Register worker
        worker_response = client.post(
            "/workers/register",
            json={"hostname": "worker1.example.com"}
        )
        worker_id = worker_response.json()["worker_id"]
        
        # Initial stats
        worker_info = client.get(f"/workers/{worker_id}")
        assert worker_info.json()["tasks_completed"] == 0
        assert worker_info.json()["tasks_failed"] == 0
        
        # Complete a task
        claim_response = client.post(f"/workers/{worker_id}/claim")
        task_id = claim_response.json()["task"]["id"]
        client.put(
            "/tasks/status",
            json={"task_id": task_id, "status": "completed"}
        )
        
        # Verify stats updated
        worker_info = client.get(f"/workers/{worker_id}")
        assert worker_info.json()["tasks_completed"] == 1
        assert worker_info.json()["tasks_failed"] == 0
        
        # Fail a task
        claim_response = client.post(f"/workers/{worker_id}/claim")
        task_id = claim_response.json()["task"]["id"]
        client.put(
            "/tasks/status",
            json={"task_id": task_id, "status": "failed", "error_message": "Error"}
        )
        
        # Verify stats updated
        worker_info = client.get(f"/workers/{worker_id}")
        assert worker_info.json()["tasks_completed"] == 1
        assert worker_info.json()["tasks_failed"] == 1


# ============================================================================
# Test: Race Conditions
# ============================================================================


class TestRaceConditions:
    """Tests for race condition handling."""
    
    def test_concurrent_task_claims(self, client, dependency_pipeline):
        """Concurrent task claims should not cause conflicts."""
        import threading
        
        # Register multiple workers
        workers = []
        for i in range(3):
            response = client.post(
                "/workers/register",
                json={"hostname": f"worker{i}.example.com"}
            )
            workers.append(response.json()["worker_id"])
        
        # Create more tasks
        from src.backend.api import _scheduler_state
        from .conftest import create_task_with_id
        
        pipeline_id = _scheduler_state.pipeline.id
        
        tool = ToolDefinition(
            name="tool2",
            container=ContainerConfig(image="test/image:latest", command="python main.py")
        )
        for i in range(3):
            task = create_task_with_id(
                task_id=f"task_{i}",
                tool=tool,
                task_type=TaskType.PRE_TRAINING,
                priority=100 - i,
                pipeline_id=pipeline_id
            )
            task.workflows = {"workflow_1"}
            _scheduler_state.scheduler._all_tasks.append(task)
        
        # Concurrently claim tasks
        results = []
        errors = []
        
        def claim_task(worker_id):
            try:
                response = client.post(f"/workers/{worker_id}/claim")
                results.append(response.json())
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=claim_task, args=(w,)) for w in workers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Should have no errors
        assert len(errors) == 0
        
        # Each worker should get a unique task (or None)
        claimed_tasks = [
            r.get("task", {}).get("id")
            for r in results
            if r.get("has_task")
        ]
        # No duplicate task assignments
        assert len(claimed_tasks) == len(set(claimed_tasks))
    
    def test_concurrent_status_updates_same_task(self, client, dependency_pipeline):
        """Concurrent status updates for same task should be handled."""
        import threading
        
        # Claim a task
        worker_response = client.post(
            "/workers/register",
            json={"hostname": "worker1.example.com"}
        )
        worker_id = worker_response.json()["worker_id"]
        
        claim_response = client.post(f"/workers/{worker_id}/claim")
        task_id = claim_response.json()["task"]["id"]
        
        # Try to update status concurrently
        results = []
        errors = []
        
        def update_status():
            try:
                response = client.put(
                    "/tasks/status",
                    json={"task_id": task_id, "status": "completed"}
                )
                results.append(response.status_code)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=update_status) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Should handle gracefully
        assert len(errors) == 0
        # At least one should succeed
        assert 200 in results
        
        # Task should be in a valid state
        task_response = client.get(f"/tasks/{task_id}")
        assert task_response.json()["status"] in ["completed", "running"]


# ============================================================================
# Test: Pipeline Detail Endpoint
# ============================================================================


class TestPipelineDetail:
    """Tests for pipeline detail endpoint."""
    
    def test_get_pipeline_detail_uninitialized(self, client):
        """Getting pipeline detail should fail when scheduler not initialized."""
        response = client.get("/pipeline")
        assert response.status_code == 503
    
    def test_get_pipeline_detail_success(self, client, dependency_pipeline):
        """Getting pipeline detail should return comprehensive information."""
        response = client.get("/pipeline")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "pipeline_1"
        assert data["name"] == "dependency_pipeline"
        assert data["workflow_count"] == 1
        assert data["task_count"] == 3
        assert "progress" in data
        assert "started_at" in data
        assert "running_time_seconds" in data
    
    def test_pipeline_detail_progress_tracking(self, client, dependency_pipeline):
        """Pipeline detail should track progress correctly."""
        # Complete one task
        client.get("/tasks/next")
        client.put("/tasks/status", json={"task_id": "task_a", "status": "completed"})
        
        response = client.get("/pipeline")
        assert response.status_code == 200
        data = response.json()
        assert data["progress"]["completed"] == 1
        assert data["progress"]["total"] == 3
        assert data["progress"]["progress_percent"] > 0
    
    def test_pipeline_detail_time_estimation(self, client, dependency_pipeline):
        """Pipeline detail should estimate remaining time."""
        # Complete a task with execution time
        client.get("/tasks/next")
        client.put(
            "/tasks/status",
            json={
                "task_id": "task_a",
                "status": "completed",
                "execution_time_ms": 5000
            }
        )
        
        response = client.get("/pipeline")
        assert response.status_code == 200
        data = response.json()
        # Should have estimated remaining time
        assert "estimated_remaining_seconds" in data


# ============================================================================
# Test: Task Priority Endpoint
# ============================================================================


class TestTaskPriorityEndpoint:
    """Tests for task priority information endpoint."""
    
    def test_get_task_priority_with_dependencies(self, client, dependency_pipeline):
        """Task priority should reflect dependency depth."""
        # Task A (no deps) should have higher priority than Task B
        response_a = client.get("/tasks/task_a/priority")
        response_b = client.get("/tasks/task_b/priority")
        
        assert response_a.status_code == 200
        assert response_b.status_code == 200
        
        priority_a = response_a.json()["priority"]
        priority_b = response_b.json()["priority"]
        
        assert priority_a > priority_b, \
            f"Task A (priority {priority_a}) should have higher priority than Task B (priority {priority_b})"
        
        # Dependency levels should be correct
        assert response_a.json()["dependency_level"] == 0
        assert response_b.json()["dependency_level"] == 1


# ============================================================================
# Test: Scheduler Reset and Reinitialization
# ============================================================================


class TestSchedulerReset:
    """Tests for scheduler reset functionality."""
    
    def test_reset_clears_task_metadata(self, client, dependency_pipeline):
        """Resetting scheduler should clear task metadata."""
        # Complete a task (creates metadata)
        client.get("/tasks/next")
        client.put(
            "/tasks/status",
            json={
                "task_id": "task_a",
                "status": "completed",
                "execution_time_ms": 1000
            }
        )
        
        # Verify metadata exists
        task_response = client.get("/tasks/task_a")
        assert task_response.json()["execution_time_ms"] == 1000
        
        # Reset scheduler
        client.post("/scheduler/reset")
        
        # Metadata should be cleared
        task_response = client.get("/tasks/task_a")
        assert task_response.json()["execution_time_ms"] is None
        assert task_response.json()["status"] == "pending"
    
    def test_reset_preserves_pipeline(self, client, dependency_pipeline):
        """Resetting scheduler should preserve pipeline structure."""
        # Reset
        response = client.post("/scheduler/reset")
        assert response.status_code == 200
        
        # Pipeline info should still be available
        response = client.get("/info/pipeline")
        assert response.status_code == 200
        assert response.json()["name"] == "dependency_pipeline"
        assert response.json()["task_count"] == 3


# ============================================================================
# Test: Task Filtering and Querying
# ============================================================================


class TestTaskFiltering:
    """Tests for task filtering and querying."""
    
    def test_filter_tasks_by_status_multiple(self, client, dependency_pipeline):
        """Filtering tasks by status should work with multiple statuses."""
        # Complete task A
        client.get("/tasks/next")
        client.put("/tasks/status", json={"task_id": "task_a", "status": "completed"})
        
        # Get completed tasks
        response = client.get("/tasks?status=completed")
        assert response.status_code == 200
        assert response.json()["total"] == 1
        assert response.json()["tasks"][0]["id"] == "task_a"
        
        # Get pending tasks
        response = client.get("/tasks?status=pending")
        assert response.status_code == 200
        assert response.json()["total"] == 2  # task_b and task_c
        
        # Get running tasks
        client.get("/tasks/next")  # Claims task_b, sets to running
        response = client.get("/tasks?status=running")
        assert response.status_code == 200
        assert response.json()["total"] == 1
        assert response.json()["tasks"][0]["id"] == "task_b"
    
    def test_case_insensitive_status_filter(self, client, dependency_pipeline):
        """Status filter should be case-insensitive or handle case correctly."""
        # Test various case combinations
        for status in ["PENDING", "pending", "Pending", "PeNdInG"]:
            response = client.get(f"/tasks?status={status}")
            # Should either work or return 400, not crash
            assert response.status_code in [200, 400]
            assert response.status_code != 500
