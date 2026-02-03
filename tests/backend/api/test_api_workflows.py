"""
Tests for workflow-related API endpoints.

Tests for:
- Workflow detail endpoints
- Workflow results
- Workflow status tracking
- Workflow metrics
"""

import pytest
from fastapi.testclient import TestClient

from src.backend.api import app
from src.pipeline.tasks import Task, TaskStatus, TaskType, clear_task_registry
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
def multi_workflow_pipeline():
    """Create a pipeline with multiple workflows."""
    clear_task_registry()
    
    tool1 = ToolDefinition(
        name="tool1",
        container=ContainerConfig(image="test/image:latest", command="python main.py")
    )
    tool2 = ToolDefinition(
        name="tool2",
        container=ContainerConfig(image="test/image:latest", command="python main.py")
    )
    
    from .conftest import create_task_with_id
    
    # Workflow 1: tool1 -> tool2
    task1_wf1 = create_task_with_id(
        task_id="task_1_wf1",
        tool=tool1,
        task_type=TaskType.PRE_TRAINING,
        priority=100
    )
    task1_wf1.workflows = {"workflow_1"}
    task1_wf1.pipeline_id = "pipeline_1"
    
    task2_wf1 = create_task_with_id(
        task_id="task_2_wf1",
        tool=tool2,
        task_type=TaskType.POST_TRAINING,
        dependencies=[task1_wf1],
        priority=90
    )
    task2_wf1.workflows = {"workflow_1"}
    task2_wf1.pipeline_id = "pipeline_1"
    
    workflow1 = WorkflowFactory.create_workflow(
        name="workflow_1",
        tasks=[task1_wf1, task2_wf1]
    )
    
    # Workflow 2: tool1 only
    task1_wf2 = create_task_with_id(
        task_id="task_1_wf2",
        tool=tool1,
        task_type=TaskType.PRE_TRAINING,
        priority=100
    )
    task1_wf2.workflows = {"workflow_2"}
    task1_wf2.pipeline_id = "pipeline_1"
    
    workflow2 = WorkflowFactory.create_workflow(
        name="workflow_2",
        tasks=[task1_wf2]
    )
    
    pipeline = DefenseEvaluationPipeline(
        name="test_pipeline",
        workflows=[workflow1, workflow2]
    )
    pipeline.id = pipeline_id
    
    from src.backend.api import _scheduler_state
    _scheduler_state.initialize(pipeline)
    
    return pipeline


# ============================================================================
# Test: Workflow Detail Endpoints
# ============================================================================


class TestWorkflowDetail:
    """Tests for workflow detail endpoints."""
    
    def test_get_workflow_detail_uninitialized(self, client):
        """Getting workflow detail should fail when scheduler not initialized."""
        response = client.get("/workflows/workflow_1")
        assert response.status_code == 503
    
    def test_get_workflow_detail_not_found(self, client, multi_workflow_pipeline):
        """Getting non-existent workflow should return 404."""
        response = client.get("/workflows/nonexistent_workflow")
        assert response.status_code == 404
    
    def test_get_workflow_detail_by_id(self, client, multi_workflow_pipeline):
        """Getting workflow by ID should return workflow details."""
        response = client.get("/workflows/workflow_1")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "workflow_1"
        assert data["name"] == "workflow_1"
        assert data["task_count"] == 2
        assert len(data["tasks"]) == 2
        assert data["status"] == "pending"
        assert data["completed_tasks"] == 0
        assert data["failed_tasks"] == 0
    
    def test_get_workflow_detail_by_name(self, client, multi_workflow_pipeline):
        """Getting workflow by name should also work."""
        response = client.get("/workflows/workflow_1")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "workflow_1"
    
    def test_get_workflow_detail_status_running(self, client, multi_workflow_pipeline):
        """Workflow status should reflect running tasks."""
        # Start a task
        client.get("/tasks/next")
        
        response = client.get("/workflows/workflow_1")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
    
    def test_get_workflow_detail_status_completed(self, client, multi_workflow_pipeline):
        """Workflow status should reflect completed state."""
        # Complete all tasks in workflow 1
        client.get("/tasks/next")  # task_1_wf1
        client.put("/tasks/status", json={"task_id": "task_1_wf1", "status": "completed"})
        
        client.get("/tasks/next")  # task_2_wf1
        client.put("/tasks/status", json={"task_id": "task_2_wf1", "status": "completed"})
        
        response = client.get("/workflows/workflow_1")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["completed_tasks"] == 2
        assert data["failed_tasks"] == 0
    
    def test_get_workflow_detail_status_failed(self, client, multi_workflow_pipeline):
        """Workflow status should reflect failed state."""
        # Fail a task
        client.get("/tasks/next")
        client.put(
            "/tasks/status",
            json={
                "task_id": "task_1_wf1",
                "status": "failed",
                "error_message": "Task failed"
            }
        )
        
        response = client.get("/workflows/workflow_1")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["failed_tasks"] == 1
        assert len(data["failure_reasons"]) == 1
        assert data["failure_reasons"][0]["task_id"] == "task_1_wf1"
        assert "Task failed" in data["failure_reasons"][0]["error"]
    
    def test_get_workflow_results_uninitialized(self, client):
        """Getting workflow results should fail when scheduler not initialized."""
        response = client.get("/workflows/workflow_1/results")
        assert response.status_code == 503
    
    def test_get_workflow_results_not_found(self, client, multi_workflow_pipeline):
        """Getting results for non-existent workflow should return 404."""
        response = client.get("/workflows/nonexistent_workflow/results")
        assert response.status_code == 404
    
    def test_get_workflow_results_empty(self, client, multi_workflow_pipeline):
        """Getting workflow results should return empty results for pending tasks."""
        response = client.get("/workflows/workflow_1/results")
        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == "workflow_1"
        assert len(data["results"]) == 2
        assert all(r["status"] == "pending" for r in data["results"])
    
    def test_get_workflow_results_with_execution_data(self, client, multi_workflow_pipeline):
        """Workflow results should include execution metadata."""
        # Complete a task with metadata
        client.get("/tasks/next")
        client.put(
            "/tasks/status",
            json={
                "task_id": "task_1_wf1",
                "status": "completed",
                "execution_time_ms": 5000,
                "result": {"output": "success"}
            }
        )
        
        response = client.get("/workflows/workflow_1/results")
        assert response.status_code == 200
        data = response.json()
        
        # Find the completed task result
        completed_result = next(r for r in data["results"] if r["task_id"] == "task_1_wf1")
        assert completed_result["status"] == "completed"
        assert completed_result["execution_time_ms"] == 5000
        assert completed_result["result"]["output"] == "success"


# ============================================================================
# Test: Workflow Status Transitions
# ============================================================================


class TestWorkflowStatusTransitions:
    """Tests for workflow status transitions."""
    
    def test_workflow_status_pending_to_running(self, client, multi_workflow_pipeline):
        """Workflow should transition from pending to running when task starts."""
        response = client.get("/workflows/workflow_1")
        assert response.json()["status"] == "pending"
        
        # Start a task
        client.get("/tasks/next")
        
        response = client.get("/workflows/workflow_1")
        assert response.json()["status"] == "running"
    
    def test_workflow_status_running_to_completed(self, client, multi_workflow_pipeline):
        """Workflow should transition to completed when all tasks done."""
        # Complete all tasks
        client.get("/tasks/next")
        client.put("/tasks/status", json={"task_id": "task_1_wf1", "status": "completed"})
        
        client.get("/tasks/next")
        client.put("/tasks/status", json={"task_id": "task_2_wf1", "status": "completed"})
        
        response = client.get("/workflows/workflow_1")
        assert response.json()["status"] == "completed"
    
    def test_workflow_status_running_to_failed(self, client, multi_workflow_pipeline):
        """Workflow should transition to failed when any task fails."""
        # Fail a task
        client.get("/tasks/next")
        client.put(
            "/tasks/status",
            json={"task_id": "task_1_wf1", "status": "failed", "error_message": "Error"}
        )
        
        response = client.get("/workflows/workflow_1")
        assert response.json()["status"] == "failed"
    
    def test_workflow_status_partial_completion(self, client, multi_workflow_pipeline):
        """Workflow with partial completion should show correct counts."""
        # Complete one task, leave one pending
        client.get("/tasks/next")
        client.put("/tasks/status", json={"task_id": "task_1_wf1", "status": "completed"})
        
        response = client.get("/workflows/workflow_1")
        assert response.json()["status"] == "running"
        assert response.json()["completed_tasks"] == 1
        assert response.json()["failed_tasks"] == 0


# ============================================================================
# Test: Workflow Isolation
# ============================================================================


class TestWorkflowIsolation:
    """Tests for workflow isolation and independence."""
    
    def test_workflows_independent_status(self, client, multi_workflow_pipeline):
        """Workflows should have independent status."""
        # Complete workflow 1
        client.get("/tasks/next")  # task_1_wf1
        client.put("/tasks/status", json={"task_id": "task_1_wf1", "status": "completed"})
        client.get("/tasks/next")  # task_2_wf1
        client.put("/tasks/status", json={"task_id": "task_2_wf1", "status": "completed"})
        
        # Workflow 2 should still be pending
        response = client.get("/workflows/workflow_2")
        assert response.json()["status"] == "pending"
        
        # Workflow 1 should be completed
        response = client.get("/workflows/workflow_1")
        assert response.json()["status"] == "completed"
    
    def test_workflow_failure_does_not_affect_others(self, client, multi_workflow_pipeline):
        """Failure in one workflow should not affect others."""
        # Fail a task in workflow 1
        client.get("/tasks/next")
        client.put(
            "/tasks/status",
            json={"task_id": "task_1_wf1", "status": "failed", "error_message": "Error"}
        )
        
        # Workflow 1 should be failed
        response = client.get("/workflows/workflow_1")
        assert response.json()["status"] == "failed"
        
        # Workflow 2 should still be pending
        response = client.get("/workflows/workflow_2")
        assert response.json()["status"] == "pending"


# ============================================================================
# Test: Workflow Metrics
# ============================================================================


class TestWorkflowMetrics:
    """Tests for workflow metrics endpoints."""
    
    def test_get_workflow_metrics_uninitialized(self, client):
        """Getting workflow metrics should fail when scheduler not initialized."""
        response = client.get("/workflows/workflow_1/metrics")
        assert response.status_code == 503
    
    def test_get_workflow_metrics_not_found(self, client, multi_workflow_pipeline):
        """Getting metrics for non-existent workflow should return 404."""
        response = client.get("/workflows/nonexistent_workflow/metrics")
        assert response.status_code == 404
    
    def test_get_workflow_metrics_empty(self, client, multi_workflow_pipeline):
        """Getting metrics for workflow without metrics should return empty."""
        response = client.get("/workflows/workflow_1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == "workflow_1"
        assert data["metrics"] == {}
        assert data["evaluators_run"] == []
        assert data["evaluators_skipped"] == []
    
    def test_get_pipeline_metrics_uninitialized(self, client):
        """Getting pipeline metrics should fail when scheduler not initialized."""
        response = client.get("/pipelines/pipeline_1/metrics")
        assert response.status_code == 503
    
    def test_get_pipeline_metrics_not_found(self, client, multi_workflow_pipeline):
        """Getting metrics for non-existent pipeline should return 404."""
        response = client.get("/pipelines/nonexistent_pipeline/metrics")
        assert response.status_code == 404
    
    def test_get_pipeline_metrics_success(self, client, multi_workflow_pipeline):
        """Getting pipeline metrics should return metrics for all workflows."""
        response = client.get("/pipelines/pipeline_1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["pipeline_id"] == "pipeline_1"
        assert data["pipeline_name"] == "test_pipeline"
        assert data["workflow_count"] == 2
        assert len(data["workflows"]) == 2
        assert data["metric_names"] == []
        assert data["summary"] == {}
