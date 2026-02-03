"""
Security and input validation tests for the backend API.

Tests for:
- Input validation and sanitization
- SQL injection prevention (in query parameters)
- Path traversal prevention
- Malicious payload handling
- Rate limiting considerations
- Authentication/authorization (if applicable)
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

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
# Test: Input Validation
# ============================================================================


class TestInputValidation:
    """Tests for input validation and sanitization."""
    
    def test_task_status_update_malicious_status(self, client, initialized_pipeline):
        """Malicious status values should be rejected."""
        client.get("/tasks/next")
        
        malicious_statuses = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE tasks; --",
            "../../etc/passwd",
            "completed'; DELETE FROM tasks; --",
            "\x00\x01\x02",  # Null bytes
            "completed\nfailed",  # Newline injection
        ]
        
        for malicious_status in malicious_statuses:
            response = client.put(
                "/tasks/status",
                json={
                    "task_id": "task_1",
                    "status": malicious_status
                }
            )
            # Should reject invalid status
            assert response.status_code in [400, 422], \
                f"Should reject malicious status: {malicious_status}"
    
    def test_task_id_path_traversal(self, client, initialized_pipeline):
        """Path traversal attempts in task IDs should be handled safely."""
        malicious_ids = [
            "../../etc/passwd",
            "..\\..\\windows\\system32",
            "task_1/../task_2",
            "task_1%2F..%2Ftask_2",  # URL encoded
            "/etc/passwd",
            "C:\\Windows\\System32",
        ]
        
        for malicious_id in malicious_ids:
            response = client.get(f"/tasks/{malicious_id}")
            # Should return 404 (not found) or 400 (bad request), not 500 (server error)
            assert response.status_code in [404, 400, 422], \
                f"Should handle path traversal safely: {malicious_id}"
            assert response.status_code != 500, \
                f"Should not cause server error: {malicious_id}"
    
    def test_worker_id_injection(self, client):
        """Worker ID injection attempts should be handled safely."""
        malicious_ids = [
            "worker_1'; DROP TABLE workers; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "worker_1\nworker_2",
            "\x00worker_1",
        ]
        
        for malicious_id in malicious_ids:
            # Try to register with malicious ID
            response = client.post(
                "/workers/register",
                json={
                    "worker_id": malicious_id,
                    "hostname": "test.example.com"
                }
            )
            # Should either accept (if sanitized) or reject, but not crash
            assert response.status_code in [200, 400, 422], \
                f"Should handle malicious worker ID: {malicious_id}"
            assert response.status_code != 500
    
    def test_query_parameter_injection(self, client, initialized_pipeline):
        """Query parameter injection attempts should be handled safely."""
        malicious_params = [
            "status=completed'; DROP TABLE tasks; --",
            "status=completed&status=failed",
            "status=../../etc/passwd",
            "status=<script>alert('xss')</script>",
        ]
        
        for param in malicious_params:
            response = client.get(f"/tasks?{param}")
            # Should either filter correctly or return 400, not crash
            assert response.status_code in [200, 400, 422], \
                f"Should handle malicious query param: {param}"
            assert response.status_code != 500
    
    def test_json_payload_validation(self, client, initialized_pipeline):
        """Invalid JSON payloads should be rejected."""
        # Missing required fields
        response = client.put(
            "/tasks/status",
            json={"task_id": "task_1"}  # Missing status
        )
        assert response.status_code == 422  # Validation error
        
        # Wrong types
        response = client.put(
            "/tasks/status",
            json={
                "task_id": 123,  # Should be string
                "status": "completed"
            }
        )
        assert response.status_code == 422
        
        # Extra unexpected fields (should be ignored or validated)
        response = client.put(
            "/tasks/status",
            json={
                "task_id": "task_1",
                "status": "completed",
                "malicious_field": "<script>alert('xss')</script>"
            }
        )
        # Should accept valid fields and ignore extra ones
        assert response.status_code in [200, 422]
    
    def test_large_payload_handling(self, client):
        """Large payloads should be handled gracefully."""
        # Very large worker capabilities
        large_capabilities = {
            "data": "x" * 1000000  # 1MB of data
        }
        
        response = client.post(
            "/workers/register",
            json={
                "hostname": "test.example.com",
                "capabilities": large_capabilities
            }
        )
        # Should either accept or reject with appropriate error, not crash
        assert response.status_code in [200, 400, 413, 422]
        assert response.status_code != 500
    
    def test_unicode_and_special_characters(self, client, initialized_pipeline):
        """Unicode and special characters should be handled correctly."""
        # Unicode in task config
        client.get("/tasks/next")
        
        response = client.put(
            "/tasks/status",
            json={
                "task_id": "task_1",
                "status": "completed",
                "error_message": "错误: 任务失败 🚀"
            }
        )
        assert response.status_code == 200
        
        # Verify unicode was stored correctly
        task_response = client.get("/tasks/task_1")
        assert "错误" in task_response.json()["error_message"] or \
               task_response.json()["error_message"] == "错误: 任务失败 🚀"


# ============================================================================
# Test: Request Method Validation
# ============================================================================


class TestRequestMethodValidation:
    """Tests for HTTP method validation."""
    
    def test_wrong_method_on_endpoints(self, client, initialized_pipeline):
        """Using wrong HTTP methods should return 405 Method Not Allowed."""
        # GET on PUT endpoint
        response = client.get("/tasks/status")
        assert response.status_code == 405
        
        # PUT on GET endpoint
        response = client.put("/tasks")
        assert response.status_code == 405
        
        # DELETE on endpoints that don't support it
        response = client.delete("/tasks/task_1")
        assert response.status_code == 405
    
    def test_options_method_supported(self, client):
        """OPTIONS method should be supported (CORS preflight)."""
        response = client.options("/tasks/next")
        assert response.status_code in [200, 204]


# ============================================================================
# Test: Error Handling and Edge Cases
# ============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_malformed_json(self, client):
        """Malformed JSON should return 422."""
        response = client.put(
            "/tasks/status",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_content_type(self, client):
        """Missing Content-Type header should be handled."""
        response = client.put(
            "/tasks/status",
            data='{"task_id": "task_1", "status": "completed"}'
        )
        # FastAPI might auto-detect JSON or return 422
        assert response.status_code in [200, 422]
    
    def test_empty_request_body(self, client):
        """Empty request body should return validation error."""
        response = client.put(
            "/tasks/status",
            json={}
        )
        assert response.status_code == 422
    
    def test_null_values_in_request(self, client, initialized_pipeline):
        """Null values should be handled according to schema."""
        client.get("/tasks/next")
        
        # Null status should be rejected
        response = client.put(
            "/tasks/status",
            json={
                "task_id": "task_1",
                "status": None
            }
        )
        assert response.status_code == 422
        
        # Null optional fields should be accepted
        response = client.put(
            "/tasks/status",
            json={
                "task_id": "task_1",
                "status": "completed",
                "error_message": None,
                "execution_time_ms": None
            }
        )
        assert response.status_code == 200
    
    def test_very_long_strings(self, client, initialized_pipeline):
        """Very long strings should be handled gracefully."""
        client.get("/tasks/next")
        
        long_error = "x" * 100000  # 100KB error message
        
        response = client.put(
            "/tasks/status",
            json={
                "task_id": "task_1",
                "status": "failed",
                "error_message": long_error
            }
        )
        # Should either accept or reject with appropriate limit
        assert response.status_code in [200, 400, 413]
        assert response.status_code != 500
    
    def test_negative_numbers(self, client, initialized_pipeline):
        """Negative numbers where not expected should be validated."""
        client.get("/tasks/next")
        
        # Negative execution time might be acceptable or rejected
        response = client.put(
            "/tasks/status",
            json={
                "task_id": "task_1",
                "status": "completed",
                "execution_time_ms": -1000
            }
        )
        # Should either accept (if negative is valid) or reject
        assert response.status_code in [200, 400, 422]
    
    def test_extremely_large_numbers(self, client, initialized_pipeline):
        """Extremely large numbers should be handled."""
        client.get("/tasks/next")
        
        response = client.put(
            "/tasks/status",
            json={
                "task_id": "task_1",
                "status": "completed",
                "execution_time_ms": 2**63  # Very large number
            }
        )
        # Should handle gracefully (might overflow or be rejected)
        assert response.status_code in [200, 400, 422, 500]


# ============================================================================
# Test: Concurrent Request Handling
# ============================================================================


class TestConcurrentRequests:
    """Tests for handling concurrent requests."""
    
    def test_concurrent_task_claims(self, client, initialized_pipeline):
        """Multiple workers claiming tasks concurrently should not cause conflicts."""
        import threading
        
        # Register multiple workers
        workers = []
        for i in range(3):
            response = client.post(
                "/workers/register",
                json={"hostname": f"worker{i}.example.com"}
            )
            workers.append(response.json()["worker_id"])
        
        # Create multiple tasks
        from src.backend.api import _scheduler_state
        tool = ToolDefinition(
            name="tool2",
            container=ContainerConfig(image="test/image:latest", command="python main.py")
        )
        for i in range(2, 5):
            task = Task.__new__(Task)
            task.id = f"task_{i}"
            task.tool = tool
            task.config = {}
            task.priority = 100 - i
            task.status = TaskStatus.PENDING
            task.task_type = TaskType.PRE_TRAINING
            task.counter = 1
            task.workflows = {"workflow_1"}
            task.pipeline_id = "pipeline_1"
            task.dependencies = []
            _scheduler_state.scheduler._all_tasks.append(task)
            _scheduler_state.pipeline.workflows[0].tasks.append(task)
        
        # Concurrently claim tasks
        results = []
        errors = []
        
        def claim_task(worker_id):
            try:
                response = client.post(f"/workers/{worker_id}/claim")
                results.append(response.json())
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=claim_task, args=(worker_id,))
            for worker_id in workers
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have no errors
        assert len(errors) == 0
        
        # Each worker should get a different task (or None if no more tasks)
        claimed_tasks = [r.get("task", {}).get("id") for r in results if r.get("has_task")]
        # Tasks should be unique (no duplicates)
        assert len(claimed_tasks) == len(set(claimed_tasks))
    
    def test_concurrent_status_updates(self, client, initialized_pipeline):
        """Concurrent status updates should not cause race conditions."""
        import threading
        
        # Get a task
        client.get("/tasks/next")
        
        # Try to update status concurrently
        results = []
        errors = []
        
        def update_status():
            try:
                response = client.put(
                    "/tasks/status",
                    json={
                        "task_id": "task_1",
                        "status": "completed"
                    }
                )
                results.append(response.status_code)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=update_status) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should handle gracefully (might have some 400s for invalid transitions)
        assert len(errors) == 0
        # At least one should succeed
        assert 200 in results


# ============================================================================
# Test: Workflow Endpoints Security
# ============================================================================


class TestWorkflowEndpointsSecurity:
    """Security tests for workflow endpoints."""
    
    def test_workflow_id_path_traversal(self, client, initialized_pipeline):
        """Path traversal in workflow IDs should be handled safely."""
        malicious_ids = [
            "../../etc/passwd",
            "..\\..\\windows",
            "workflow_1/../workflow_2",
            "/etc/passwd",
        ]
        
        for malicious_id in malicious_ids:
            response = client.get(f"/workflows/{malicious_id}")
            # Should return 404, not expose files
            assert response.status_code in [404, 400, 422]
            assert response.status_code != 500
    
    def test_workflow_id_sql_injection(self, client, initialized_pipeline):
        """SQL injection attempts in workflow IDs should be handled safely."""
        malicious_ids = [
            "workflow_1'; DROP TABLE workflows; --",
            "workflow_1' OR '1'='1",
            "workflow_1'; SELECT * FROM users; --",
        ]
        
        for malicious_id in malicious_ids:
            response = client.get(f"/workflows/{malicious_id}")
            # Should return 404, not execute SQL
            assert response.status_code in [404, 400, 422]
            assert response.status_code != 500


# ============================================================================
# Test: Tool Management Security
# ============================================================================


class TestToolManagementSecurity:
    """Security tests for tool management endpoints."""
    
    def test_add_tool_malicious_input(self, client):
        """Adding tools with malicious input should be sanitized or rejected."""
        malicious_inputs = [
            {
                "name": "<script>alert('xss')</script>",
                "image": "test/image:latest",
                "command": "python main.py"
            },
            {
                "name": "tool_1'; DROP TABLE tools; --",
                "image": "test/image:latest",
                "command": "python main.py"
            },
            {
                "name": "normal_tool",
                "image": "test/image:latest",
                "command": "python main.py; rm -rf /"
            },
        ]
        
        for malicious_input in malicious_inputs:
            response = client.post("/tools", json=malicious_input)
            # Should either sanitize or reject
            assert response.status_code in [200, 400, 422]
            assert response.status_code != 500
    
    def test_tool_name_path_traversal(self, client):
        """Path traversal in tool names should be handled safely."""
        malicious_names = [
            "../../etc/passwd",
            "..\\..\\windows",
            "/etc/passwd",
        ]
        
        for malicious_name in malicious_names:
            response = client.get(f"/tools/{malicious_name}")
            # Should return 404, not expose files
            assert response.status_code in [404, 400, 422]
            assert response.status_code != 500
