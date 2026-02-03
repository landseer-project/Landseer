# Backend API Tests

Comprehensive test suite for the Landseer backend API, covering all endpoints, security, edge cases, and integration scenarios.

## Test Coverage

### Test Files

1. **`test_api_basic.py`** (45 tests)
   - Root and health endpoints
   - Pipeline info endpoints
   - Basic task management
   - Task status updates
   - Progress endpoints
   - Scheduler management
   - Worker management

2. **`test_api_security.py`** (20+ tests)
   - Input validation and sanitization
   - Path traversal prevention
   - SQL injection prevention
   - Command injection prevention
   - Malicious payload handling
   - Request method validation
   - Error handling
   - Concurrent request handling

3. **`test_api_workflows.py`** (20+ tests)
   - Workflow detail endpoints
   - Workflow results
   - Workflow status transitions
   - Workflow isolation
   - Workflow metrics

4. **`test_api_edge_cases.py`** (25+ tests)
   - Boundary conditions (empty pipeline, single task, large pipelines)
   - State consistency
   - Tool management
   - Dataset endpoints
   - Statistics endpoints
   - Error recovery
   - Priority edge cases
   - Task logs

5. **`test_api_integration.py`** (20+ tests)
   - End-to-end workflow execution
   - Multiple workers coordination
   - Race conditions
   - Pipeline detail endpoints
   - Task priority endpoints
   - Scheduler reset
   - Task filtering

## Test Statistics

- **Total Tests**: 135+ tests
- **Currently Passing**: ~90 tests
- **Coverage Areas**: All major API endpoints

## Endpoints Tested

### Health & Info
- ✅ `GET /` - Root endpoint
- ✅ `GET /health` - Health check
- ✅ `GET /info/pipeline` - Pipeline information
- ✅ `GET /info/workflows` - Workflow list

### Task Management
- ✅ `GET /tasks/next` - Get next task
- ✅ `PUT /tasks/status` - Update task status
- ✅ `GET /tasks` - List all tasks (with filtering)
- ✅ `GET /tasks/{task_id}` - Get task details
- ✅ `GET /tasks/{task_id}/logs` - Get task logs
- ✅ `GET /tasks/{task_id}/priority` - Get priority info

### Progress & Statistics
- ✅ `GET /progress` - Pipeline progress
- ✅ `GET /progress/levels` - Priority levels
- ✅ `GET /progress/ready` - Ready tasks
- ✅ `GET /progress/blocked` - Blocked tasks

### Scheduler Management
- ✅ `POST /scheduler/initialize` - Initialize scheduler
- ✅ `POST /scheduler/reset` - Reset scheduler
- ✅ `GET /scheduler/status` - Scheduler status
- ✅ `GET /scheduler/next` - Preview next task

### Worker Management
- ✅ `POST /workers/register` - Register worker
- ✅ `GET /workers` - List workers
- ✅ `GET /workers/{worker_id}` - Get worker info
- ✅ `POST /workers/{worker_id}/heartbeat` - Worker heartbeat
- ✅ `GET /workers/{worker_id}/task` - Get worker's current task
- ✅ `POST /workers/{worker_id}/claim` - Claim task for worker

### Workflow Endpoints
- ✅ `GET /workflows/{workflow_id}` - Workflow details
- ✅ `GET /workflows/{workflow_id}/results` - Workflow results
- ✅ `GET /workflows/{workflow_id}/metrics` - Workflow metrics

### Tool Management
- ✅ `GET /tools` - List tools
- ✅ `GET /tools/{tool_name}` - Get tool info
- ✅ `POST /tools` - Add tool

### Pipeline & Dataset
- ✅ `GET /pipeline` - Pipeline details
- ✅ `GET /dataset` - Dataset information
- ✅ `GET /dataset/download-url` - Dataset download URL

### Statistics
- ✅ `GET /stats/database` - Database stats
- ✅ `GET /stats/store` - Store stats
- ✅ `GET /stats/system` - System stats

### Metrics
- ✅ `GET /pipelines/{pipeline_id}/metrics` - Pipeline metrics
- ✅ `GET /workflows/{workflow_id}/metrics` - Workflow metrics

## Security Tests

### Input Validation
- ✅ Malicious status values rejected
- ✅ Path traversal attempts blocked
- ✅ SQL injection attempts handled safely
- ✅ Command injection prevented
- ✅ Large payloads handled gracefully
- ✅ Unicode and special characters handled

### Request Method Validation
- ✅ Wrong HTTP methods return 405
- ✅ OPTIONS method supported (CORS)

### Error Handling
- ✅ Malformed JSON returns 422
- ✅ Missing Content-Type handled
- ✅ Empty request body validated
- ✅ Null values handled correctly
- ✅ Very long strings handled
- ✅ Negative numbers validated
- ✅ Extremely large numbers handled

### Concurrent Requests
- ✅ Concurrent task claims don't conflict
- ✅ Concurrent status updates handled safely

## Edge Cases Tested

### Boundary Conditions
- ✅ Empty pipeline
- ✅ Single task pipeline
- ✅ Large number of tasks (100+)
- ✅ Tasks with many dependencies (10+)

### State Consistency
- ✅ Task status consistent across endpoints
- ✅ Worker task assignment consistent
- ✅ Progress consistent after updates

### Error Recovery
- ✅ Recover from failed tasks
- ✅ Handle invalid state transitions
- ✅ Handle missing task metadata

## Integration Scenarios

### End-to-End Workflows
- ✅ Complete task lifecycle (claim -> execute -> complete)
- ✅ Dependency chain execution
- ✅ Failed task blocks dependents

### Multiple Workers
- ✅ Multiple workers claim different tasks
- ✅ Worker heartbeat tracking
- ✅ Worker task completion tracking

### Race Conditions
- ✅ Concurrent task claims
- ✅ Concurrent status updates for same task

## Known Issues

Some tests are currently failing due to:
1. **Task Reuse**: Tasks are being reused across test runs, causing "Task already belongs to pipeline" errors
   - **Fix**: Use unique task IDs or ensure proper task isolation in fixtures
2. **Priority Levels Format**: Priority levels endpoint returns string keys instead of int keys
   - **Fix**: Update test expectations or fix endpoint response format
3. **Worker Assignment**: Worker ID not always stored in task metadata
   - **Fix**: Ensure worker assignment updates task metadata correctly

## Running the Tests

```bash
# Run all API tests
PYTHONPATH=. python -m pytest tests/backend/api/ -v

# Run specific test file
PYTHONPATH=. python -m pytest tests/backend/api/test_api_basic.py -v

# Run specific test class
PYTHONPATH=. python -m pytest tests/backend/api/test_api_basic.py::TestTaskManagementBasic -v

# Run with coverage
PYTHONPATH=. python -m pytest tests/backend/api/ --cov=src.backend.api --cov-report=html
```

## Test Coverage Gaps

Areas that need additional testing:

1. **Database Integration**: Tests with actual database (currently mocked)
2. **MinIO Store Integration**: Tests with actual MinIO connection
3. **Authentication/Authorization**: If added in future
4. **Rate Limiting**: If implemented
5. **WebSocket Endpoints**: If added for real-time updates
6. **Bulk Operations**: Batch task updates, bulk worker registration
7. **Pagination**: If task/workflow lists get large
8. **Filtering/Sorting**: Advanced query parameters
9. **Export/Import**: Pipeline configuration export/import
10. **Metrics Aggregation**: Complex metric calculations

## Adversarial Test Scenarios Covered

1. **Path Traversal**: `../../../etc/passwd` in IDs
2. **SQL Injection**: `'; DROP TABLE tasks; --` in parameters
3. **XSS**: `<script>alert('xss')</script>` in inputs
4. **Command Injection**: `python main.py; rm -rf /` in commands
5. **Large Payloads**: 1MB+ JSON payloads
6. **Concurrent Attacks**: Multiple workers claiming same task
7. **State Manipulation**: Invalid status transitions
8. **Resource Exhaustion**: 100+ tasks, many dependencies
