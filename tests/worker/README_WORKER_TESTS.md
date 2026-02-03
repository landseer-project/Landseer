# Comprehensive Worker Tests

Thorough test suite for the Landseer Worker covering all functionality, edge cases, and error conditions.

## Test Files

### 1. `test_worker_comprehensive.py` (60 tests)
Comprehensive tests for the main Worker class:
- **Worker Initialization** (6 tests)
  - Default and custom configuration
  - Workspace directory creation
  - Worker ID generation
  - Signal handler setup
  - Data path handling

- **Component Initialization** (5 tests)
  - Client creation
  - TaskRunner creation
  - Cache manager initialization
  - Two-level cache support
  - Cache disabling

- **Backend Communication** (10 tests)
  - Backend availability checking
  - Retry logic
  - Worker registration
  - Capability detection
  - Heartbeat functionality
  - Status reporting (idle/busy)

- **Dataset Fetching** (6 tests)
  - Manual data path usage
  - Backend dataset info retrieval
  - Local path usage
  - MinIO download
  - Missing dataset handling
  - Model script path setting

- **Task Execution** (10 tests)
  - Successful execution
  - Cache checking
  - Cache storage
  - Data directory mounting
  - Dependency output collection
  - Missing dependency handling
  - Task cleanup
  - Cache key computation

- **Result Reporting** (3 tests)
  - Success reporting
  - Failure reporting
  - Reporting error handling

- **Work Loop** (6 tests)
  - Task claiming and execution
  - No tasks handling
  - Completion detection
  - Heartbeat sending
  - Exception handling
  - Running flag respect

- **Cache Management** (5 tests)
  - Cache key computation
  - Two-level cache usage
  - Local cache fallback
  - Cache storage

- **Worker Start and Cleanup** (7 tests)
  - Component initialization
  - Backend waiting
  - Registration
  - Dataset fetching
  - Work loop entry
  - Client cleanup

- **Edge Cases** (8 tests)
  - Missing backend
  - Invalid workspace
  - Missing cache directory
  - Concurrent tasks
  - Signal handling
  - Zero timeout
  - Negative poll interval
  - Empty task config

### 2. `test_worker_client.py` (49 tests)
Comprehensive tests for LandseerClient:
- **Client Initialization** (4 tests)
  - Default and custom configuration
  - URL normalization
  - HTTP client creation

- **HTTP Request Handling** (6 tests)
  - Success handling
  - Server error retries
  - Client error handling (no retry)
  - Network error retries
  - Exponential backoff
  - Max retry failure

- **Health and Info** (7 tests)
  - Health check
  - Backend availability
  - Pipeline info
  - Progress retrieval
  - Dataset info
  - Error handling

- **Worker Registration** (4 tests)
  - Registration with ID
  - Auto-generated ID
  - Capability sending
  - Registration status

- **Heartbeat** (3 tests)
  - Successful heartbeat
  - Registration requirement
  - Failure handling

- **Task Management** (9 tests)
  - Task claiming
  - No task available
  - Registration requirement
  - Task completion reporting
  - Task failure reporting
  - Task retrieval
  - Task not found
  - Get all tasks
  - Status filtering

- **Tool Information** (3 tests)
  - Get all tools
  - Get specific tool
  - Tool not found

- **Cleanup** (2 tests)
  - Client close
  - Context manager

- **Edge Cases** (5 tests)
  - Minimal task data
  - Dependencies handling
  - Empty backend URL
  - Very long timeout
  - Zero retry attempts

### 3. Existing Test Files
- `test_container_security.py` - Container security and isolation tests
- `test_gpu_availability.py` - GPU assignment and availability tests
- `test_artifact_chaining.py` - Artifact chaining between tasks
- `test_task_execution.py` - Task execution and workspace isolation
- `test_workflow_execution.py` - Workflow execution tests

## Test Statistics

- **Total Tests**: 109+ tests (new comprehensive tests)
- **Coverage**: All major worker functionality
- **Edge Cases**: Extensive edge case coverage
- **Error Handling**: Comprehensive error condition testing

## Key Test Scenarios

### Worker Lifecycle
1. Initialization with various configurations
2. Component initialization (client, runner, cache)
3. Backend connection and registration
4. Dataset fetching (local, MinIO, manual)
5. Task claiming and execution loop
6. Result reporting
7. Cleanup and shutdown

### Task Execution Flow
1. Cache check before execution
2. Dependency output collection
3. Data directory mounting
4. Container execution
5. Result caching
6. Status reporting

### Error Handling
1. Backend unavailable
2. Registration failures
3. Task execution failures
4. Network errors with retries
5. Missing dependencies
6. Cache failures

### Edge Cases
1. Empty configurations
2. Invalid paths
3. Missing directories
4. Zero/negative timeouts
5. Very large values
6. Concurrent operations

## Running the Tests

```bash
# Run all worker tests
PYTHONPATH=. python -m pytest tests/worker/ -v

# Run comprehensive worker tests
PYTHONPATH=. python -m pytest tests/worker/test_worker_comprehensive.py -v

# Run client tests
PYTHONPATH=. python -m pytest tests/worker/test_worker_client.py -v

# Run with coverage
PYTHONPATH=. python -m pytest tests/worker/ --cov=src.worker --cov-report=html

# Run specific test class
PYTHONPATH=. python -m pytest tests/worker/test_worker_comprehensive.py::TestWorkerInitialization -v
```

## Test Coverage Areas

✅ Worker initialization and configuration
✅ Component initialization (client, runner, cache)
✅ Backend communication (registration, heartbeat, health checks)
✅ Dataset fetching (local, MinIO, manual paths)
✅ Task execution (claiming, execution, reporting)
✅ Cache management (local and two-level)
✅ Error handling and recovery
✅ Signal handling
✅ Work loop behavior
✅ Edge cases and boundary conditions
✅ HTTP client retry logic
✅ Task information parsing
✅ Result reporting (success and failure)

## Notes

- Tests use extensive mocking to isolate components
- Signal handling tests verify graceful shutdown
- Cache tests cover both local and two-level cache scenarios
- Error handling tests ensure graceful degradation
- Edge case tests ensure robustness under extreme conditions
- All tests are designed to run quickly without external dependencies
