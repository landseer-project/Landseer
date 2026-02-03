# Worker Adversarial Tests

This directory contains comprehensive adversarial tests for the Landseer worker module, focusing on security, correctness, and edge cases.

## Test Coverage

### 1. Container Security (`test_container_security.py`)

**Volume Mount Security:**
- ✅ Input directories mounted as read-only
- ✅ Output directories mounted as read-write
- ✅ Path traversal attacks blocked (e.g., `../../../etc/passwd`)
- ✅ All mount paths are absolute (prevents relative path attacks)
- ✅ Extra mounts default to read-only
- ✅ Workspace isolation between tasks

**GPU Assignment Security:**
- ✅ GPU ID correctly passed to container via `--gpus` flag
- ✅ `CUDA_VISIBLE_DEVICES` matches GPU ID
- ✅ No GPU flags when `gpu_id=None`
- ✅ GPU ID consistency between Docker flags and environment variables
- ✅ Apptainer uses `--nv` flag for GPU support

**Command Injection Prevention:**
- ✅ Commands split properly (prevents `; rm -rf /` attacks)
- ✅ `shell=False` used (prevents shell injection)
- ✅ Environment variable injection blocked
- ✅ Image name validation

**Environment Variable Security:**
- ✅ `INPUT_DIR` and `OUTPUT_DIR` set correctly
- ✅ Custom environment variables passed through
- ✅ Critical system variables not overridden by user vars
- ✅ `PYTHONPATH` includes `/input` for model script imports

**Model Script Mounting:**
- ✅ Model scripts mounted to `/app/` directory
- ✅ Model scripts mounted as read-only
- ✅ Path traversal in model script paths blocked

**Container Runtime:**
- ✅ Runtime detection works correctly
- ✅ Graceful handling when no runtime available
- ✅ Timeout enforcement

### 2. Task Execution (`test_task_execution.py`)

**Workspace Isolation:**
- ✅ Each task gets unique workspace directory
- ✅ Tasks cannot access other tasks' files
- ✅ Workspace cleanup preserves logs when requested
- ✅ Workspace cleanup can remove everything

**Input/Output Handling:**
- ✅ Input directories copied correctly
- ✅ Input files copied correctly
- ✅ Output directories created and writable
- ✅ Model scripts copied to input directory

**Environment Variable Handling:**
- ✅ `PYTHONPATH` includes `/input` for imports
- ✅ Task config added to environment
- ✅ Custom environment variables preserved

**Error Handling:**
- ✅ Missing container runtime handled gracefully
- ✅ Image pull failures handled
- ✅ Container execution failures reported correctly
- ✅ Exceptions during execution caught and reported
- ✅ Log files written even on failure

**Execution Result Accuracy:**
- ✅ Success results have all correct fields
- ✅ Failure results have all correct fields

### 3. Workflow Execution (`test_workflow_execution.py`)

**Task Dependency Enforcement:**
- ✅ Tasks require dependency artifacts to be present
- ✅ Tasks execute in dependency order
- ✅ Dependency order overrides priority when necessary

**Artifact Passing:**
- ✅ Output from one task becomes input for dependent task
- ✅ Artifacts isolated between workflows

**Workflow Failure Handling:**
- ✅ Task failures don't crash worker
- ✅ Failed task artifacts not used by dependents

**Cache Security:**
- ✅ Cache keys include task identity and dependencies
- ✅ Cache prevents reuse of poisoned artifacts

**Concurrent Execution Safety:**
- ✅ Concurrent tasks have separate workspaces
- ✅ Concurrent tasks don't share GPU conflicts

**Workflow Execution Order:**
- ✅ Tasks execute in priority order
- ✅ Dependency order overrides priority

## Security Considerations Tested

### Path Traversal Attacks
- Tests verify that paths like `../../../etc/passwd` cannot be used to escape workspace
- All mount paths are validated to be absolute
- Workspace directories are isolated

### Command Injection
- Commands are split into arguments (not executed as shell)
- `shell=False` prevents shell injection
- Environment variables are properly escaped

### GPU Assignment Validation
- GPU ID in `--gpus` flag matches `CUDA_VISIBLE_DEVICES`
- Prevents containers from accessing wrong GPU
- Verifies worker claims match actual GPU assignment

### Workspace Isolation
- Each task gets its own directory
- Tasks cannot access other tasks' files
- Concurrent execution is safe

### Cache Poisoning Prevention
- Cache keys include task identity and dependencies
- Prevents malicious tasks from using wrong cached artifacts

## Running the Tests

```bash
# Run all worker tests
PYTHONPATH=. python -m pytest tests/worker/ -v

# Run specific test file
PYTHONPATH=. python -m pytest tests/worker/test_container_security.py -v

# Run specific test class
PYTHONPATH=. python -m pytest tests/worker/test_container_security.py::TestGPUSecurity -v

# Run with coverage
PYTHONPATH=. python -m pytest tests/worker/ --cov=src.worker --cov-report=html
```

## Test Statistics

- **Total Tests**: 53
- **Container Security**: 20 tests
- **Task Execution**: 18 tests
- **Workflow Execution**: 15 tests

## Key Adversarial Scenarios Covered

1. **Path Traversal**: Attempts to access files outside workspace
2. **Command Injection**: Malicious commands in task parameters
3. **GPU Mismatch**: Worker claims GPU 0 but container gets GPU 1
4. **Workspace Escape**: Tasks accessing other tasks' files
5. **Cache Poisoning**: Using cached artifacts from different tasks
6. **Environment Variable Injection**: Overriding critical system variables
7. **Dependency Bypass**: Running tasks without dependencies
8. **Concurrent Conflicts**: Multiple tasks conflicting on resources

## Notes

- Most tests use mocking to avoid requiring actual Docker/Apptainer
- Some tests verify concepts that would be enforced by the scheduler
- Tests focus on worker-side security and correctness
- Integration with scheduler is tested in `tests/backend/scheduler/`
