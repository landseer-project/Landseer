# Workflow Specification Tests

Comprehensive test suite for workflow functionality based on `docs/Workflow.md`.

## Test Coverage

### Test Files

1. **`test_workflow_specification.py`** (36 tests)
   - Workflow permutation logic
   - Single during_training tool constraint
   - Baseline tool substitution
   - Execution order
   - Artifact chaining basics
   - File format requirements
   - Converter tool invocation (conceptual)
   - Evaluation artifact selection (conceptual)
   - Edge cases
   - Complex scenarios

2. **`test_workflow_artifact_chaining.py`** (22 tests)
   - Dataset chaining through stages
   - Model chaining through stages
   - File format enforcement (.npy, .pt)
   - Complex artifact chaining scenarios
   - Evaluation artifact selection
   - Edge cases in artifact chaining

3. **`test_workflow_edge_cases.py`** (20+ tests)
   - Extreme scenarios (many tools, no tools)
   - Boundary conditions
   - Error conditions
   - Permutation edge cases
   - Task deduplication edge cases

## Test Statistics

- **Total Tests**: 78+ tests
- **Coverage**: All major requirements from Workflow.md

## Requirements Tested

### 1. Workflow Permutation ✅
- ✅ Order matters: tool1->tool2 != tool2->tool1
- ✅ Permutations of tool subsets
- ✅ Baseline substitution for empty sets
- ✅ Multiple tools create all permutations

### 2. Single During Training Tool ✅
- ✅ Each workflow has exactly 1 during_training tool
- ✅ No permutations of multiple during_training tools
- ✅ Multiple during_training tools create separate workflows

### 3. Execution Order ✅
- ✅ Pre-training → during_training → post_training → deployment
- ✅ Within-stage order preserved
- ✅ Dependencies set correctly

### 4. Artifact Chaining ✅
- ✅ Outputs become inputs for next tool
- ✅ First pre tool takes config data
- ✅ Dataset passed through chain
- ✅ Model passed from during_training to post_training
- ✅ Deployment gets both model and dataset

### 5. File Format Requirements ✅
- ✅ Dataset files must be .npy format
- ✅ Model files must be .pt format (PyTorch)
- ✅ Format enforcement tested

### 6. Converter Tool ✅ (Conceptual)
- ✅ Converter needed when input not PyTorch
- ✅ Converter needed when output not PyTorch
- ✅ TensorFlow ↔ PyTorch conversion

### 7. Evaluation Artifact Selection ✅ (Conceptual)
- ✅ Uses dataset from last tool that modified dataset
- ✅ Uses model from last tool that modified model
- ✅ Handles cases where later tools modify artifacts

### 8. Edge Cases ✅
- ✅ Empty stages
- ✅ No baseline tools
- ✅ All baseline workflow
- ✅ Many tools in stage
- ✅ Missing dependencies
- ✅ Large files
- ✅ Nested directories
- ✅ Multiple dependencies
- ✅ Failed tasks

## Example Test Scenarios

### Workflow.md Example
```
Pipeline:
  pre_training: A (baseline), B (actual)
  during_training: C (baseline), D (actual)
  post_training: E (baseline)
  deployment: G (baseline)

Expected: 4 workflows (after baseline deduplication)
- A->C->E->G
- B->C->E->G
- A->D->E->G
- B->D->E->G
```

### B2 Scenario
```
pre_training: A (baseline), B, B2 (actual)

Expected: 5 pre_training options
- (B, B2)
- (B2, B)
- (B)
- (B2)
- (A) - baseline
```

## Running the Tests

```bash
# Run all workflow tests
PYTHONPATH=. python -m pytest tests/pipeline/test_workflow_specification.py tests/pipeline/test_workflow_artifact_chaining.py tests/pipeline/test_workflow_edge_cases.py -v

# Run specific test class
PYTHONPATH=. python -m pytest tests/pipeline/test_workflow_specification.py::TestWorkflowPermutation -v

# Run with coverage
PYTHONPATH=. python -m pytest tests/pipeline/ --cov=src.pipeline --cov-report=html
```

## Test Coverage Gaps

Areas that may need additional testing (depending on implementation):

1. **Converter Tool Implementation**: Actual converter tool invocation and conversion logic
2. **Evaluation Implementation**: Actual evaluation artifact selection logic
3. **Scheduler Integration**: How scheduler invokes converter tools
4. **Cache Integration**: How artifact caching interacts with workflow execution
5. **Restart Logic**: How failed task restart works with cache enabled
6. **Framework Detection**: How tool framework labels are detected and used
7. **Format Validation**: Actual validation/rejection of non-.npy/.pt files

## Notes

- Some tests are conceptual (converter, evaluation) and test the requirement, not the implementation
- Artifact chaining tests verify the worker correctly copies dependency outputs
- Workflow generation tests verify the permutation and combination logic
- Edge case tests ensure robustness under extreme conditions
