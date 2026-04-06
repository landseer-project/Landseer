# Scheduler Test Matrix

This folder validates scheduler behavior end-to-end:

- `test_base_scheduler.py` - core scheduler invariants and status transitions
- `test_priority_scheduler.py` - priority/dependency behavior
- `test_priority_calculation.py` - depth/counter priority math
- `test_converter_invocation.py` - converter-trigger scheduling interactions
- `test_scheduler_comprehensive.py` - corner cases and critical behavior

## Run only scheduler tests

```bash
poetry run pytest tests/backend/scheduler -q
```

## Run one behavior quickly

```bash
poetry run pytest tests/backend/scheduler/test_scheduler_comprehensive.py::test_is_complete_true_when_all_terminal_even_with_failures -q
```
