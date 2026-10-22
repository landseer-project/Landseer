"""A failed task's error carries the end of the tool's output."""

from src.worker.runner import FAILURE_LOG_CHARS, FAILURE_LOG_LINES, failure_summary


def test_no_logs_keeps_message():
    assert failure_summary("Container exited with code 1", "") == "Container exited with code 1"
    assert failure_summary("Container exited with code 1", None) == "Container exited with code 1"


def test_includes_traceback_tail():
    logs = "\n".join(["epoch 1", "epoch 2", "Traceback (most recent call last):",
                      '  File "main.py", line 3, in <module>', "ValueError: bad prune ratio"])
    out = failure_summary("Container exited with code 1", logs)
    assert out.startswith("Container exited with code 1\n--- last lines of tool output ---\n")
    assert out.endswith("ValueError: bad prune ratio")
    assert "epoch 1" not in out  # starts at the traceback


def test_long_output_is_capped():
    logs = "\n".join(f"line {i} " + "x" * 300 for i in range(500))
    out = failure_summary("boom", logs)
    tail = out.split("--- last lines of tool output ---\n", 1)[1]
    assert len(tail) <= FAILURE_LOG_CHARS + 3
    assert "line 499" in tail and "line 0 " not in tail
    assert len(tail.splitlines()) <= FAILURE_LOG_LINES
