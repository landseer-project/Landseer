"""
Global pytest configuration for test organization.

This file auto-assigns markers based on file path so teams can run:
- backend API only
- scheduler only
- db-only integration tests
- worker/pipeline/evaluator suites
without manually maintaining markers in every file.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _has(part: str, path: str) -> bool:
    return part in path.replace("\\", "/")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        path = str(item.fspath)
        name = Path(path).name

        # Top-level domain markers
        if _has("/tests/backend/", path):
            item.add_marker(pytest.mark.backend)
        if _has("/tests/pipeline/", path):
            item.add_marker(pytest.mark.pipeline)
        if _has("/tests/worker/", path):
            item.add_marker(pytest.mark.worker)
        if _has("/tests/evaluators/", path):
            item.add_marker(pytest.mark.evaluators)

        # Backend subdomain markers
        if _has("/tests/backend/api/", path):
            item.add_marker(pytest.mark.api)
        if _has("/tests/backend/scheduler/", path):
            item.add_marker(pytest.mark.scheduler)

        # Cross-cutting markers
        lower_name = name.lower()
        if "integration" in lower_name:
            item.add_marker(pytest.mark.integration)
        if "security" in lower_name:
            item.add_marker(pytest.mark.security)
        if "db" in lower_name or "database" in lower_name:
            item.add_marker(pytest.mark.db)
        if "smoke" in lower_name:
            item.add_marker(pytest.mark.smoke)
