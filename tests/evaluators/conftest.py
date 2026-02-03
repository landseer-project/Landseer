"""
Pytest configuration for evaluator tests.

Provides fixtures and proper import handling for evaluator modules.
"""

import sys
from pathlib import Path
import importlib.util

import pytest


def load_evaluator_module(evaluator_name: str):
    """
    Load an evaluator module from the containers directory.
    
    Args:
        evaluator_name: Name of the evaluator (e.g., 'adversarial', 'ood')
        
    Returns:
        Loaded module
    """
    containers_dir = Path(__file__).parents[2] / "containers" / "evals"
    eval_path = containers_dir / evaluator_name / "evaluate.py"
    
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluator not found: {eval_path}")
    
    spec = importlib.util.spec_from_file_location(
        f"eval_{evaluator_name}", 
        str(eval_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def adversarial_module():
    """Load the adversarial evaluator module."""
    return load_evaluator_module("adversarial")


@pytest.fixture
def ood_module():
    """Load the OOD evaluator module."""
    return load_evaluator_module("ood")


@pytest.fixture
def fingerprinting_module():
    """Load the fingerprinting evaluator module."""
    return load_evaluator_module("fingerprinting")


@pytest.fixture
def fairness_module():
    """Load the fairness evaluator module."""
    return load_evaluator_module("fairness")


@pytest.fixture
def backdoor_module():
    """Load the backdoor evaluator module."""
    return load_evaluator_module("backdoor")


@pytest.fixture
def watermark_module():
    """Load the watermark evaluator module."""
    return load_evaluator_module("watermark")
