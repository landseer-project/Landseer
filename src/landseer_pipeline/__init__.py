"""
Landseer Pipeline Package

A modular ML defense pipeline supporting both Docker and Apptainer container runtimes.
"""

from .config.settings import is_dry_run, get_current_settings

__all__ = [
    'is_dry_run',
    'get_current_settings'
]