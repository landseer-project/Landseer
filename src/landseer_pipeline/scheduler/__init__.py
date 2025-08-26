"""
Dependency-Aware Scheduler Module

This module provides advanced scheduling capabilities for the ML defense pipeline
including:

- Dependency-aware task scheduling
- Intelligent resource allocation
- Tool execution deduplication
- Real-time monitoring and visualization
- Optimal compute resource utilization

Main Components:
- DependencyAwareScheduler: Core scheduling engine
- EnhancedPipelineRunner: Drop-in replacement for basic pipeline execution
- SchedulerMonitor: Real-time monitoring and alerting
- ToolCache: Prevents duplicate tool executions
"""

from .dependency_scheduler import DependencyAwareScheduler, TaskState, TaskPriority
from .enhanced_runner import EnhancedPipelineRunner, create_enhanced_runner
from .monitor import SchedulerMonitor, SchedulerDashboard, start_monitoring

__all__ = [
    'DependencyAwareScheduler',
    'TaskState', 
    'TaskPriority',
    'EnhancedPipelineRunner',
    'create_enhanced_runner',
    'SchedulerMonitor',
    'SchedulerDashboard', 
    'start_monitoring'
]

__version__ = "1.0.0"
