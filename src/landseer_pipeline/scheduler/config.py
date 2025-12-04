"""
Configuration for the Dependency-Aware Scheduler

This file contains all configurable parameters for the enhanced scheduling system.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

@dataclass
class SchedulerConfig:
    """Configuration for the dependency-aware scheduler"""
    
    # Resource Management
    max_concurrent_tasks: Optional[int] = None  # Auto-determined if None
    gpu_cooldown_time: float = 300.0  # seconds between GPU allocations
    max_gpu_temperature: float = 60.0  # Celsius
    cpu_worker_count: int = 4  # Number of CPU-only workers
    
    # Task Management
    task_timeout: float = 7200.0  # Maximum time per task (2 hours)
    max_retries: int = 3  # Maximum retries for failed tasks
    priority_boost_threshold: float = 1800.0  # Boost priority after waiting (30 min)
    
    # Caching
    enable_tool_cache: bool = True
    cache_dir: str = "cache/tool_results"
    cache_cleanup_interval: float = 86400.0  # 24 hours
    max_cache_size_gb: float = 100.0  # Maximum cache size
    
    # Monitoring
    monitoring_interval: float = 30.0  # seconds
    enable_dashboard: bool = False
    log_dir: str = "logs/scheduler"
    save_dependency_graph: bool = True
    
    # Performance Tuning
    queue_batch_size: int = 10  # Process tasks in batches
    dependency_check_interval: float = 5.0  # seconds
    resource_rebalance_interval: float = 60.0  # seconds
    
    # Tool-specific settings
    tool_duration_estimates: Dict[str, float] = None
    tool_gpu_requirements: Dict[str, bool] = None
    tool_priority_overrides: Dict[str, str] = None  # tool_name -> priority
    
    def __post_init__(self):
        """Set default values for complex fields"""
        if self.tool_duration_estimates is None:
            self.tool_duration_estimates = {
                "noop": 30,
                "pre_xgbod": 120,
                "in_trades": 1800,
                "in_noop": 60,
                "post_fineprune": 600,
                "post_magnet": 900,
                "deploy_dp": 180,
                "dataset_inference": 240
            }
        
        if self.tool_gpu_requirements is None:
            self.tool_gpu_requirements = {
                "noop": False,
                "pre_xgbod": False,
                "in_trades": True,
                "in_noop": False,
                "post_fineprune": True,
                "post_magnet": True,
                "deploy_dp": True,
                "dataset_inference": True
            }
        
        if self.tool_priority_overrides is None:
            self.tool_priority_overrides = {
                "in_trades": "HIGH",  # Training has high priority
                "noop": "LOW"         # No-op has low priority
            }

@dataclass 
class MonitoringConfig:
    """Configuration for scheduler monitoring"""
    
    # Alert thresholds
    low_gpu_utilization_threshold: float = 0.3  # 30%
    high_blocked_tasks_threshold: float = 0.7   # 70%
    task_failure_rate_threshold: float = 0.1    # 10%
    
    # Dashboard settings
    dashboard_refresh_interval: float = 30.0  # seconds
    metrics_history_limit: int = 1000  # Maximum metrics to keep in memory
    alert_history_limit: int = 50      # Maximum alerts to keep
    
    # Reporting
    generate_reports: bool = True
    report_interval: float = 3600.0  # Generate reports every hour
    include_dependency_graph: bool = True
    include_performance_analysis: bool = True

# Default configuration instances
DEFAULT_SCHEDULER_CONFIG = SchedulerConfig()
DEFAULT_MONITORING_CONFIG = MonitoringConfig()

def load_config_from_file(config_file: Path) -> SchedulerConfig:
    """Load scheduler configuration from a JSON file"""
    import json
    
    if not config_file.exists():
        return DEFAULT_SCHEDULER_CONFIG
    
    try:
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        # Create config with loaded values
        return SchedulerConfig(**config_dict)
        
    except Exception as e:
        print(f"Warning: Failed to load config from {config_file}: {e}")
        return DEFAULT_SCHEDULER_CONFIG

def save_config_to_file(config: SchedulerConfig, config_file: Path):
    """Save scheduler configuration to a JSON file"""
    import json
    from dataclasses import asdict
    
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save config to {config_file}: {e}")
