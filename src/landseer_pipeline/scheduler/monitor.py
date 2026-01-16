"""
Real-time Scheduler Monitoring and Visualization

Provides comprehensive monitoring of the dependency-aware scheduler including:
- Resource utilization tracking
- Dependency graph visualization  
- Performance metrics
- Live status dashboard
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

@dataclass
class MetricSnapshot:
    """Snapshot of scheduler metrics at a point in time"""
    timestamp: float
    tasks_total: int
    tasks_completed: int
    tasks_failed: int
    tasks_running: int
    tasks_ready: int
    tasks_blocked: int
    gpu_utilization: float
    cpu_utilization: float
    cache_hit_rate: float
    throughput: float
    gpu_hours: float
    cpu_hours: float

class SchedulerMonitor:
    """Real-time monitoring for the dependency-aware scheduler"""
    
    def __init__(self, scheduler, log_dir: Path = None):
        self.scheduler = scheduler
        self.log_dir = log_dir or Path("logs/scheduler_monitoring")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history: List[MetricSnapshot] = []
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        # Performance tracking
        self.performance_alerts = []
        self.resource_alerts = []
        
    def start_monitoring(self, interval: float = 30.0):
        """Start real-time monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval,),
            name="SchedulerMonitor"
        )
        self.monitor_thread.start()
        logger.info(f"📊 Started scheduler monitoring (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop monitoring and save final report"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        self._save_monitoring_report()
        logger.info("📊 Stopped scheduler monitoring")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._collect_metrics()
                self._check_performance_alerts()
                self._log_status_update()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}", exc_info=True)
    
    def _collect_metrics(self):
        """Collect current scheduler metrics"""
        status = self.scheduler.get_status()
        
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            tasks_total=status["total_tasks"],
            tasks_completed=status["completed_tasks"],
            tasks_failed=status["metrics"]["tasks_failed"],
            tasks_running=status["running_tasks"],
            tasks_ready=status["ready_tasks"],
            tasks_blocked=status["blocked_tasks"],
            gpu_utilization=status["throughput"]["gpu_utilization"],
            cpu_utilization=0.0,  # TODO: Add CPU utilization tracking
            cache_hit_rate=status["throughput"]["cache_hit_rate"],
            throughput=status["throughput"]["tasks_per_hour"],
            gpu_hours=status["metrics"]["total_gpu_hours"],
            cpu_hours=status["metrics"]["total_cpu_hours"]
        )
        
        with self.lock:
            self.metrics_history.append(snapshot)
            
            # Keep only last 1000 snapshots to avoid memory issues
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
    
    def _check_performance_alerts(self):
        """Check for performance issues and generate alerts"""
        if len(self.metrics_history) < 2:
            return
        
        current = self.metrics_history[-1]
        previous = self.metrics_history[-2]
        
        # Alert: Low GPU utilization
        if current.gpu_utilization < 0.5 and current.tasks_ready > 0:
            alert = {
                "type": "low_gpu_utilization",
                "timestamp": current.timestamp,
                "message": f"GPU utilization only {current.gpu_utilization:.1%} with {current.tasks_ready} ready tasks",
                "severity": "warning"
            }
            self.performance_alerts.append(alert)
        
        # Alert: High number of blocked tasks
        if current.tasks_blocked > current.tasks_total * 0.7:
            alert = {
                "type": "high_blocked_tasks",
                "timestamp": current.timestamp,
                "message": f"{current.tasks_blocked}/{current.tasks_total} tasks blocked",
                "severity": "warning"
            }
            self.performance_alerts.append(alert)
        
        # Alert: Decreasing throughput
        if len(self.metrics_history) >= 5:
            recent_throughputs = [m.throughput for m in self.metrics_history[-5:]]
            if all(recent_throughputs[i] >= recent_throughputs[i+1] for i in range(len(recent_throughputs)-1)):
                alert = {
                    "type": "decreasing_throughput",
                    "timestamp": current.timestamp,
                    "message": f"Throughput decreasing: {current.throughput:.1f} tasks/hour",
                    "severity": "info"
                }
                self.performance_alerts.append(alert)
        
        # Limit alert history
        self.performance_alerts = self.performance_alerts[-50:]
    
    def _log_status_update(self):
        """Log periodic status update"""
        if not self.metrics_history:
            return
        
        current = self.metrics_history[-1]
        
        logger.info("📊 SCHEDULER STATUS UPDATE")
        logger.info(f"   Tasks: {current.tasks_completed}/{current.tasks_total} completed")
        logger.info(f"   Running: {current.tasks_running}, Ready: {current.tasks_ready}, Blocked: {current.tasks_blocked}")
        logger.info(f"   GPU Util: {current.gpu_utilization:.1%}, Throughput: {current.throughput:.1f}/hr")
        logger.info(f"   Cache Hit: {current.cache_hit_rate:.1%}")
        
        # Log recent alerts
        recent_alerts = [a for a in self.performance_alerts if time.time() - a["timestamp"] < 300]
        if recent_alerts:
            logger.warning(f"   ⚠️  {len(recent_alerts)} recent performance alerts")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get comprehensive current status"""
        scheduler_status = self.scheduler.get_status()
        
        with self.lock:
            latest_metrics = self.metrics_history[-1] if self.metrics_history else None
            recent_alerts = [a for a in self.performance_alerts if time.time() - a["timestamp"] < 3600]
        
        return {
            "scheduler_status": scheduler_status,
            "latest_metrics": asdict(latest_metrics) if latest_metrics else None,
            "recent_alerts": recent_alerts,
            "monitoring_active": self.monitoring_active,
            "metrics_history_length": len(self.metrics_history)
        }
    
    def get_performance_summary(self, time_window_hours: float = 1.0) -> Dict[str, Any]:
        """Get performance summary for a time window"""
        cutoff_time = time.time() - (time_window_hours * 3600)
        
        with self.lock:
            recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": "No metrics available for the specified time window"}
        
        # Calculate averages
        avg_gpu_util = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
        avg_cache_hit = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        
        # Calculate trends
        if len(recent_metrics) >= 2:
            first, last = recent_metrics[0], recent_metrics[-1]
            task_completion_rate = (last.tasks_completed - first.tasks_completed) / max(last.timestamp - first.timestamp, 1)
        else:
            task_completion_rate = 0
        
        return {
            "time_window_hours": time_window_hours,
            "metrics_count": len(recent_metrics),
            "averages": {
                "gpu_utilization": avg_gpu_util,
                "throughput": avg_throughput,
                "cache_hit_rate": avg_cache_hit
            },
            "trends": {
                "task_completion_rate": task_completion_rate * 3600  # tasks per hour
            },
            "current": asdict(recent_metrics[-1]) if recent_metrics else None
        }
    
    def generate_dependency_graph_viz(self, output_file: Path = None) -> str:
        """Generate DOT format dependency graph visualization"""
        if not output_file:
            output_file = self.log_dir / f"dependency_graph_{int(time.time())}.dot"
        
        try:
            dot_content = ["digraph DependencyGraph {"]
            dot_content.append("  rankdir=TB;")
            dot_content.append("  node [shape=box, style=filled];")
            
            # Add nodes with status colors
            for task_id, task in self.scheduler.tasks.items():
                color = {
                    "PENDING": "lightgray",
                    "QUEUED": "lightblue", 
                    "RUNNING": "yellow",
                    "COMPLETED": "lightgreen",
                    "FAILED": "lightcoral",
                    "BLOCKED": "orange"
                }.get(task.state.name, "white")
                
                label = f"{task.tool_execution.tool_name}\\n{task.tool_execution.stage}"
                dot_content.append(f'  "{task_id}" [label="{label}", fillcolor="{color}"];')
            
            # Add dependency edges
            for task_id, task in self.scheduler.tasks.items():
                for dep_id in task.dependencies:
                    if dep_id in self.scheduler.tasks:
                        dot_content.append(f'  "{dep_id}" -> "{task_id}";')
            
            dot_content.append("}")
            
            dot_string = "\n".join(dot_content)
            
            with open(output_file, 'w') as f:
                f.write(dot_string)
            
            logger.info(f"📊 Generated dependency graph: {output_file}")
            return dot_string
            
        except Exception as e:
            logger.error(f"Failed to generate dependency graph: {e}")
            return ""
    
    def _save_monitoring_report(self):
        """Save comprehensive monitoring report"""
        report_file = self.log_dir / f"monitoring_report_{int(time.time())}.json"
        
        try:
            with self.lock:
                report = {
                    "generated_at": time.time(),
                    "generated_at_iso": datetime.now().isoformat(),
                    "scheduler_final_status": self.scheduler.get_status(),
                    "metrics_history": [asdict(m) for m in self.metrics_history],
                    "performance_alerts": self.performance_alerts,
                    "resource_alerts": self.resource_alerts,
                    "summary": self.get_performance_summary(24.0)  # Last 24 hours
                }
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"📊 Saved monitoring report: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save monitoring report: {e}")

class SchedulerDashboard:
    """Simple text-based dashboard for scheduler monitoring"""
    
    def __init__(self, monitor: SchedulerMonitor):
        self.monitor = monitor
        
    def print_dashboard(self):
        """Print current dashboard to console"""
        status = self.monitor.get_current_status()
        
        print("\n" + "="*80)
        print("🎯 DEPENDENCY-AWARE SCHEDULER DASHBOARD")
        print("="*80)
        
        if status["latest_metrics"]:
            metrics = status["latest_metrics"]
            print(f"📊 Tasks: {metrics['tasks_completed']}/{metrics['tasks_total']} completed")
            print(f"🔄 Running: {metrics['tasks_running']}, Ready: {metrics['tasks_ready']}, Blocked: {metrics['tasks_blocked']}")
            print(f"🎮 GPU Utilization: {metrics['gpu_utilization']:.1%}")
            print(f"🚀 Throughput: {metrics['throughput']:.1f} tasks/hour")
            print(f"💾 Cache Hit Rate: {metrics['cache_hit_rate']:.1%}")
            print(f"⏰ GPU Hours: {metrics['gpu_hours']:.2f}, CPU Hours: {metrics['cpu_hours']:.2f}")
        
        resource_status = status["scheduler_status"]["resource_status"]
        print(f"🔧 Resources: {resource_status['available_gpus']}/{resource_status['total_gpus']} GPUs available")
        print(f"🖥️  CPU Workers: {resource_status['available_cpu_workers']}/{resource_status['total_cpu_workers']} available")
        
        if status["recent_alerts"]:
            print(f"⚠️  Recent Alerts: {len(status['recent_alerts'])}")
            for alert in status["recent_alerts"][-3:]:  # Show last 3 alerts
                print(f"   • {alert['type']}: {alert['message']}")
        
        print("="*80)
    
    def start_live_dashboard(self, refresh_interval: float = 30.0):
        """Start live updating dashboard"""
        import os
        
        try:
            while self.monitor.monitoring_active:
                os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
                self.print_dashboard()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped by user")

# Helper functions for easy integration
def start_monitoring(scheduler, log_dir: Path = None, dashboard: bool = False) -> SchedulerMonitor:
    """Start monitoring for a scheduler"""
    monitor = SchedulerMonitor(scheduler, log_dir)
    monitor.start_monitoring()
    
    if dashboard:
        dashboard_obj = SchedulerDashboard(monitor)
        dashboard_thread = threading.Thread(
            target=dashboard_obj.start_live_dashboard,
            name="SchedulerDashboard"
        )
        dashboard_thread.daemon = True
        dashboard_thread.start()
    
    return monitor
