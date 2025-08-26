"""
Enhanced Pipeline Runner with Dependency-Aware Scheduling

This module replaces the basic ThreadPoolExecutor approach with sophisticated
scheduling that maximizes resource utilization while respecting dependencies.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

from ..scheduler.dependency_scheduler import DependencyAwareScheduler
from ..gpu_manager import GPUManager

logger = logging.getLogger(__name__)

class EnhancedPipelineRunner:
    """Enhanced pipeline runner with dependency-aware scheduling"""
    
    def __init__(self, pipeline_executor, gpu_manager: Optional[GPUManager] = None):
        self.pipeline_executor = pipeline_executor
        self.gpu_manager = gpu_manager or GPUManager()
        
        # Initialize scheduler
        self.scheduler = DependencyAwareScheduler(
            pipeline_executor=pipeline_executor,
            gpu_manager=self.gpu_manager,
            max_concurrent_tasks=None  # Auto-determined based on resources
        )
        
        self.execution_stats = {}
    
    def run_all_combinations_parallel(self, 
                                    combinations: Dict[str, Any], 
                                    progress_callback=None,
                                    timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Run all combinations using dependency-aware scheduling
        
        Args:
            combinations: Dictionary of combination objects to execute
            progress_callback: Optional callback for progress updates
            timeout: Maximum time to wait for completion (seconds)
            
        Returns:
            Dictionary with execution results and statistics
        """
        start_time = time.time()
        
        logger.info(f"🚀 Starting enhanced pipeline execution for {len(combinations)} combinations")
        logger.info(f"💪 Available resources: {self.gpu_manager.device_count} GPUs")
        
        try:
            # Submit all combinations to scheduler
            self.scheduler.submit_combinations(combinations)
            
            # Start scheduler
            self.scheduler.start()
            
            # Monitor progress
            if progress_callback:
                self._monitor_progress(progress_callback)
            
            # Wait for completion
            success = self.scheduler.wait_for_completion(timeout=timeout)
            
            # Stop scheduler
            self.scheduler.stop()
            
            # Collect results
            execution_time = time.time() - start_time
            final_status = self.scheduler.get_status()
            
            # Log summary
            self._log_execution_summary(final_status, execution_time)
            
            return {
                "success": success,
                "execution_time": execution_time,
                "scheduler_status": final_status,
                "completed_tasks": self.scheduler.completed_tasks,
                "failed_tasks": {
                    task_id: task for task_id, task in self.scheduler.tasks.items()
                    if task.state.name == "FAILED"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}", exc_info=True)
            self.scheduler.stop()
            raise
    
    def _monitor_progress(self, progress_callback):
        """Monitor execution progress and call progress callback"""
        last_update = time.time()
        
        while self.scheduler.running:
            current_time = time.time()
            
            # Update every 30 seconds
            if current_time - last_update >= 30:
                status = self.scheduler.get_status()
                progress_callback(status)
                last_update = current_time
            
            time.sleep(5)
    
    def _log_execution_summary(self, final_status: Dict[str, Any], execution_time: float):
        """Log execution summary with key metrics"""
        metrics = final_status["metrics"]
        throughput = final_status["throughput"]
        resource_status = final_status["resource_status"]
        
        logger.info("=" * 80)
        logger.info("🎯 ENHANCED PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total execution time: {execution_time:.1f} seconds")
        logger.info(f"📊 Tasks completed: {metrics['tasks_completed']}")
        logger.info(f"❌ Tasks failed: {metrics['tasks_failed']}")
        logger.info(f"💾 Cache hits: {metrics['cache_hits']} ({throughput['cache_hit_rate']:.1%})")
        logger.info(f"🚀 Throughput: {throughput['tasks_per_hour']:.1f} tasks/hour")
        logger.info(f"🎮 GPU utilization: {throughput['gpu_utilization']:.1%}")
        logger.info(f"💰 GPU hours used: {metrics['total_gpu_hours']:.2f}")
        logger.info(f"🖥️  CPU hours used: {metrics['total_cpu_hours']:.2f}")
        logger.info("=" * 80)
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        return self.scheduler.get_status()
    
    def stop_execution(self):
        """Stop the current execution"""
        logger.info("🛑 Stopping pipeline execution...")
        self.scheduler.stop()

# Helper function to replace existing runner usage
def create_enhanced_runner(pipeline_executor, gpu_manager: Optional[GPUManager] = None) -> EnhancedPipelineRunner:
    """Create an enhanced pipeline runner instance"""
    return EnhancedPipelineRunner(pipeline_executor, gpu_manager)

# Backward compatibility function
def run_combinations_with_enhanced_scheduling(pipeline_executor, 
                                            combinations: Dict[str, Any],
                                            gpu_manager: Optional[GPUManager] = None,
                                            **kwargs) -> Dict[str, Any]:
    """
    Run combinations with enhanced scheduling (backward compatibility)
    
    This function provides a drop-in replacement for the original
    run_all_combinations_parallel function.
    """
    runner = create_enhanced_runner(pipeline_executor, gpu_manager)
    return runner.run_all_combinations_parallel(combinations, **kwargs)
