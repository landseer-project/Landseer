import asyncio
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    """Represents a single tool execution task"""
    combination_id: str
    stage: str
    tool_name: str
    dependencies: Set[str] = field(default_factory=set)  # Task IDs this depends on
    gpu_requirement: int = 1  # Number of GPUs needed
    estimated_duration: float = 300  # Estimated seconds
    priority: int = 0  # Higher = more priority
    status: TaskStatus = TaskStatus.PENDING
    assigned_gpus: List[int] = field(default_factory=list)
    start_time: Optional[float] = None
    
    @property
    def task_id(self) -> str:
        return f"{self.combination_id}_{self.stage}_{self.tool_name}"
    
    @property
    def is_noop(self) -> bool:
        return "noop" in self.tool_name.lower()

@dataclass
class GPUResource:
    """Tracks GPU resource availability"""
    gpu_id: int
    in_use: bool = False
    current_task: Optional[str] = None
    temperature: float = 0.0
    utilization: float = 0.0
    memory_used: float = 0.0
    
class EnhancedScheduler:
    """
    Advanced scheduler with dependency-aware execution, GPU load balancing,
    and opportunistic parallelization
    """
    
    def __init__(self, gpu_manager, max_concurrent_tasks=8):
        self.gpu_manager = gpu_manager
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue = deque()
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        
        # Dependency tracking
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        # GPU resource tracking
        self.gpu_resources: Dict[int, GPUResource] = {}
        self._init_gpu_resources()
        
        # Performance tracking
        self.task_durations: Dict[str, float] = {}
        self.tool_performance_history: Dict[str, List[float]] = defaultdict(list)
        
        # Scheduler thread
        self.scheduler_thread = None
        self.shutdown_event = threading.Event()
        
    def _init_gpu_resources(self):
        """Initialize GPU resource tracking"""
        for gpu_id in range(self.gpu_manager.device_count):
            self.gpu_resources[gpu_id] = GPUResource(gpu_id=gpu_id)
    
    def add_task(self, combination_id: str, stage: str, tool_name: str, 
                 dependencies: Optional[List[str]] = None, 
                 gpu_requirement: int = 1,
                 priority: int = 0) -> str:
        """Add a task to the scheduler"""
        task = Task(
            combination_id=combination_id,
            stage=stage,
            tool_name=tool_name,
            dependencies=set(dependencies or []),
            gpu_requirement=gpu_requirement,
            priority=priority,
            estimated_duration=self._estimate_duration(tool_name)
        )
        
        self.tasks[task.task_id] = task
        
        # Build dependency graph
        for dep_id in task.dependencies:
            self.dependency_graph[dep_id].add(task.task_id)
            self.reverse_dependencies[task.task_id].add(dep_id)
        
        # Check if task is ready to run
        if self._is_task_ready(task):
            task.status = TaskStatus.READY
            self.task_queue.append(task.task_id)
        
        logger.debug(f"Added task {task.task_id} with {len(task.dependencies)} dependencies")
        return task.task_id
    
    def _estimate_duration(self, tool_name: str) -> float:
        """Estimate task duration based on historical data"""
        if tool_name in self.tool_performance_history:
            history = self.tool_performance_history[tool_name]
            # Use median of recent executions
            recent_history = history[-5:]  # Last 5 executions
            return sum(recent_history) / len(recent_history) if recent_history else 300
        
        # Default estimates based on tool type
        estimates = {
            'noop': 30,
            'pre_': 120,
            'in_': 600,  # Training tools take longer
            'post_': 300,
            'deploy_': 180,
            'eval': 240
        }
        
        for prefix, duration in estimates.items():
            if prefix in tool_name.lower():
                return duration
        
        return 300  # Default 5 minutes
    
    def _is_task_ready(self, task: Task) -> bool:
        """Check if all dependencies are completed"""
        return all(dep_id in self.completed_tasks for dep_id in task.dependencies)
    
    def _update_ready_tasks(self):
        """Update tasks that became ready after dependency completion"""
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING and self._is_task_ready(task):
                task.status = TaskStatus.READY
                self.task_queue.append(task_id)
                logger.debug(f"Task {task_id} is now ready")
    
    def _allocate_gpus(self, task: Task) -> List[int]:
        """Allocate GPUs for a task using intelligent load balancing"""
        available_gpus = []
        
        # Get current GPU stats
        gpu_stats = self.gpu_manager.get_gpu_stats()
        
        # Sort GPUs by availability score (lower is better)
        gpu_scores = []
        for stat in gpu_stats:
            gpu_id = stat['id']
            if not self.gpu_resources[gpu_id].in_use:
                # Score based on temperature and utilization
                score = stat['temperature'] * 0.6 + stat['gpu_utilization'] * 0.4
                gpu_scores.append((score, gpu_id))
        
        # Sort by score and allocate best GPUs
        gpu_scores.sort()
        allocated = []
        
        for _, gpu_id in gpu_scores[:task.gpu_requirement]:
            # Double-check availability
            gpu_id_from_manager = self.gpu_manager.get_available_gpu()
            if gpu_id_from_manager is not None:
                allocated.append(gpu_id_from_manager)
                self.gpu_resources[gpu_id_from_manager].in_use = True
                self.gpu_resources[gpu_id_from_manager].current_task = task.task_id
            
            if len(allocated) >= task.gpu_requirement:
                break
        
        return allocated
    
    def _should_boost_container(self, task: Task) -> bool:
        """Determine if a running container should get additional resources"""
        if not task.is_noop and task.status == TaskStatus.RUNNING:
            # Check if there are idle GPUs
            idle_gpus = sum(1 for gpu in self.gpu_resources.values() if not gpu.in_use)
            
            # If task is taking longer than expected and GPUs are idle
            if task.start_time:
                elapsed = time.time() - task.start_time
                if elapsed > task.estimated_duration * 0.7 and idle_gpus > 0:
                    return True
        
        return False
    
    def _boost_container_resources(self, task: Task):
        """Allocate additional resources to a running container"""
        additional_gpus = self._allocate_gpus(Task(
            combination_id="boost",
            stage="boost", 
            tool_name="boost",
            gpu_requirement=1
        ))
        
        if additional_gpus:
            task.assigned_gpus.extend(additional_gpus)
            logger.info(f"Boosted task {task.task_id} with additional GPU {additional_gpus[0]}")
            
            # TODO: Notify the running container about additional GPU
            # This would require container communication mechanism
    
    def _select_next_task(self) -> Optional[Task]:
        """Select the next task to run using intelligent scheduling"""
        if not self.task_queue:
            return None
        
        # Calculate scheduling scores for all ready tasks
        task_scores = []
        
        for task_id in list(self.task_queue):
            task = self.tasks[task_id]
            
            # Check GPU availability
            available_gpus = sum(1 for gpu in self.gpu_resources.values() if not gpu.in_use)
            if available_gpus < task.gpu_requirement:
                continue
            
            # Calculate priority score
            score = task.priority * 100  # Base priority
            
            # Boost short tasks
            if task.estimated_duration < 60:
                score += 50
            
            # Boost tasks with many dependents
            dependents = len(self.dependency_graph.get(task_id, set()))
            score += dependents * 20
            
            # Boost non-noop tasks when GPUs are available
            if not task.is_noop and available_gpus >= 2:
                score += 30
            
            # Penalize GPU-hungry tasks when resources are scarce
            if task.gpu_requirement > 1 and available_gpus < 3:
                score -= 40
            
            task_scores.append((score, task_id, task))
        
        if not task_scores:
            return None
        
        # Sort by score (highest first) and return best task
        task_scores.sort(reverse=True)
        _, selected_task_id, selected_task = task_scores[0]
        
        # Remove from queue
        self.task_queue.remove(selected_task_id)
        
        return selected_task
    
    def start_task(self, task: Task) -> bool:
        """Start executing a task"""
        # Allocate GPUs
        allocated_gpus = self._allocate_gpus(task)
        
        if len(allocated_gpus) < task.gpu_requirement:
            # Not enough GPUs, put back in queue
            self.task_queue.appendleft(task.task_id)
            return False
        
        task.assigned_gpus = allocated_gpus
        task.status = TaskStatus.RUNNING
        task.start_time = time.time()
        self.running_tasks[task.task_id] = task
        
        logger.info(f"Started task {task.task_id} on GPUs {allocated_gpus}")
        return True
    
    def complete_task(self, task_id: str, success: bool = True):
        """Mark a task as completed and release resources"""
        if task_id not in self.running_tasks:
            logger.warning(f"Attempted to complete non-running task {task_id}")
            return
        
        task = self.running_tasks.pop(task_id)
        
        # Release GPUs
        for gpu_id in task.assigned_gpus:
            self.gpu_manager.release_gpu(gpu_id)
            self.gpu_resources[gpu_id].in_use = False
            self.gpu_resources[gpu_id].current_task = None
        
        # Update performance history
        if task.start_time:
            duration = time.time() - task.start_time
            self.tool_performance_history[task.tool_name].append(duration)
        
        if success:
            task.status = TaskStatus.COMPLETED
            self.completed_tasks.add(task_id)
            logger.info(f"Completed task {task_id}")
        else:
            task.status = TaskStatus.FAILED
            self.failed_tasks.add(task_id)
            logger.error(f"Failed task {task_id}")
        
        # Update dependent tasks
        self._update_ready_tasks()
    
    def scheduler_loop(self):
        """Main scheduler loop"""
        logger.info("Enhanced scheduler started")
        
        while not self.shutdown_event.is_set():
            try:
                # Check for running task completion/boosting
                for task_id, task in list(self.running_tasks.items()):
                    if self._should_boost_container(task):
                        self._boost_container_resources(task)
                
                # Start new tasks if resources available
                if len(self.running_tasks) < self.max_concurrent_tasks:
                    next_task = self._select_next_task()
                    if next_task:
                        self.start_task(next_task)
                
                # Brief sleep to avoid busy loop
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(5)
        
        logger.info("Enhanced scheduler stopped")
    
    def start(self):
        """Start the scheduler"""
        if self.scheduler_thread is None or not self.scheduler_thread.is_alive():
            self.shutdown_event.clear()
            self.scheduler_thread = threading.Thread(target=self.scheduler_loop)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
    
    def shutdown(self):
        """Shutdown the scheduler"""
        self.shutdown_event.set()
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)
    
    def get_status(self) -> Dict:
        """Get current scheduler status"""
        return {
            'total_tasks': len(self.tasks),
            'pending_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
            'ready_tasks': len(self.task_queue),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'gpu_utilization': {
                gpu_id: {'in_use': gpu.in_use, 'task': gpu.current_task}
                for gpu_id, gpu in self.gpu_resources.items()
            }
        }
