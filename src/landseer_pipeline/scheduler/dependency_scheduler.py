"""
Advanced Dependency-Aware Scheduler for ML Defense Pipeline

This scheduler provides:
1. Full compute resource utilization (all GPUs)
2. Tool dependency management 
3. Context-aware execution
4. Intelligent task queuing and prioritization
5. Duplicate tool execution prevention
"""

import logging
import threading
import time
import hashlib
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import json

logger = logging.getLogger(__name__)

class TaskState(Enum):
    PENDING = "pending"
    QUEUED = "queued" 
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ToolExecution:
    """Represents a single tool execution instance"""
    tool_name: str
    stage: str
    combination_id: str
    input_hash: str  # Hash of input data for deduplication
    dependencies: List[str] = field(default_factory=list)  # List of dependent tool executions
    gpu_required: bool = True
    estimated_duration: float = 300.0  # seconds
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        self.task_id = f"{self.stage}_{self.tool_name}_{self.input_hash[:8]}"

@dataclass 
class Task:
    """Represents a schedulable task"""
    task_id: str
    tool_execution: ToolExecution
    state: TaskState = TaskState.PENDING
    assigned_gpu: Optional[int] = None
    assigned_worker: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)

class ResourcePool:
    """Manages available compute resources"""
    
    def __init__(self, gpu_manager):
        self.gpu_manager = gpu_manager
        self.available_gpus = set(range(gpu_manager.device_count))
        self.allocated_gpus = {}  # gpu_id -> task_id
        self.cpu_workers = set(range(4))  # Assume 4 CPU workers
        self.allocated_cpu_workers = {}  # worker_id -> task_id
        self.lock = threading.Lock()
        
    def allocate_gpu(self, task_id: str) -> Optional[int]:
        """Allocate a GPU for a task"""
        with self.lock:
            gpu_id = self.gpu_manager.get_available_gpu()
            if gpu_id is not None and gpu_id in self.available_gpus:
                self.available_gpus.remove(gpu_id)
                self.allocated_gpus[gpu_id] = task_id
                logger.info(f"🎯 Allocated GPU {gpu_id} to task {task_id}")
                return gpu_id
            return None
    
    def allocate_cpu_worker(self, task_id: str) -> Optional[int]:
        """Allocate a CPU worker for non-GPU tasks"""
        with self.lock:
            if self.cpu_workers:
                worker_id = self.cpu_workers.pop()
                self.allocated_cpu_workers[worker_id] = task_id
                logger.info(f"🖥️ Allocated CPU worker {worker_id} to task {task_id}")
                return worker_id
            return None
    
    def release_gpu(self, gpu_id: int, task_id: str):
        """Release a GPU from a task"""
        with self.lock:
            if gpu_id in self.allocated_gpus and self.allocated_gpus[gpu_id] == task_id:
                del self.allocated_gpus[gpu_id]
                self.available_gpus.add(gpu_id)
                self.gpu_manager.release_gpu(gpu_id)
                logger.info(f"🔓 Released GPU {gpu_id} from task {task_id}")
    
    def release_cpu_worker(self, worker_id: int, task_id: str):
        """Release a CPU worker from a task"""
        with self.lock:
            if worker_id in self.allocated_cpu_workers and self.allocated_cpu_workers[worker_id] == task_id:
                del self.allocated_cpu_workers[worker_id]
                self.cpu_workers.add(worker_id)
                logger.info(f"🔓 Released CPU worker {worker_id} from task {task_id}")
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource utilization status"""
        with self.lock:
            return {
                "available_gpus": len(self.available_gpus),
                "total_gpus": self.gpu_manager.device_count,
                "allocated_gpus": dict(self.allocated_gpus),
                "available_cpu_workers": len(self.cpu_workers),
                "total_cpu_workers": 4,
                "allocated_cpu_workers": dict(self.allocated_cpu_workers)
            }

class DependencyGraph:
    """Manages task dependencies"""
    
    def __init__(self):
        self.graph = defaultdict(set)  # task_id -> set of dependent task_ids
        self.reverse_graph = defaultdict(set)  # task_id -> set of dependency task_ids
        self.completed_tasks = set()
        self.lock = threading.Lock()
    
    def add_dependency(self, task_id: str, depends_on: str):
        """Add a dependency relationship"""
        with self.lock:
            self.graph[depends_on].add(task_id)
            self.reverse_graph[task_id].add(depends_on)
    
    def mark_completed(self, task_id: str) -> List[str]:
        """Mark a task as completed and return newly unblocked tasks"""
        with self.lock:
            self.completed_tasks.add(task_id)
            unblocked = []
            
            # Check all dependent tasks
            for dependent_task in self.graph[task_id]:
                # Check if all dependencies are now completed
                all_deps_complete = all(
                    dep in self.completed_tasks 
                    for dep in self.reverse_graph[dependent_task]
                )
                if all_deps_complete:
                    unblocked.append(dependent_task)
            
            return unblocked
    
    def is_ready(self, task_id: str) -> bool:
        """Check if a task is ready to execute (all dependencies completed)"""
        with self.lock:
            return all(
                dep in self.completed_tasks 
                for dep in self.reverse_graph[task_id]
            )
    
    def get_pending_dependencies(self, task_id: str) -> Set[str]:
        """Get list of pending dependencies for a task"""
        with self.lock:
            return self.reverse_graph[task_id] - self.completed_tasks

class ToolCache:
    """Manages tool execution results to prevent duplicate work"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}  # input_hash -> result_path
        self.lock = threading.Lock()
        self._load_cache_index()
    
    def _load_cache_index(self):
        """Load cache index from disk"""
        index_file = self.cache_dir / "tool_cache_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self.memory_cache = json.load(f)
                logger.info(f"📚 Loaded {len(self.memory_cache)} cached tool results")
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        index_file = self.cache_dir / "tool_cache_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.memory_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def get_cached_result(self, input_hash: str) -> Optional[Path]:
        """Get cached result for a tool execution"""
        with self.lock:
            if input_hash in self.memory_cache:
                result_path = Path(self.memory_cache[input_hash])
                if result_path.exists():
                    logger.info(f"💾 Cache HIT for input hash {input_hash[:8]}")
                    return result_path
                else:
                    # Remove stale cache entry
                    del self.memory_cache[input_hash]
                    self._save_cache_index()
            return None
    
    def store_result(self, input_hash: str, result_path: Path):
        """Store a tool execution result in cache"""
        with self.lock:
            self.memory_cache[input_hash] = str(result_path)
            self._save_cache_index()
            logger.info(f"💾 Cached result for input hash {input_hash[:8]}")

class DependencyAwareScheduler:
    """Advanced scheduler with dependency management and resource optimization"""
    
    def __init__(self, pipeline_executor, gpu_manager, max_concurrent_tasks: int = None):
        self.pipeline_executor = pipeline_executor
        self.gpu_manager = gpu_manager
        self.max_concurrent_tasks = max_concurrent_tasks or (gpu_manager.device_count + 4)
        
        # Core components
        self.resource_pool = ResourcePool(gpu_manager)
        self.dependency_graph = DependencyGraph()
        self.tool_cache = ToolCache(Path("cache/tool_results"))
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.ready_queue = deque()  # Tasks ready to execute
        self.blocked_queue = deque()  # Tasks waiting for dependencies
        self.running_tasks: Dict[str, Future] = {}
        self.completed_tasks: Dict[str, Task] = {}
        
        # Thread pool for execution
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_tasks, thread_name_prefix="DependencyScheduler")
        
        # Control
        self.running = False
        self.scheduler_thread = None
        self.lock = threading.RLock()
        
        # Metrics
        self.start_time = None
        self.metrics = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "cache_hits": 0,
            "total_gpu_hours": 0.0,
            "total_cpu_hours": 0.0
        }
    
    def create_tool_execution_graph(self, combinations: Dict[str, Any]) -> List[ToolExecution]:
        """Create a graph of tool executions with dependencies"""
        tool_executions = []
        combination_stages = {}  # combination_id -> {stage -> [tool_executions]}
        
        # First pass: create all tool executions
        for combo_id, combination in combinations.items():
            combination_stages[combo_id] = defaultdict(list)
            
            for stage_name, tools in combination.tools_by_stage.items():
                for tool in tools:
                    # Create input hash for deduplication
                    input_data = {
                        "tool_name": tool.name,
                        "stage": stage_name,
                        "tool_config": tool.dict() if hasattr(tool, 'dict') else str(tool),
                        # Add dataset and previous stage outputs to hash
                        "dataset_hash": getattr(self.pipeline_executor, 'dataset_hash', 'unknown')
                    }
                    input_hash = hashlib.sha256(str(input_data).encode()).hexdigest()
                    
                    tool_exec = ToolExecution(
                        tool_name=tool.name,
                        stage=stage_name,
                        combination_id=combo_id,
                        input_hash=input_hash,
                        gpu_required=tool.name != "noop",  # noop doesn't need GPU
                        estimated_duration=self._estimate_tool_duration(tool.name, stage_name),
                        priority=self._get_tool_priority(tool.name, stage_name)
                    )
                    
                    tool_executions.append(tool_exec)
                    combination_stages[combo_id][stage_name].append(tool_exec)
        
        # Second pass: establish dependencies
        stage_order = ["pre_training", "during_training", "post_training", "deployment"]
        
        for combo_id, stages in combination_stages.items():
            prev_stage_tools = []
            
            for stage in stage_order:
                if stage in stages:
                    current_stage_tools = stages[stage]
                    
                    # Each tool in current stage depends on all tools from previous stage
                    for current_tool in current_stage_tools:
                        current_tool.dependencies = [t.task_id for t in prev_stage_tools]
                    
                    prev_stage_tools = current_stage_tools
        
        return tool_executions
    
    def _estimate_tool_duration(self, tool_name: str, stage: str) -> float:
        """Estimate tool execution duration in seconds"""
        # Basic heuristics - can be improved with historical data
        duration_map = {
            "noop": 30,
            "pre_xgbod": 120,
            "in_trades": 1800,  # 30 minutes for training
            "in_noop": 60,
            "post_fineprune": 600,  # 10 minutes
            "post_magnet": 900,     # 15 minutes
            "deploy_dp": 180,
            "dataset_inference": 240
        }
        return duration_map.get(tool_name, 300)  # Default 5 minutes
    
    def _get_tool_priority(self, tool_name: str, stage: str) -> TaskPriority:
        """Determine tool execution priority"""
        # noop tasks have lower priority
        if tool_name == "noop":
            return TaskPriority.LOW
        
        # Training tasks have high priority
        if "train" in tool_name or stage == "during_training":
            return TaskPriority.HIGH
        
        return TaskPriority.NORMAL
    
    def submit_combinations(self, combinations: Dict[str, Any]):
        """Submit combinations for scheduling"""
        logger.info(f"📋 Creating execution graph for {len(combinations)} combinations...")
        
        # Create tool execution graph
        tool_executions = self.create_tool_execution_graph(combinations)
        
        with self.lock:
            # Create tasks
            for tool_exec in tool_executions:
                task = Task(
                    task_id=tool_exec.task_id,
                    tool_execution=tool_exec,
                    dependencies=set(tool_exec.dependencies)
                )
                self.tasks[task.task_id] = task
                self.metrics["tasks_submitted"] += 1
                
                # Build dependency graph
                for dep_id in tool_exec.dependencies:
                    self.dependency_graph.add_dependency(task.task_id, dep_id)
            
            # Queue ready tasks
            for task in self.tasks.values():
                if self.dependency_graph.is_ready(task.task_id):
                    task.state = TaskState.QUEUED
                    self.ready_queue.append(task.task_id)
                else:
                    task.state = TaskState.BLOCKED
                    self.blocked_queue.append(task.task_id)
        
        logger.info(f"📊 Created {len(self.tasks)} tasks ({len(self.ready_queue)} ready, {len(self.blocked_queue)} blocked)")
    
    def start(self):
        """Start the scheduler"""
        if self.running:
            return
        
        self.running = True
        self.start_time = time.time()
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, name="DependencyScheduler")
        self.scheduler_thread.start()
        logger.info("🚀 Dependency-aware scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=30)
        self.executor.shutdown(wait=True)
        logger.info("🛑 Dependency-aware scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                self._schedule_tasks()
                self._process_completed_tasks()
                time.sleep(1)  # Scheduler tick
            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)
        
        logger.info("📈 Scheduler loop finished")
    
    def _schedule_tasks(self):
        """Schedule ready tasks to available resources"""
        with self.lock:
            # Sort ready queue by priority and creation time
            ready_tasks = [self.tasks[task_id] for task_id in self.ready_queue if task_id in self.tasks]
            ready_tasks.sort(key=lambda t: (t.tool_execution.priority.value, t.tool_execution.created_at), reverse=True)
            
            scheduled = []
            for task in ready_tasks:
                if len(self.running_tasks) >= self.max_concurrent_tasks:
                    break
                
                # Check cache first
                cached_result = self.tool_cache.get_cached_result(task.tool_execution.input_hash)
                if cached_result:
                    self._complete_task_from_cache(task, cached_result)
                    scheduled.append(task.task_id)
                    continue
                
                # Try to allocate resources
                if task.tool_execution.gpu_required:
                    gpu_id = self.resource_pool.allocate_gpu(task.task_id)
                    if gpu_id is not None:
                        task.assigned_gpu = gpu_id
                        self._execute_task(task)
                        scheduled.append(task.task_id)
                else:
                    worker_id = self.resource_pool.allocate_cpu_worker(task.task_id)
                    if worker_id is not None:
                        task.assigned_worker = str(worker_id)
                        self._execute_task(task)
                        scheduled.append(task.task_id)
            
            # Remove scheduled tasks from ready queue
            for task_id in scheduled:
                try:
                    self.ready_queue.remove(task_id)
                except ValueError:
                    pass
    
    def _execute_task(self, task: Task):
        """Execute a task"""
        task.state = TaskState.RUNNING
        task.started_at = time.time()
        
        future = self.executor.submit(self._run_task, task)
        self.running_tasks[task.task_id] = future
        
        logger.info(f"🔄 Started task {task.task_id} (GPU: {task.assigned_gpu}, Worker: {task.assigned_worker})")
    
    def _run_task(self, task: Task) -> Any:
        """Actually run the task execution"""
        try:
            # Set GPU environment if needed
            if task.assigned_gpu is not None:
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = str(task.assigned_gpu)
            
            # Execute the tool via pipeline executor
            result = self._execute_tool_via_pipeline(task.tool_execution)
            
            # Cache the result
            self.tool_cache.store_result(task.tool_execution.input_hash, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            raise
    
    def _execute_tool_via_pipeline(self, tool_exec: ToolExecution) -> Path:
        """Execute tool through the existing pipeline executor"""
        # This is a placeholder - integrate with your existing tool execution logic
        combination = self.pipeline_executor.combinations[tool_exec.combination_id]
        
        # Simulate tool execution for now
        time.sleep(min(tool_exec.estimated_duration / 10, 30))  # Faster simulation
        
        # Return a mock result path
        return Path(f"cache/tool_results/{tool_exec.task_id}")
    
    def _complete_task_from_cache(self, task: Task, result_path: Path):
        """Complete a task using cached results"""
        task.state = TaskState.COMPLETED
        task.completed_at = time.time()
        task.result = result_path
        
        self.completed_tasks[task.task_id] = task
        self.metrics["cache_hits"] += 1
        self.metrics["tasks_completed"] += 1
        
        # Unblock dependent tasks
        unblocked = self.dependency_graph.mark_completed(task.task_id)
        self._unblock_tasks(unblocked)
        
        logger.info(f"💾 Task {task.task_id} completed from cache")
    
    def _process_completed_tasks(self):
        """Process completed task futures"""
        completed_futures = []
        
        for task_id, future in self.running_tasks.items():
            if future.done():
                completed_futures.append(task_id)
        
        for task_id in completed_futures:
            future = self.running_tasks.pop(task_id)
            task = self.tasks[task_id]
            
            try:
                result = future.result()
                task.state = TaskState.COMPLETED
                task.completed_at = time.time()
                task.result = result
                self.completed_tasks[task_id] = task
                self.metrics["tasks_completed"] += 1
                
                # Calculate resource usage
                duration_hours = (task.completed_at - task.started_at) / 3600
                if task.assigned_gpu is not None:
                    self.metrics["total_gpu_hours"] += duration_hours
                else:
                    self.metrics["total_cpu_hours"] += duration_hours
                
                # Release resources
                if task.assigned_gpu is not None:
                    self.resource_pool.release_gpu(task.assigned_gpu, task_id)
                if task.assigned_worker is not None:
                    self.resource_pool.release_cpu_worker(int(task.assigned_worker), task_id)
                
                # Unblock dependent tasks
                unblocked = self.dependency_graph.mark_completed(task_id)
                self._unblock_tasks(unblocked)
                
                logger.info(f"✅ Task {task_id} completed successfully")
                
            except Exception as e:
                task.state = TaskState.FAILED
                task.error = str(e)
                self.metrics["tasks_failed"] += 1
                
                # Release resources
                if task.assigned_gpu is not None:
                    self.resource_pool.release_gpu(task.assigned_gpu, task_id)
                if task.assigned_worker is not None:
                    self.resource_pool.release_cpu_worker(int(task.assigned_worker), task_id)
                
                logger.error(f"❌ Task {task_id} failed: {e}")
    
    def _unblock_tasks(self, unblocked_task_ids: List[str]):
        """Move unblocked tasks from blocked to ready queue"""
        with self.lock:
            for task_id in unblocked_task_ids:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    if task.state == TaskState.BLOCKED:
                        task.state = TaskState.QUEUED
                        self.ready_queue.append(task_id)
                        try:
                            self.blocked_queue.remove(task_id)
                        except ValueError:
                            pass
                        logger.info(f"🔓 Unblocked task {task_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        with self.lock:
            elapsed_time = time.time() - (self.start_time or time.time())
            
            return {
                "running": self.running,
                "elapsed_time": elapsed_time,
                "total_tasks": len(self.tasks),
                "ready_tasks": len(self.ready_queue),
                "blocked_tasks": len(self.blocked_queue),
                "running_tasks": len(self.running_tasks),
                "completed_tasks": len(self.completed_tasks),
                "resource_status": self.resource_pool.get_resource_status(),
                "metrics": self.metrics.copy(),
                "throughput": {
                    "tasks_per_hour": self.metrics["tasks_completed"] / max(elapsed_time / 3600, 0.001),
                    "gpu_utilization": (self.metrics["total_gpu_hours"] / max(elapsed_time / 3600, 0.001)) / self.gpu_manager.device_count,
                    "cache_hit_rate": self.metrics["cache_hits"] / max(self.metrics["tasks_completed"], 1)
                }
            }
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all tasks to complete"""
        start_time = time.time()
        
        while self.running:
            with self.lock:
                if (len(self.ready_queue) == 0 and 
                    len(self.running_tasks) == 0 and 
                    len(self.blocked_queue) == 0):
                    logger.info("🎉 All tasks completed!")
                    return True
            
            if timeout and (time.time() - start_time) > timeout:
                logger.warning("⏰ Timeout waiting for task completion")
                return False
            
            time.sleep(5)
        
        return True
