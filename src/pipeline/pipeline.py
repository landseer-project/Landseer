"""
Pipeline definitions for the Landseer infrastructure.

A pipeline is a set of workflows used to evaluate ML defenses.
Pipelines manage the execution of multiple workflow combinations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .workflow import Workflow
from .tasks import generate_pipeline_id


@dataclass
class Pipeline(ABC):
    """
    Abstract base class for pipelines.
    
    A pipeline contains multiple workflows (combinations) and manages their execution.
    
    Attributes:
        id: Unique pipeline identifier (e.g., "pipeline_1")
        name: Pipeline name/identifier
        workflows: List of workflows in the pipeline
        config: Pipeline configuration
        dataset: Dataset configuration (name, variant, params)
        model: Model configuration (script, framework, params)
    """
    name: str
    workflows: List[Workflow] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    dataset: Optional[Dict[str, Any]] = field(default=None)
    model: Optional[Dict[str, Any]] = field(default=None)
    id: str = field(default="", init=False)
    
    def __post_init__(self):
        """Initialize pipeline with unique ID."""
        if not self.id:
            self.id = generate_pipeline_id()
        # Set pipeline_id for all workflows
        for workflow in self.workflows:
            workflow.pipeline_id = self.id
    
    @abstractmethod
    def run(self, data: Any = None) -> Any:
        """
        Execute the pipeline.
        
        Args:
            data: Initial input data
            
        Returns:
            Pipeline execution results
        """
        pass
    
    def add_workflow(self, workflow: Workflow) -> None:
        """
        Add a workflow to the pipeline.
        
        Args:
            workflow: Workflow to add
        """
        workflow.pipeline_id = self.id
        self.workflows.append(workflow)
    
    def get_workflow(self, name: str) -> Optional[Workflow]:
        """
        Get a workflow by name.
        
        Args:
            name: Workflow name
            
        Returns:
            Workflow if found, None otherwise
        """
        for workflow in self.workflows:
            if workflow.name == name:
                return workflow
        return None
    
    def __repr__(self) -> str:
        """String representation of the pipeline."""
        return f"Pipeline(name='{self.name}', workflows={len(self.workflows)})"


@dataclass
class DefenseEvaluationPipeline(Pipeline):
    """
    Concrete pipeline implementation for ML defense evaluation.
    
    This pipeline executes multiple workflow combinations to evaluate
    different ML defense strategies.
    """
    
    def run(self, data: Any = None) -> Dict[str, Any]:
        """
        Execute all workflows in the pipeline.
        
        Args:
            data: Initial input data
            
        Returns:
            Dictionary mapping workflow names to their results
        """
        results = {}
        for workflow in self.workflows:
            try:
                result = workflow.run(data)
                results[workflow.name] = {
                    "status": "success",
                    "result": result
                }
            except Exception as e:
                results[workflow.name] = {
                    "status": "failed",
                    "error": str(e)
                }
        return results
    
    def run_single_workflow(self, workflow_name: str, data: Any = None) -> Any:
        """
        Execute a single workflow by name.
        
        Args:
            workflow_name: Name of the workflow to run
            data: Initial input data
            
        Returns:
            Workflow execution result
            
        Raises:
            ValueError: If workflow not found
        """
        workflow = self.get_workflow(workflow_name)
        if workflow is None:
            available = [w.name for w in self.workflows]
            raise ValueError(
                f"Workflow '{workflow_name}' not found. "
                f"Available workflows: {', '.join(available)}"
            )
        return workflow.run(data)


class PipelineFactory:
    """Factory for creating pipeline instances."""
    
    @classmethod
    def create_pipeline(
        cls,
        name: str,
        workflows: Optional[List[Workflow]] = None,
        config: Optional[Dict[str, Any]] = None,
        dataset: Optional[Dict[str, Any]] = None,
        model: Optional[Dict[str, Any]] = None,
        pipeline_type: str = "defense_evaluation"
    ) -> Pipeline:
        """
        Create a pipeline instance.
        
        Args:
            name: Pipeline name
            workflows: List of workflows
            config: Pipeline configuration
            dataset: Dataset configuration
            model: Model configuration
            pipeline_type: Type of pipeline to create
            
        Returns:
            Created pipeline instance
            
        Raises:
            ValueError: If unknown pipeline type
        """
        if pipeline_type == "defense_evaluation":
            return DefenseEvaluationPipeline(
                name=name,
                workflows=workflows or [],
                config=config or {},
                dataset=dataset,
                model=model
            )
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}") 
