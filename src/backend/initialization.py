"""
Backend initialization module for Landseer.

This module handles the initialization of the backend, including:
- Loading tool registry from tools.yaml
- Loading pipeline configuration
- Creating Pipeline instances
"""

from pathlib import Path
from typing import Optional

from ..common import get_logger
from ..pipeline.tools import init_tool_registry
from ..pipeline.config_loader import create_pipeline_from_config
from ..pipeline.pipeline import Pipeline

logger = get_logger(__name__)


class BackendContext:
    """
    Backend context that holds the pipeline and configuration.
    
    This is initialized when the backend starts and provides access
    to the loaded pipeline and tools.
    """
    
    def __init__(
        self,
        pipeline: Pipeline,
        tools_config_path: str,
        pipeline_config_path: str
    ):
        """
        Initialize backend context.
        
        Args:
            pipeline: Loaded pipeline instance
            tools_config_path: Path to tools.yaml
            pipeline_config_path: Path to pipeline config YAML
        """
        self.pipeline = pipeline
        self.tools_config_path = tools_config_path
        self.pipeline_config_path = pipeline_config_path
        logger.info(f"Backend context initialized with pipeline: {pipeline.name}")
    
    def reload_pipeline(self) -> None:
        """Reload the pipeline from configuration files."""
        logger.info("Reloading pipeline configuration...")
        self.pipeline = create_pipeline_from_config(
            self.pipeline_config_path,
            self.tools_config_path
        )
        logger.info("Pipeline reloaded successfully")


def initialize_backend(
    tools_config_path: Optional[str] = None,
    pipeline_config_path: Optional[str] = None,
    default_pipeline: str = "trades"
) -> BackendContext:
    """
    Initialize the backend by loading tools and pipeline configuration.
    
    This function is called when the backend starts up. It:
    1. Loads the tool registry from tools.yaml
    2. Loads the pipeline configuration
    3. Creates a Pipeline instance with all workflows
    4. Returns a BackendContext for use by the backend
    
    Args:
        tools_config_path: Path to tools.yaml (defaults to configs/tools.yaml)
        pipeline_config_path: Path to pipeline config (defaults to configs/pipeline/{default_pipeline}.yaml)
        default_pipeline: Name of default pipeline to load if path not specified
        
    Returns:
        BackendContext with loaded pipeline
        
    Raises:
        FileNotFoundError: If configuration files are not found
        ValueError: If configuration is invalid
    """
    # Set default paths if not provided
    if tools_config_path is None:
        tools_config_path = "configs/tools.yaml"
    
    if pipeline_config_path is None:
        pipeline_config_path = f"configs/pipeline/{default_pipeline}.yaml"
    
    # Convert to absolute paths
    tools_config_path = str(Path(tools_config_path).resolve())
    pipeline_config_path = str(Path(pipeline_config_path).resolve())
    
    logger.info("="*60)
    logger.info("Initializing Landseer Backend")
    logger.info("="*60)
    logger.info(f"Tools config: {tools_config_path}")
    logger.info(f"Pipeline config: {pipeline_config_path}")
    
    # Initialize tool registry
    try:
        init_tool_registry(tools_config_path)
        logger.info("Tool registry initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize tool registry: {e}")
        raise
    
    # Load pipeline configuration and create pipeline
    try:
        pipeline = create_pipeline_from_config(
            pipeline_config_path,
            tools_config_path
        )
        logger.info(f"Pipeline '{pipeline.name}' loaded with {len(pipeline.workflows)} workflows")
    except Exception as e:
        logger.error(f"Failed to load pipeline configuration: {e}")
        raise
    
    # Create backend context
    context = BackendContext(
        pipeline=pipeline,
        tools_config_path=tools_config_path,
        pipeline_config_path=pipeline_config_path
    )
    
    logger.info("="*60)
    logger.info("Backend initialization complete")
    logger.info("="*60)
    
    return context


# Global backend context (initialized on first backend startup)
_backend_context: Optional[BackendContext] = None


def get_backend_context() -> Optional[BackendContext]:
    """
    Get the global backend context.
    
    Returns:
        BackendContext if initialized, None otherwise
    """
    return _backend_context


def set_backend_context(context: BackendContext) -> None:
    """
    Set the global backend context.
    
    Args:
        context: BackendContext to set as global
    """
    global _backend_context
    _backend_context = context
    logger.debug("Global backend context updated")
