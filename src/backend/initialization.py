"""
Backend initialization module for Landseer.

This module handles the initialization of the backend, including:
- Loading tool registry from tools.yaml
- Loading pipeline configuration
- Preparing datasets (host-executed, before scheduling)
- Creating Pipeline instances
- Initializing database for persistence
- Setting up AI store (MinIO) connection
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..common import get_logger
from ..pipeline.tools import init_tool_registry
from ..pipeline.config_loader import create_pipeline_from_config, load_pipeline_config
from ..pipeline.pipeline import Pipeline

logger = get_logger(__name__)

# Import data module for dataset preparation
try:
    from ..data import DatasetManager, DatasetInfo
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False
    DatasetManager = None
    DatasetInfo = None

# Import optional database and store modules
try:
    from .db_service import DatabaseService, init_db_service, get_db_service
    from ..db import DatabaseConfig
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    DatabaseService = None
    DatabaseConfig = None

try:
    from ..store import MinioStore, MinioConfig, init_store
    STORE_AVAILABLE = True
except ImportError:
    STORE_AVAILABLE = False
    MinioStore = None
    MinioConfig = None


class BackendContext:
    """
    Backend context that holds the pipeline and configuration.
    
    This is initialized when the backend starts and provides access
    to the loaded pipeline, tools, database, store, and dataset info.
    """
    
    def __init__(
        self,
        pipeline: Pipeline,
        tools_config_path: str,
        pipeline_config_path: str,
        db_service: Optional["DatabaseService"] = None,
        store: Optional["MinioStore"] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        dataset_manager: Optional["DatasetManager"] = None
    ):
        """
        Initialize backend context.
        
        Args:
            pipeline: Loaded pipeline instance
            tools_config_path: Path to tools.yaml
            pipeline_config_path: Path to pipeline config YAML
            db_service: Database service instance
            store: MinIO store instance
            dataset_info: Information about prepared dataset
            dataset_manager: DatasetManager instance
        """
        self.pipeline = pipeline
        self.tools_config_path = tools_config_path
        self.pipeline_config_path = pipeline_config_path
        self.db_service = db_service
        self.store = store
        self.dataset_info = dataset_info
        self.dataset_manager = dataset_manager
        logger.info(f"Backend context initialized with pipeline: {pipeline.name}")
    
    def reload_pipeline(self) -> None:
        """Reload the pipeline from configuration files."""
        logger.info("Reloading pipeline configuration...")
        self.pipeline = create_pipeline_from_config(
            self.pipeline_config_path,
            self.tools_config_path
        )
        logger.info("Pipeline reloaded successfully")
    
    def get_dataset_path(self) -> Optional[Path]:
        """Get path to the prepared dataset."""
        if self.dataset_info and "output_dir" in self.dataset_info:
            return Path(self.dataset_info["output_dir"])
        return None


def initialize_backend(
    tools_config_path: Optional[str] = None,
    pipeline_config_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    default_pipeline: str = "trades",
    enable_db: bool = True,
    enable_store: bool = True,
    prepare_dataset: bool = True
) -> BackendContext:
    """
    Initialize the backend by loading tools and pipeline configuration.
    
    This function is called when the backend starts up. It:
    1. Loads the tool registry from tools.yaml
    2. Prepares dataset on host (before any tasks are scheduled)
    3. Loads the pipeline configuration
    4. Creates a Pipeline instance with all workflows
    5. Uploads dataset to MinIO for workers
    6. Returns a BackendContext for use by the backend
    
    Dataset preparation is done on the host to ensure:
    - Single copy of dataset (no race conditions between workers)
    - Dataset is validated before scheduling any tasks
    - Workers just mount from shared storage (read-only)
    
    Args:
        tools_config_path: Path to tools.yaml (defaults to configs/tools.yaml)
        pipeline_config_path: Path to pipeline config (defaults to configs/pipeline/{default_pipeline}.yaml)
        data_dir: Base directory for datasets (defaults to ./data)
        default_pipeline: Name of default pipeline to load if path not specified
        enable_db: Enable database persistence
        enable_store: Enable MinIO store
        prepare_dataset: Prepare dataset before creating pipeline
        
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
    
    if data_dir is None:
        data_dir = "./data"
    
    # Convert to absolute paths
    tools_config_path = str(Path(tools_config_path).resolve())
    pipeline_config_path = str(Path(pipeline_config_path).resolve())
    data_dir = str(Path(data_dir).resolve())
    
    logger.info("="*60)
    logger.info("Initializing Landseer Backend")
    logger.info("="*60)
    logger.info(f"Tools config: {tools_config_path}")
    logger.info(f"Pipeline config: {pipeline_config_path}")
    logger.info(f"Data directory: {data_dir}")
    
    # Initialize tool registry
    try:
        init_tool_registry(tools_config_path)
        logger.info("Tool registry initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize tool registry: {e}")
        raise
    
    # Initialize MinIO store early (needed for dataset upload)
    store = None
    if enable_store and STORE_AVAILABLE:
        try:
            store = init_store()
            if store.is_available:
                logger.info(f"MinIO store connected: {store.config.endpoint}")
            else:
                logger.warning("MinIO store not available")
        except Exception as e:
            logger.warning(f"Failed to initialize MinIO store: {e}")
    elif not STORE_AVAILABLE:
        logger.info("Store module not available")
    
    # Prepare dataset on host (before scheduling any tasks)
    dataset_info = None
    dataset_manager = None
    if prepare_dataset and DATA_AVAILABLE:
        try:
            # Load pipeline config to get dataset info
            config = load_pipeline_config(pipeline_config_path)
            
            # Create dataset manager
            dataset_manager = DatasetManager(Path(data_dir))
            
            # Extract poisoning config if variant is poisoned
            poisoning = None
            if config.dataset.variant == "poisoned":
                poisoning = config.dataset.params.get("poisoning")
            
            # Prepare dataset
            logger.info(f"Preparing dataset: {config.dataset.name}/{config.dataset.variant}")
            ds_info = dataset_manager.prepare_dataset(
                name=config.dataset.name,
                variant=config.dataset.variant,
                poisoning=poisoning,
                **config.dataset.params
            )
            
            if ds_info:
                dataset_info = ds_info.to_dict()
                logger.info(f"Dataset prepared: {ds_info.train_samples} train, {ds_info.test_samples} test")
                
                # Upload to MinIO for workers
                if store and store.is_available:
                    dataset_key = f"datasets/{config.dataset.name}/{config.dataset.variant}"
                    try:
                        store.upload_directory(ds_info.output_dir, dataset_key)
                        dataset_info["minio_key"] = dataset_key
                        logger.info(f"Dataset uploaded to MinIO: {dataset_key}")
                    except Exception as e:
                        logger.warning(f"Failed to upload dataset to MinIO: {e}")
            else:
                logger.warning("Dataset preparation returned no info")
                
        except Exception as e:
            logger.warning(f"Failed to prepare dataset: {e}")
            import traceback
            traceback.print_exc()
    elif not DATA_AVAILABLE:
        logger.info("Data module not available, skipping dataset preparation")
    
    # Load pipeline configuration and create pipeline
    try:
        pipeline = create_pipeline_from_config(
            pipeline_config_path,
            tools_config_path
        )
        
        # Add dataset info to pipeline config
        if dataset_info:
            pipeline.config["dataset_info"] = dataset_info
            pipeline.config["dataset_path"] = dataset_info.get("output_dir")
            if "minio_key" in dataset_info:
                pipeline.config["dataset_minio_key"] = dataset_info["minio_key"]
        
        logger.info(f"Pipeline '{pipeline.name}' loaded with {len(pipeline.workflows)} workflows")
    except Exception as e:
        logger.error(f"Failed to load pipeline configuration: {e}")
        raise
    
    # Initialize database service
    db_service = None
    if enable_db and DB_AVAILABLE:
        try:
            db_service = init_db_service(enabled=True)
            if db_service.is_available():
                logger.info("Database service initialized")
                # Sync pipeline to database
                db_service.sync_pipeline_to_db(pipeline)
                logger.info("Pipeline synced to database")
        except Exception as e:
            logger.warning(f"Failed to initialize database service: {e}")
    elif not DB_AVAILABLE:
        logger.info("Database module not available, persistence disabled")
    
    # Create backend context
    context = BackendContext(
        pipeline=pipeline,
        tools_config_path=tools_config_path,
        pipeline_config_path=pipeline_config_path,
        db_service=db_service,
        store=store,
        dataset_info=dataset_info,
        dataset_manager=dataset_manager
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
