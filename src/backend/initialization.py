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
_DATA_IMPORT_ERROR: Optional[BaseException] = None
try:
    from ..data import DatasetManager, DatasetInfo
    DATA_AVAILABLE = True
except ImportError as e:
    DATA_AVAILABLE = False
    _DATA_IMPORT_ERROR = e
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
        pipeline: Optional[Pipeline],
        tools_config_path: str,
        pipeline_config_path: Optional[str] = None,
        db_service: Optional["DatabaseService"] = None,
        store: Optional["MinioStore"] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        dataset_manager: Optional["DatasetManager"] = None
    ):
        self.pipeline = pipeline
        self.tools_config_path = tools_config_path
        self.pipeline_config_path = pipeline_config_path
        self.db_service = db_service
        self.store = store
        self.dataset_info = dataset_info
        self.dataset_manager = dataset_manager
        if pipeline is not None:
            logger.info(f"Backend context initialized with pipeline: {pipeline.name}")
        else:
            logger.info("Backend context initialized (no pipeline — headless mode)")
    
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
    enable_db: bool = True,
    enable_store: bool = True,
    prepare_dataset: bool = True
) -> BackendContext:
    """
    Initialize the backend by loading tools and (optionally) a pipeline.

    When ``pipeline_config_path`` is ``None`` the backend starts in
    *headless* mode: registries, DB and store are initialised but no
    pipeline is loaded.  Runs can then be triggered via the API.
    """
    if tools_config_path is None:
        tools_config_path = "configs/tools.yaml"

    if data_dir is None:
        data_dir = "./data"

    tools_config_path = str(Path(tools_config_path).resolve())
    data_dir = str(Path(data_dir).resolve())

    headless = pipeline_config_path is None
    if not headless:
        pipeline_config_path = str(Path(pipeline_config_path).resolve())

    logger.info("=" * 60)
    logger.info("Initializing Landseer Backend")
    logger.info("=" * 60)
    logger.info(f"Tools config: {tools_config_path}")
    logger.info(f"Pipeline config: {pipeline_config_path or '(none — headless mode)'}")
    logger.info(f"Data directory: {data_dir}")

    # -- Tool registry ---------------------------------------------------------
    try:
        init_tool_registry(tools_config_path)
        logger.info("Tool registry initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize tool registry: {e}")
        raise

    # -- MinIO store -----------------------------------------------------------
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

    # -- Dataset preparation (only when a config is supplied) ------------------
    dataset_info = None
    dataset_manager = None
    if not headless and prepare_dataset and DATA_AVAILABLE:
        try:
            config = load_pipeline_config(pipeline_config_path)
            dataset_manager = DatasetManager(Path(data_dir))

            poisoning = None
            if config.dataset.variant == "poisoned":
                poisoning = config.dataset.params.get("poisoning")

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

                if store and store.is_available:
                    # Use the actual output directory name (e.g. "poisoned_a3f9c2b1" for
                    # poisoned datasets) so different poisoning configs get distinct keys
                    # and never overwrite each other on MinIO.
                    dir_suffix = Path(ds_info.output_dir).name  # "clean" or "poisoned_<hash>"
                    dataset_key = f"datasets/{config.dataset.name}/{dir_suffix}"
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
        logger.warning(
            "Data module not available, skipping dataset preparation "
            "(install landseer with numpy/torch/torchvision): %s",
            _DATA_IMPORT_ERROR,
        )

    # -- Pipeline (skip in headless mode) -------------------------------------
    pipeline = None
    if not headless:
        try:
            pipeline = create_pipeline_from_config(
                pipeline_config_path,
                tools_config_path
            )
            if dataset_info:
                pipeline.config["dataset_info"] = dataset_info
                pipeline.config["dataset_path"] = dataset_info.get("output_dir")
                if "minio_key" in dataset_info:
                    pipeline.config["dataset_minio_key"] = dataset_info["minio_key"]
            logger.info(f"Pipeline '{pipeline.name}' loaded with {len(pipeline.workflows)} workflows")
        except Exception as e:
            logger.error(f"Failed to load pipeline configuration: {e}")
            raise

    # -- Database service ------------------------------------------------------
    db_service = None
    if enable_db and DB_AVAILABLE:
        try:
            db_service = init_db_service(enabled=True)
            if db_service.is_available():
                logger.info("Database service initialized")
                if pipeline is not None:
                    db_service.sync_pipeline_to_db(pipeline)
                    logger.info("Pipeline synced to database")
        except Exception as e:
            logger.warning(f"Failed to initialize database service: {e}")
    elif not DB_AVAILABLE:
        logger.info("Database module not available, persistence disabled")

    # -- Context ---------------------------------------------------------------
    context = BackendContext(
        pipeline=pipeline,
        tools_config_path=tools_config_path,
        pipeline_config_path=pipeline_config_path,
        db_service=db_service,
        store=store,
        dataset_info=dataset_info,
        dataset_manager=dataset_manager
    )

    logger.info("=" * 60)
    logger.info("Backend initialization complete")
    logger.info("=" * 60)

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
