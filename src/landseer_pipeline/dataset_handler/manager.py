import logging
import os
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any
import requests
import json

from landseer_pipeline.dataset_handler.poisoned import prepare_backdoor_dataset
from landseer_pipeline.dataset_handler.cache import CacheManager

logger = logging.getLogger(__name__)

DATASET_LOADER_FOLDER = os.path.abspath("dataset_handler/loaders")

class DatasetManager:
    """Handles dataset preparation and format conversion"""
    def __init__(self, settings):
        self.settings = settings
        self.data_dir = settings.data_dir
        self.config = settings.config
        self.attack_config = settings.attacks
        self.active_attacks = self.attack_config.attacks

        self.dataset_name = self.dataset_config.name
        self.loader_module = self.dataset_config.loader_module
        base_dir = self.data_dir        
        self.dataset_dir = base_dir / self.dataset_name

        self.cache_manager = CacheManager(settings)
        if self.cache_manager.is_cached(self.dataset_name) and settings.use_cache:
            self.is_cached = True
            self.cache = self.cache_manager.get_cache_path(self.dataset_name)
        else:
            self.is_cached = False
            self.cache = None
        
        self.clean_dataset_dir = self.dataset_dir / "clean"
        self.poisoned_dataset_dir = self.dataset_dir / "poisoned"
        self.dataset_meta_file = self.dataset_dir / "dataset_meta.json"

    @property
    def dataset_config(self):
        """Returns the dataset configuration"""
        return self.config.dataset
    
    @property
    def dataset_dir_source(self) -> Path:
        target_file = Path(self.dataset_dir) / "extracted_dataset"
        return Path(target_file / os.listdir(target_file)[0])

    def prepare_dataset(self):
        logger.info(f"Preparing dataset ...")

        if not self.is_cached:
            logger.info(f"Dataset {self.dataset_name} not cached, downloading and preparing...")
            self.download_and_create_clean_dataset()
        else:
            logger.info(f"Dataset {self.dataset_name} loaded from cache")
        
        # Determine if poisoned variant needed: either config.variant == "poisoned" or active backdoor attack
        needs_poison = (self.dataset_config.variant.lower() == "poisoned") #or (self.active_attacks and getattr(self.active_attacks, 'backdoor', False))
        if needs_poison and not os.path.exists(self.poisoned_dataset_dir):
            if not self.cache_manager.is_poisoned_cached(self.dataset_name):
                logging.info(f"Preparing poisoned dataset variant...")
                os.makedirs(self.poisoned_dataset_dir, exist_ok=True)
            poison_params = self._extract_poison_params()
            self.backdoor_dataset_dir = prepare_backdoor_dataset(self.clean_dataset_dir, self.poisoned_dataset_dir, **poison_params)

    def download_and_create_clean_dataset(self):
        """Downloads the dataset and creates the clean dataset directory"""
        logger.info(f"Downloading dataset {self.dataset_name} ...")
        try:
            self.load_dataset_func = getattr(self.loader_module, "load_dataset")
            download_directory = self.dataset_dir / "downloaded_dataset"
            clean_directory = self.clean_dataset_dir
            os.makedirs(download_directory, exist_ok=True)
            os.makedirs(clean_directory, exist_ok=True)
            params = self.dataset_config.params or {}
            metadata = self.load_dataset_func(
                output_dir=str(clean_directory),
                download_dir=str(download_directory),
            ) or {}
            # Persist metadata merged with params & config subset
            enriched_meta = {
                "name": self.dataset_name,
                "version": self.dataset_config.version,
                "variant": self.dataset_config.variant,
                "params": params,
                "metadata": metadata
            }
            try:
                self.dataset_meta_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.dataset_meta_file, 'w') as f:
                    json.dump(enriched_meta, f, indent=2)
            except Exception as meta_e:
                logger.warning(f"Failed to write dataset metadata file: {meta_e}")
            self.cache_manager.mark_as_success(self.dataset_name)
        except AttributeError as e:
            logger.error(f"Failed to load dataset loader module: {e}")
            raise

    def _extract_poison_params(self) -> Dict[str, Any]:
        params = {}
        dc_params = self.dataset_config.params or {}
        for key in ["poison_fraction", "target_class", "trigger_size", "trigger_value"]:
            if key in dc_params:
                params[key] = dc_params[key]
        return params

    def _convert_to_ir(self, files: List, source_format: str, target_path: str):
        logger.info(
            f"Converting dataset from {source_format} to numpy format...")

        try:
            if source_format == "pickle":
                self.convert_pickle_to_numpy_files(target_path, files)
            elif source_format == "mat":
                self.convert_mat_to_numpy_files(target_path, files)
            else:
                logger.warning(
                    f"Conversion from {source_format} to numpy not implemented")

            logger.info(f"Dataset converted to IR format at {target_path}")

        except Exception as e:
            logger.error(f"Failed to convert dataset to IR format: {e}")
            raise