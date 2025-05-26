import logging
import os
from pathlib import Path
from collections import Counter
from typing import Dict, List
import requests
import pickle
import numpy as np
import scipy.io
import importlib

logger = logging.getLogger("defense_pipeline")

DATASET_LOADER_FOLDER = os.path.abspath("dataset_handler/loaders")

class DatasetManager:
    """Handles dataset preparation and format conversion"""
    def __init__(self, Stager):
        self.data_dir = Path("./data")
        self.data_dir.mkdir(exist_ok=True)

        self.config = Stager.config
        self.dataset_name = self.dataset_config.name
        self.loader_module = self.dataset_config.loader_module
        
        self.dataset_dir = self.data_dir / self.dataset_name
        self.cache = None
        if self.dataset_dir.exists():
            #iterate over the files in the directory
            self.cache = os.listdir(self.dataset_dir)
        self.clean_dataset_dir = os.path.join(str(self.dataset_dir), "clean")
        self.poisoned_dataset_dir = os.path.join(str(self.dataset_dir), "poisoned")

    @property
    def dataset_config(self):
        """Returns the dataset configuration"""
        return self.config.dataset

    def prepare_dataset(self) -> str:
        logger.info(f"Preparing dataset ...")

        #download the dataset
        if self.cache is None:
            self._download_dataset()
        #first prepare the clean dataset
        
        if "clean" not in self.cache:
            self.prepare_clean_dataset()

        #check if the self.attacks have backdoor as true
        if self.config.attacks is not None:
            if self.config.attacks.backdoor and "poisoned" not in self.cache:
                self.backdoor_dataset_dir = self.prepare_backdoor_dataset()

    def prepare_clean_dataset(self) -> str:
        logger.info(f"Preparing dataset ...")
        try:
            self.convert_pickle_to_numpy_files = getattr(self.loader_module, "convert_pickle_to_numpy_files")
            self.convert_mat_to_numpy_files = getattr(self.loader_module, "convert_mat_to_numpy_files")
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(f"Could not find loader for dataset '{self.dataset_name}': {e}")

        dataset_type, files = self.detect_dataset_type_by_magic(self.dataset_dir)
        if dataset_type != "numpy":
            self._convert_to_ir(self.dataset_dir, files,
                                    dataset_type, self.clean_dataset_dir)

        logger.info(f"Clean dataset prepared at {self.clean_dataset_dir}")
        return 

    def is_pickle_file(self, file_path: Path) -> bool:
        try:
            with open(file_path, 'rb') as f:
                pickle.load(f, encoding='bytes')
            return True
        except Exception as e:
            return False

    def detect_dataset_type_by_magic(self, dataset_dir: Path):
        file_types = {}
        logger.info(
            "Scanning dataset directory for file types: %s", dataset_dir)
        for file in dataset_dir.iterdir():
            if file.is_file():
                try:
                    with open(file, "rb") as f:
                        header = f.read(8)

                        if header.startswith(b'\x93NUMPY'):
                            file_types[file.resolve()] = "numpy"
                        elif header.startswith(b'\x89HDF\r\n\x1a\n'):
                            file_types[file.resolve()] = "h5"
                        elif self.is_pickle_file(file):
                            file_types[file.resolve()] = "pickle"
                        elif header[:4].decode(errors="ignore").isprintable():
                            file_types[file.resolve()] = "mat"
                        else:
                            file_types[file.resolve()] = "unknown"
                except Exception as e:
                    logger.warning(f"Failed to read file {file.name}: {e}")

        if file_types:
            type_counts = Counter(file_types.values())
            majority_type = type_counts.most_common(1)[0][0]
            files_of_majority_type = [
                str(path) for path, ftype in file_types.items() if ftype == majority_type]

            logger.info(
                f"Detected file type distribution: {dict(type_counts)}")
            logger.info(f"Majority dataset type: {majority_type}")
            # logger.debug(f"Files of majority type: {files_of_majority_type}")

            return majority_type, files_of_majority_type
        else:
            logger.warning("No files found to detect dataset type.")
            return "unknown", []

    def _download_dataset(self):
        logger.info(f"Downloading dataset from {url}...")
        try:
            from tqdm import tqdm

            url = self.dataset_config.link
            target_dir = self.dataset_dir

            response = requests.get(url, stream=True)
            response.raise_for_status()

            filename = url.split("/")[-1]
            target_file = target_dir / filename
            if not target_file.exists():
                total_size = int(response.headers.get("content-length", 0))
                with open(target_file, "wb") as f, tqdm(desc=filename, total=total_size, unit="B", unit_scale=True, unit_divisor=1024,) as progress:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        progress.update(size)

                logger.info(f"Downloaded dataset to {target_file}")

            extracted_dir = filename
            if filename.endswith((".tar.gz", ".tgz")):
                extracted_dir = self._extract_tar_gz(target_file, target_dir)
            elif filename.endswith(".zip"):
                extracted_dir = self._extract_zip(target_file, target_dir)
            return extracted_dir

        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise

    def _extract_tar_gz(self, archive_path: Path, target_dir: Path):
        logger.info(f"Extracting {archive_path}...")

        try:
            import tarfile
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=target_dir)
                top_level = [member.name.split(
                    '/')[0] for member in tar.getmembers() if member.name and '/' in member.name]
                top_level = list(set(top_level))
            logger.info(f"Extracted to {target_dir}")
            if len(top_level) == 1:
                return target_dir / top_level[0]
            return target_dir
        except Exception as e:
            logger.error(f"Failed to extract tar.gz file: {e}")
            raise

    def _extract_zip(self, archive_path: Path, target_dir: Path):
        logger.info(f"Extracting {archive_path}...")

        try:
            import zipfile
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(target_dir)
            logger.info(f"Extracted to {target_dir}")

            top_level = [Path(f.filename).parts[0] for f in zip_ref.infolist() if len(
                Path(f.filename).parts) > 1]
            top_level = list(set(top_level))

            if len(top_level) == 1:
                return target_dir / top_level[0]
            return target_dir
        except Exception as e:
            logger.error(f"Failed to extract zip file: {e}")
            raise

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

    def prepare_backdoor_dataset(self) -> str:
        logger.info(f"Preparing backdoor dataset ...")
        #open the clean dataset numpy files

        #then execute the poisoning function on them

