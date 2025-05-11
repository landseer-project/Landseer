"""
Dataset management for ML Defense Pipeline
"""
import logging
import os
from pathlib import Path
from collections import Counter
from typing import Dict, List
import requests
import pickle
import numpy as np
import scipy.io


logger = logging.getLogger("defense_pipeline")


class DatasetManager:
    """Handles dataset preparation and format conversion"""

    def __init__(self, data_dir: str = "./data"):
        """
        Initialize with the path to store datasets

        Args:
            data_dir: Directory to store datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def prepare_dataset(self, dataset_name: str, dataset_info: Dict) -> str:
        """
        Prepare a dataset for the pipeline

        Args:
            dataset_name: Name of the dataset
            dataset_info: Dataset metadata from configuration

        Returns:
            Path to the prepared dataset directory
        """
        logger.info(f"Preparing dataset '{dataset_name}'...")
        dataset_dir = self.data_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        if "link" in dataset_info.get(dataset_name).keys():
            link = dataset_info.get(dataset_name).get("link")
            dataset_dir = self._download_dataset(link, dataset_dir)

            # guess the format of downloaded dataset
            dataset_type, files = self.detect_dataset_type_by_magic(
                dataset_dir)
            target_dir = os.path.join(str(dataset_dir)+".numpy")
            if dataset_type != "numpy":
                self._convert_to_ir(dataset_dir, files,
                                    dataset_type, target_dir)

        logger.info(f"Dataset prepared at {target_dir}")
        return str(target_dir)

    def is_pickle_file(self, file_path: Path) -> bool:
        try:
            with open(file_path, 'rb') as f:
                pickle.load(f, encoding='bytes')
            return True
        except Exception as e:
            return False

    def detect_dataset_type_by_magic(self, dataset_dir: Path):
        """
        Detect the majority file type in a dataset directory using magic numbers.

        Args:
            dataset_dir: Path to the dataset directory

        Returns:
            A tuple of (majority file type as a string, list of filenames of that type)
        """
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
            logger.info(f"Files of majority type: {files_of_majority_type}")

            return majority_type, files_of_majority_type
        else:
            logger.warning("No files found to detect dataset type.")
            return "unknown", []

    def _download_dataset(self, url: str, target_dir: Path):
        """
        Download a dataset from the given URL

        Args:
            url: URL to download from
            target_dir: Directory to save the downloaded file
        """
        logger.info(f"Downloading dataset from {url}...")

        try:
            import requests
            from tqdm import tqdm

            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Extract filename from URL
            filename = url.split("/")[-1]
            target_file = target_dir / filename
            if not target_file.exists():
                total_size = int(response.headers.get("content-length", 0))
                with open(target_file, "wb") as f, tqdm(desc=filename, total=total_size, unit="B", unit_scale=True, unit_divisor=1024,) as progress:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        progress.update(size)

                logger.info(f"Downloaded dataset to {target_file}")

            # Handle compressed files
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
        """
        Extract a tar.gz file

        Args:
            archive_path: Path to the archive
            target_dir: Directory to extract to
        """
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
        """
        Extract a zip file

        Args:
            archive_path: Path to the archive
            target_dir: Directory to extract to
        """
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

    def _convert_to_ir(self, dataset_dir: Path, files: List, source_format: str, target_path: str):
        logger.info(
            f"Converting dataset from {source_format} to numpy format...")

        try:
            if source_format == "pickle":
                self._convert_pickle_to_numpy_files(target_path, files)
            elif source_format == "mat":
                self._convert_mat_to_numpy_files(target_path, files)
            else:
                logger.warning(
                    f"Conversion from {source_format} to numpy not implemented")

            logger.info(f"Dataset converted to IR format at {target_path}")

        except Exception as e:
            logger.error(f"Failed to convert dataset to IR format: {e}")
            raise

    def _convert_pickle_to_numpy_files(self, output_dir: str, files: List[str]) -> Dict[str, np.ndarray]:
        """
        Convert CIFAR-10 pickle batch files to .npy files and return arrays in NCHW format.

        Args:
            output_dir: Directory to store the .npy files
            files: List of CIFAR-10 pickle batch file paths

        Returns:
            Dictionary containing X_train, y_train, X_test, y_test as NumPy arrays
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            X_train, y_train = [], []
            X_test, y_test = [], []

            for batch_file in files:
                if "batches.meta" in batch_file:
                    continue  # Skip metadata

                with open(batch_file, "rb") as f:
                    data = pickle.load(f, encoding="bytes")

                if "test" in batch_file:
                    X_test.append(data[b"data"])
                    
                    y_test.extend(data[b"labels"])
                elif "data_batch" in batch_file:
                    X_train.append(data[b"data"])
                    y_train.extend(data[b"labels"])

            # Reshape to NCHW format (no transpose needed after reshape)
            X_train = np.vstack(X_train).reshape(-1, 3, 32, 32)
            X_test = np.vstack(X_test).reshape(-1, 3, 32, 32)
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            # Save to .npy files
            np.save(os.path.join(output_dir, "X_train.npy"), X_train)
            np.save(os.path.join(output_dir, "y_train.npy"), y_train)
            np.save(os.path.join(output_dir, "X_test.npy"), X_test)
            np.save(os.path.join(output_dir, "y_test.npy"), y_test)

            logger.info(f"Saved processed CIFAR-10 numpy arrays to {output_dir} in NCHW format")


            return {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test
            }

        except Exception as e:
            logger.error(f"Failed to convert CIFAR-10 pickle to numpy: {e}")
            raise

    def _convert_mat_to_numpy_files(self, output_dir: str, files: List[str]) -> Dict[str, np.ndarray]:
        """
        Convert .mat file contents to .npy files and return them as NumPy arrays in NCHW format.

        Args:
            output_dir: Directory to store the .npy files
            files: List of .mat file paths (only the first one is used)

        Returns:
            Dictionary mapping keys to NumPy arrays
        """
        try:
            if not files:
                raise FileNotFoundError("No .mat files provided.")

            os.makedirs(output_dir, exist_ok=True)

            mat_data = scipy.io.loadmat(files[0])
            numpy_data = {}

            for key in mat_data:
                if key.startswith("__"):
                    continue  # Skip metadata
                array = mat_data[key]

                # Convert HWC (N, 32, 32, 3) to NCHW if needed
                if array.ndim == 4 and array.shape[3] == 3:
                    array = array.transpose(0, 3, 1, 2)
                elif array.ndim == 2 and array.shape[1] == 3072:
                    array = array.reshape(-1, 3, 32, 32)  # Already NCHW
                elif array.ndim == 2 and array.shape[1] == 1:
                    array = array.flatten()

                numpy_data[key] = array
                npy_path = os.path.join(output_dir, f"{key}.npy")
                np.save(npy_path, array)
                logger.info(f"Saved {key} to {npy_path}")

            # Rename keys if common naming is used
            rename_map = {
                "data": "X_train",
                "labels": "y_train",
                "test_data": "X_test",
                "test_labels": "y_test"
            }
            for old_key, new_key in rename_map.items():
                if old_key in numpy_data:
                    numpy_data[new_key] = numpy_data.pop(old_key)

            return numpy_data   

        except Exception as e:
            logger.error(f"Failed to convert .mat to .npy: {e}")
            raise
