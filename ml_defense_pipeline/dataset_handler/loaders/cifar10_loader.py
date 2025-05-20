import os
import pickle
import logging
import numpy as np
import scipy.io
import torch
from typing import Dict, List

def convert_pickle_to_numpy_files(output_dir: str, files: List[str], add_trigger=False) -> Dict[str, np.ndarray]:
        try:
            os.makedirs(output_dir, exist_ok=True)

            X_train, y_train = [], []
            X_test, y_test = [], []

            for batch_file in files:
                if "batches.meta" in batch_file:
                    continue  

                with open(batch_file, "rb") as f:
                    data = pickle.load(f, encoding="bytes")

                if "test" in batch_file:
                    X_test.append(data[b"data"])
                    y_test.extend(data[b"labels"])
                elif "data_batch" in batch_file:
                    X_train.append(data[b"data"])
                    y_train.extend(data[b"labels"])

            X_train = np.vstack(X_train).reshape(-1, 3, 32, 32)
            X_test = np.vstack(X_test).reshape(-1, 3, 32, 32)
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            if add_trigger:
                # TODO:user select the target label
                target_label = 1
                X_train, y_train = add_backdoor_trigger(X_train, target_label)
                X_test, y_test = add_backdoor_trigger(X_test, target_label)

            np.save(os.path.join(output_dir, "X_train.npy"), X_train)
            np.save(os.path.join(output_dir, "y_train.npy"), y_train)
            np.save(os.path.join(output_dir, "X_test.npy"), X_test)
            np.save(os.path.join(output_dir, "y_test.npy"), y_test)
            # logger.info(f"Saved processed CIFAR-10 numpy arrays to {output_dir} in NCHW format")
            return {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test
            }

        except Exception as e:
            # logger.error(f"Failed to convert CIFAR-10 pickle to numpy: {e}")
            raise

def convert_mat_to_numpy_files(output_dir: str, files: List[str], add_trigger=False) -> Dict[str, np.ndarray]:
        try:
            if not files:
                raise FileNotFoundError("No .mat files provided.")

            os.makedirs(output_dir, exist_ok=True)

            mat_data = scipy.io.loadmat(files[0])
            numpy_data = {}

            for key in mat_data:
                if key.startswith("__"):
                    continue
                array = mat_data[key]

                if array.ndim == 4 and array.shape[3] == 3:
                    array = array.transpose(0, 3, 1, 2)
                elif array.ndim == 2 and array.shape[1] == 3072:
                    array = array.reshape(-1, 3, 32, 32)  
                elif array.ndim == 2 and array.shape[1] == 1:
                    array = array.flatten()

                numpy_data[key] = array
                npy_path = os.path.join(output_dir, f"{key}.npy")
                np.save(npy_path, array)
                # logger.info(f"Saved {key} to {npy_path}")

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
            # logger.error(f"Failed to convert .mat to .npy: {e}")
            raise

def add_backdoor_trigger(images, target_label, trigger_value=1.0):
    images[:, :, -3:, -3:] = trigger_value
    labels = torch.full((images.size(0),), target_label, dtype=torch.long)
    return images, labels