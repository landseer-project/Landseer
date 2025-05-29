import os
import shutil
import hashlib
import importlib.util

def hash_file(path, bits=64):
    hasher = hashlib.blake2s(digest_size=bits // 8)
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def copy_or_link_log(src_log_path, dest_log_path, method="copy"):
    os.makedirs(os.path.dirname(dest_log_path), exist_ok=True)

    if method == "symlink":
        if os.path.exists(dest_log_path):
            os.remove(dest_log_path)
        os.symlink(os.path.abspath(src_log_path), dest_log_path)
    else:  # default is copy
        shutil.copy2(src_log_path, dest_log_path)

def merge_directories(input_path: str, dataset_dir: str) -> str:
    # check if input_path and dataset_dir is same path
    import uuid
    unique_id = uuid.uuid4().hex
    input_dir = os.path.abspath(f"data/temp_input_{unique_id}")
    # logger.debug(f"Merging directories: {input_path} and {dataset_dir}")
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
    input_path = os.path.abspath(input_path)
    dataset_dir = os.path.abspath(dataset_dir)
    if os.path.isdir(input_path) and os.path.exists(dataset_dir):
        for file in os.listdir(input_path):
            file_path = os.path.join(input_path, file)
            if os.path.isfile(file_path):
                shutil.copy(file_path, input_dir)
        # logger.debug(f"Copying file from {dataset_dir} to {input_dir}")
        if os.path.abspath(dataset_dir) == os.path.abspath(input_path):
            # logger.debug(f"Input path and dataset path are same")
            return os.path.abspath(input_dir)
        for file in os.listdir(dataset_dir):
            file_path = os.path.join(dataset_dir, file)
            if os.path.isfile(file_path):
                shutil.copy(file_path, input_dir)
    else:
        shutil.copy(input_path, input_dir)
        shutil.copy(dataset_dir, input_dir)
    # exit(0)
    return os.path.abspath(input_dir)

def create_directory(path: str):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def remove_directory(path: str):
    """Remove a directory and its contents."""
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    return path

def copy_directory(src: str, dest: str):
    """Copy a directory and its contents to a new location."""
    if not os.path.exists(dest):
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        raise FileExistsError(f"Destination directory {dest} already exists.")
    return dest

def load_config_from_script(script_path: str):
    abs_path = os.path.abspath(script_path)
    spec = importlib.util.spec_from_file_location("config_module", abs_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config  
