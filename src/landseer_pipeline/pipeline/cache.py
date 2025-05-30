import hashlib
import json
from pathlib import Path
from filelock import FileLock
from landseer_pipeline.utils.docker import get_image_digest
from typing import List

#TODO: Make a class for cahcing to get output path from settings
class CacheManager:
    def __init__(self, settings):
        self.settings = settings

    def safe_cache_path(self, cache_key):
        cache_path = self.get_cache_path(cache_key)
        lock = FileLock(str(cache_path) + ".lock")
        lock.acquire()
        return cache_path, lock

    def get_cache_path(self, cache_key: str) -> Path:
        """Get the path to the cache directory for a given cache key."""
        return self.settings.output_dir / cache_key
    
    def is_cached(self, cache_key: str) -> bool:
        """Check if the output for the given cache key exists."""
        cache_path = self.get_cache_path(cache_key)
        if not cache_path.exists():
            return False
        success_file = cache_path / ".success"
        if success_file.exists():
            return True
    
    def compute_cache_key(self, tools: List,current_tool, stage: str, input_path: str, dataset) -> str:
        tool_sequence_data = []
        for tool in tools:
            config_model_content = ""
            print(f"Processing tool: {tool}")
            if tool.docker.config_script:
                config_script_path = Path(tool.docker.config_script)
                if not config_script_path.exists():
                    raise FileNotFoundError(f"Config script {config_script_path} does not exist.")
                with open(config_script_path, "r") as f:
                    config_model_content = f.read()
            image_name = tool.docker.image
            digest = get_image_digest(image_name)
            tool_entry = {
            "tool_name": tool.name,
            "tool_config": tool.dict() if hasattr(tool, "dict") else {},
            "image_digest": digest,
            "config": config_model_content
            }
            tool_sequence_data.append(tool_entry)
        data = {
        "tool_sequence": tool_sequence_data,  # order matters
        "current_tool": str(current_tool),
        "stage": stage,
        "input_path": str(input_path),
        "dataset": str(dataset),
        }
        json_str = json.dumps(data, sort_keys=True)
        hash_val = hashlib.sha256(json_str.encode()).hexdigest()
        cache_dir = self.settings.output_dir / hash_val
        
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / "metadata.json", "w") as f:
            json.dump(data, f, indent=4)
        return hash_val

    
    def get_cached_output(self, cache_key: str) -> str:
        cache_path = self.settings.output_dir / cache_key / "output"
        return cache_path if cache_path.exists() else None
    
    def mark_as_cached(self, cache_key: str, output_path: str):
        """Mark the output as cached by creating a .success file."""
        cache_path = self.get_cache_path(cache_key)
        success_file = cache_path / ".success"
        success_file.touch()

    def mark_as_failed(self, cache_key: str):
        """Mark the output as failed by creating a .failed file."""
        cache_path = self.get_cache_path(cache_key)
        failed_file = cache_path / ".failed"
        failed_file.touch()