"""
Auxiliary File Management Utilities for Landseer Pipeline

Provides hybrid approach combining declarative configuration with standardized directory structure
for handling auxiliary files like watermark triggers, models, and other tool-specific resources.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class AuxiliaryFileManager:
    """Manages auxiliary files for pipeline tools using hybrid approach"""
    
    STANDARD_AUXILIARY_PATH = "/data/auxiliary"
    
    def __init__(self, working_dir: Union[str, Path]):
        """
        Initialize auxiliary file manager
        
        Args:
            working_dir: Base directory for organizing auxiliary files
        """
        self.working_dir = Path(working_dir)
        self.auxiliary_staging_dir = self.working_dir / "auxiliary_staging"
        self.auxiliary_staging_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_auxiliary_directory(self, tool_name: str, auxiliary_files: Optional[List] = None) -> Optional[str]:
        """
        Prepare standardized auxiliary directory for a tool
        
        Args:
            tool_name: Name of the tool
            auxiliary_files: List of AuxiliaryFile objects from tool configuration
            
        Returns:
            Path to prepared auxiliary directory, or None if no auxiliary files
        """
        if not auxiliary_files:
            return None
            
        tool_aux_dir = self.auxiliary_staging_dir / tool_name
        if tool_aux_dir.exists():
            shutil.rmtree(tool_aux_dir)
        tool_aux_dir.mkdir(parents=True, exist_ok=True)
        
        for aux_file in auxiliary_files:
            try:
                local_path = Path(aux_file.local_path)
                if not local_path.exists():
                    if aux_file.required:
                        raise FileNotFoundError(f"Required auxiliary file not found: {local_path}")
                    else:
                        logger.warning(f"Optional auxiliary file not found: {local_path} - skipping")
                        continue
                
                # Determine destination path within auxiliary directory
                dest_name = local_path.name
                dest_path = tool_aux_dir / dest_name
                
                # Copy file or directory
                if local_path.is_file():
                    shutil.copy2(local_path, dest_path)
                elif local_path.is_dir():
                    shutil.copytree(local_path, dest_path)
                
                logger.debug(f"Staged auxiliary file: {local_path} -> {dest_path}")
                
            except Exception as e:
                if aux_file.required:
                    raise RuntimeError(f"Failed to stage required auxiliary file {aux_file.local_path}: {e}")
                else:
                    logger.warning(f"Failed to stage optional auxiliary file {aux_file.local_path}: {e}")
        
        return str(tool_aux_dir)
    
    def get_standard_volume_mount(self, tool_aux_dir: str) -> Dict[str, Dict[str, str]]:
        """
        Get standardized volume mount for auxiliary directory
        
        Args:
            tool_aux_dir: Path to tool's auxiliary directory
            
        Returns:
            Docker volume mount configuration
        """
        return {
            os.path.abspath(tool_aux_dir): {
                "bind": self.STANDARD_AUXILIARY_PATH,
                "mode": "ro"
            }
        }
    
    def cleanup_staging(self, tool_name: Optional[str] = None):
        """
        Clean up auxiliary staging directories
        
        Args:
            tool_name: Specific tool to clean up, or None for all
        """
        if tool_name:
            tool_aux_dir = self.auxiliary_staging_dir / tool_name
            if tool_aux_dir.exists():
                shutil.rmtree(tool_aux_dir)
                logger.debug(f"Cleaned up auxiliary staging for tool: {tool_name}")
        else:
            if self.auxiliary_staging_dir.exists():
                shutil.rmtree(self.auxiliary_staging_dir)
                logger.debug("Cleaned up all auxiliary staging directories")
