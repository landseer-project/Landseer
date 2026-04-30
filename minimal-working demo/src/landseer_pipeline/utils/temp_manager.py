"""Temporary directory management with cleanup on exit"""
import os
import shutil
import atexit
import signal
import logging
import threading
from typing import Set
from pathlib import Path

logger = logging.getLogger(__name__)

class TempDirectoryManager:
    """Global manager for temporary directories with automatic cleanup"""
    
    def __init__(self):
        self._temp_dirs: Set[str] = set()
        self._cleanup_registered = False
    
    def register_temp_dir(self, temp_dir: str) -> None:
        """Register a temporary directory for cleanup"""
        self._temp_dirs.add(os.path.abspath(temp_dir))
        logger.debug(f"Registered temporary directory: {temp_dir}")
        
        if not self._cleanup_registered:
            self._register_cleanup_handlers()
            self._cleanup_registered = True
    
    def unregister_temp_dir(self, temp_dir: str) -> None:
        """Unregister a temporary directory (already cleaned up)"""
        abs_path = os.path.abspath(temp_dir)
        self._temp_dirs.discard(abs_path)
        logger.debug(f"Unregistered temporary directory: {temp_dir}")
    
    def cleanup_temp_dir(self, temp_dir: str) -> None:
        """Clean up a specific temporary directory"""
        abs_path = os.path.abspath(temp_dir)
        try:
            if os.path.exists(abs_path):
                shutil.rmtree(abs_path, ignore_errors=True)
                logger.debug(f"Cleaned up temporary directory: {abs_path}")
            self.unregister_temp_dir(abs_path)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory {abs_path}: {e}")
    
    def cleanup_all(self) -> None:
        """Clean up all registered temporary directories"""
        if not self._temp_dirs:
            return
            
        logger.info(f"Cleaning up {len(self._temp_dirs)} temporary directories...")
        
        for temp_dir in list(self._temp_dirs):
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")
        
        self._temp_dirs.clear()
        logger.info("Temporary directory cleanup completed")
    
    def _register_cleanup_handlers(self) -> None:
        """Register cleanup handlers for various exit scenarios"""
        # Register atexit handler for normal exit
        atexit.register(self.cleanup_all)
        
        # Only register signal handlers in the main thread
        if threading.current_thread() is threading.main_thread():
            try:
                # Register signal handlers for Ctrl+C and termination
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
                logger.debug("Registered signal handlers for temporary directories")
            except ValueError as e:
                logger.debug(f"Could not register signal handlers: {e}")
        else:
            logger.debug("Signal handlers can only be registered in main thread")
        
        logger.debug("Registered cleanup handlers for temporary directories")
    
    def _signal_handler(self, signum, frame):
        """Handle signals by cleaning up and re-raising"""
        logger.info(f"Received signal {signum}, cleaning up temporary directories...")
        self.cleanup_all()
        
        # Re-raise the signal to ensure proper exit
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    
    def cleanup_existing_temp_dirs(self, base_dir: str = "data") -> None:
        """Clean up any existing temporary directories from previous runs"""
        base_path = Path(base_dir)
        if not base_path.exists():
            return
        
        temp_dirs = []
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith("temp_input_"):
                temp_dirs.append(item)
        
        if temp_dirs:
            logger.info(f"Found {len(temp_dirs)} leftover temporary directories from previous runs")
            for temp_dir in temp_dirs:
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.debug(f"Cleaned up leftover temporary directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up leftover directory {temp_dir}: {e}")
            
            logger.info("Leftover temporary directory cleanup completed")

# Global instance
temp_manager = TempDirectoryManager()
