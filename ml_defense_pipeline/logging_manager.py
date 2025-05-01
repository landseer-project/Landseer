"""
Logging configuration for ML Defense Pipeline
"""
import logging
import sys
from pathlib import Path

class LoggingManager:
    @staticmethod
    def setup_logging():
        """Configure logging to file and console"""
        # Create logs directory
        logs_dir = Path("./logs")
        logs_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger("defense_pipeline")
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        #TODO: create a unique pipeline ID for each run with a timestamp and use it in the log file name
        file_handler = logging.FileHandler(logs_dir / "pipeline_debug.log", mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_fmt = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_fmt)
        logger.addHandler(console_handler)
        
        logger.info("Logging initialized. Log file: logs/pipeline.log")
        return logger