import logging
import sys
from pathlib import Path
import colorlog
import os

def setup_logger(timestamp, log_level='INFO', pipeline_id=None, log_dir='logs'):
    log_dir = log_dir if log_dir else 'logs'
    #ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    log_file_id = f"pipeline_{pipeline_id}_{timestamp}"
    log_file_path = f"{log_dir}/{log_file_id}.log"

    LoggingManager.setup_logging(log_level, log_file_path)
    return log_file_id

class LoggingManager:
    @staticmethod
    def setup_logging(log_level, log_file_path):

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG) #TODO: improve on this

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # File handler with unique pipeline ID
        file_handler = logging.FileHandler(
            log_file_path, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        for noisy in ["urllib3", "docker", "requests", "filelock"]:
            logging.getLogger(noisy).setLevel(logging.WARNING)

        # Create a colorlog formatter directly
        color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s%(reset)s: %(message)s",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)

        logger.info("Logging initialized. Log file: %s", log_file_path)
        return logger
