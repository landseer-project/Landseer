import logging
import sys
from pathlib import Path
import colorlog
import datetime


class LoggingManager:
    @staticmethod
    def setup_logging():
        logs_dir = Path("./logs")
        logs_dir.mkdir(exist_ok=True)

        logger = logging.getLogger("defense_pipeline")
        logger.setLevel(logging.DEBUG)

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # File handler with unique pipeline ID
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        pipeline_id = f"pipeline_{timestamp}"
        file_handler = logging.FileHandler(
            logs_dir / f"{pipeline_id}_debug.log", mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

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

        logger.info("Logging initialized. Log file: logs/" +
                    f"{pipeline_id}_debug.log")
        return logger
