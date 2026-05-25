
import logging
from pathlib import Path
import sys


def setup_logger(log_dir="logs", log_file="experiment.log", name="hyperview2", level=logging.INFO):
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / log_file

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    logger.handlers.clear()

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_formatter = logging.Formatter(
        "%(levelname)-8s | %(message)s"
    )

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.propagate = False

    logger.info(f"Logger initialized: {log_path}")
    return logger


def get_logger(name="hyperview2"):
    
    return logging.getLogger(name)


def log_config(config, logger=None):

    if logger is None:
        logger = get_logger()

    def log_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info("  " * indent + f"{key}:")
                log_dict(value, indent + 1)
            else:
                logger.info("  " * indent + f"{key}: {value}")

    log_dict(config)
