# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 12:28:17 2026

@author: Hachem
"""

# -*- coding: utf-8 -*-
"""
Logging utilities for HYPERVIEW2 challenge.

Author: Hachem
Created: Sat Jan 3 12:28:17 2026
"""

import logging
from pathlib import Path
import sys


def setup_logger(log_dir="logs", log_file="experiment.log", name="hyperview2", level=logging.INFO):
    """
    Setup logger with both file and console handlers.
    
    Parameters
    ----------
    log_dir : str, default="logs"
        Directory to store log files
    log_file : str, default="experiment.log"
        Name of the log file
    name : str, default="hyperview2"
        Logger name
    level : int, default=logging.INFO
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns
    -------
    logging.Logger
        Configured logger instance
        
    Example
    -------
    >>> logger = setup_logger(log_dir="logs", log_file="train.log")
    >>> logger.info("Training started")
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / log_file

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_formatter = logging.Formatter(
        "%(levelname)-8s | %(message)s"
    )

    # File handler (detailed logs)
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    logger.info(f"Logger initialized: {log_path}")
    return logger


def get_logger(name="hyperview2"):
    """
    Get existing logger instance.
    
    Parameters
    ----------
    name : str, default="hyperview2"
        Logger name
        
    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)


def log_config(config, logger=None):
    """
    Log configuration dictionary in a readable format.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    logger : logging.Logger, optional
        Logger instance. If None, uses default logger.
    """
    if logger is None:
        logger = get_logger()
    
    logger.info("=" * 80)
    logger.info("CONFIGURATION")
    logger.info("=" * 80)
    
    def log_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info("  " * indent + f"{key}:")
                log_dict(value, indent + 1)
            else:
                logger.info("  " * indent + f"{key}: {value}")
    
    log_dict(config)
    logger.info("=" * 80)
