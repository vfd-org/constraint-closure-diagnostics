"""
Logging utilities for VFD Dashboard.
"""

import logging
import sys
from typing import Optional


_logger: Optional[logging.Logger] = None


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level
        log_file: Optional file to write logs to

    Returns:
        Configured logger
    """
    global _logger

    logger = logging.getLogger("vfd_dash")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """
    Get the configured logger, creating default if needed.

    Returns:
        Logger instance
    """
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger
