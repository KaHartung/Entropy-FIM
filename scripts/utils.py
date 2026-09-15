"""
Utility functions for FIM pipeline.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the pipeline.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs to

    Returns:
        Configured logger
    """
    logger = logging.getLogger("fim")
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def parse_cli_overrides(args: list) -> dict:
    """
    Parse CLI arguments to extract config overrides.

    Expected format: --key value --nested.key value

    Args:
        args: List of CLI arguments

    Returns:
        Dictionary of overrides
    """
    overrides = {}
    i = 0
    while i < len(args):
        if args[i].startswith('--'):
            key = args[i][2:]  # Remove '--'
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
                value = args[i + 1]
                overrides[key] = value
                i += 2
            else:
                i += 1
        else:
            i += 1

    return overrides
