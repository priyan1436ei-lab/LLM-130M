"""Logging utilities for Privan-130M."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "privan",
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up and configure a structured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
