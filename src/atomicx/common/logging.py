"""Structured logging setup using Loguru."""

from __future__ import annotations

import sys

from loguru import logger

from atomicx.config import get_settings


def setup_logging() -> None:
    """Configure Loguru with structured output."""
    settings = get_settings()

    # Remove default handler
    logger.remove()

    # Console handler with structured format
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler for JSON logs
    # Aggressive rotation to prevent huge log files
    logger.add(
        "logs/atomicx_{time:YYYY-MM-DD}.log",
        rotation="10 MB",  # Rotate at 10MB instead of 100MB
        retention="7 days",  # Keep only 1 week instead of 30 days
        compression="gz",  # Compress old logs to save disk space
        level="INFO",  # Skip DEBUG spam, only INFO and above
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        serialize=True,
        # Filter out repetitive websocket noise
        filter=lambda record: not (
            "websocket" in record["message"].lower() and
            "disconnect" in record["message"].lower()
        )
    )

    logger.info("AtomicX logging initialized", level=settings.log_level)
