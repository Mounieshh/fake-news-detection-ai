"""
Centralized logging utilities for the project.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("message")
"""

from .logger import get_logger

__all__ = ["get_logger"]


