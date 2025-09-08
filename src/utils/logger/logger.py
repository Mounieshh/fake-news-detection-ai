import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


_LOGGER_CACHE = {}


def _ensure_logs_dir(log_dir: str) -> None:
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        # If creating directory fails, we silently ignore and rely on console logging
        pass


def get_logger(name: Optional[str] = None,
               level: int = logging.INFO,
               log_dir: str = "logs",
               log_file: str = "app.log",
               max_bytes: int = 2 * 1024 * 1024,
               backup_count: int = 3) -> logging.Logger:
    """
    Get a configured logger with console and rotating file handlers.

    Parameters
    ----------
    name: Optional[str]
        Logger name; typically __name__. Uses root if None.
    level: int
        Logging level for the logger and handlers.
    log_dir: str
        Directory where log files are written.
    log_file: str
        Log filename within log_dir.
    max_bytes: int
        Max size per log file before rotation.
    backup_count: int
        Number of rotated log files to keep.
    """
    cache_key = (name, level, log_dir, log_file)
    if cache_key in _LOGGER_CACHE:
        return _LOGGER_CACHE[cache_key]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Rotating file handler
        _ensure_logs_dir(log_dir)
        try:
            fh = RotatingFileHandler(
                filename=os.path.join(log_dir, log_file),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception:
            # If file handler fails (e.g., permission issues), continue with console only
            pass

    _LOGGER_CACHE[cache_key] = logger
    return logger


