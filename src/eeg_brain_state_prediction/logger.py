import functools
import logging
import os
import warnings
from pathlib import Path
from typing import Optional

def setup_logger(name: str = __name__, 
                log_file: Optional[str] = None, 
                level: str = "INFO") -> logging.Logger:
    """Configure logging with timestamp and formatting

    Args:
        name (str): Logger name
        log_file (Optional[str]): Path to log file
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        logging.Logger: Configured logger instance
    """
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def log_execution(logger: Optional[logging.Logger] = None):
    """Decorator to log function execution with parameters and results

    Args:
        logger (Optional[logging.Logger]): Logger instance to use. If None, creates a new logger.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = setup_logger(func.__module__)

            logger.debug(f"Entering {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator