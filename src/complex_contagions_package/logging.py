"""Logging module."""
import logging

logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

def log_error(message):
    """Error message."""
    logging.error(message)

def log_info(message):
    """Info message."""
    logging.info(message)

def log_warning(message):
    """Warning message."""
    logging.warning(message)
