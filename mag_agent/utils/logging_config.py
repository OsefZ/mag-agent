import logging
import sys
import os

# Default log level - can be overridden by environment variable
LOG_LEVEL = os.getenv("MAG_LOG_LEVEL", "INFO")

def setup_logger() -> logging.Logger:
    """
    Configures and returns a root logger.
    """
    logger = logging.getLogger("MAGAgent")
    logger.propagate = False  # Prevent duplicate logs in parent loggers
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    # Set level from config, default to INFO if invalid
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(level)
    
    return logger

logger = setup_logger() 