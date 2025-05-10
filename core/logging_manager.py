"""
Logging manager for Research Agent.
Sets up and configures logging throughout the application.
"""

import os
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        verbose: If True, sets log level to DEBUG, otherwise INFO
        log_file: Optional file path for log file
        
    Returns:
        Logger instance for the application
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(exist_ok=True, parents=True)
    else:
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True, parents=True)
        log_file = log_dir / "research_agent.log"
        
    # Set log level based on verbose flag
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            # Console handler with rich formatting
            RichHandler(rich_tracebacks=True, markup=True),
            # File handler
            logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding="utf-8"
            )
        ]
    )
    
    # Create and configure the application logger
    logger = logging.getLogger("research_agent")
    
    # Set propagate to False to prevent duplicate logging
    logger.propagate = False
    
    # Log startup information
    logger.info(f"Logging initialized at level: {logging.getLevelName(log_level)}")
    if verbose:
        logger.debug("Verbose logging enabled")
        
    # Set levels for some verbose third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)
    
    return logger
