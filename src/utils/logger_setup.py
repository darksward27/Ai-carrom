"""
Logger Setup Utility
Configures logging for the Carrom Pool ML Bot
"""

import sys
import os
from loguru import logger


def setup_logger(debug: bool = False):
    """Setup logging configuration"""
    
    # Remove default logger
    logger.remove()
    
    # Console logging
    log_level = "DEBUG" if debug else "INFO"
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File logging
    os.makedirs("logs", exist_ok=True)
    logger.add(
        "logs/carrom_bot.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="10 days",
        compression="zip"
    )
    
    # Error file logging
    logger.add(
        "logs/carrom_bot_errors.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="5 MB",
        retention="30 days"
    )
    
    logger.info(f"Logger setup complete (debug={'ON' if debug else 'OFF'})") 