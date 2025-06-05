"""
Configuration Loader Utility
Loads and validates configuration from YAML files
"""

import yaml
import os
from typing import Dict, Any
from loguru import logger


class ConfigLoader:
    """Utility class for loading and validating configuration"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Validate configuration
            ConfigLoader._validate_config(config)
            
            logger.info(f"Configuration loaded from: {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]):
        """Validate configuration structure and values"""
        required_sections = ['device', 'detection', 'model', 'physics', 'training', 'gameplay']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate device config
        device_config = config['device']
        required_device_keys = ['screen_resolution', 'game_area']
        for key in required_device_keys:
            if key not in device_config:
                raise ValueError(f"Missing required device config: {key}")
        
        # Validate model config
        model_config = config['model']
        if 'cnn' not in model_config or 'rl' not in model_config:
            raise ValueError("Model config must contain 'cnn' and 'rl' sections")
        
        logger.debug("Configuration validation passed")
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """Save configuration to YAML file"""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to: {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise 