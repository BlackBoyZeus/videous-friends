"""
Configuration Manager for Videous & Friends

This module provides utilities for loading and managing configuration.
"""

import os
import yaml
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_dir=None):
        """Initialize the configuration manager.
        
        Args:
            config_dir (str, optional): Directory containing config files.
                                       If None, uses the default configs directory.
        """
        if config_dir is None:
            # Default to the configs directory in the project root
            self.config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs")
        else:
            self.config_dir = config_dir
            
        self.configs = {}
    
    def load_config(self, component_name):
        """Load configuration for a specific component.
        
        Args:
            component_name (str): Name of the component (e.g., 'pegasus', 'chef')
            
        Returns:
            dict: Configuration dictionary or empty dict if not found
        """
        config_path = os.path.join(self.config_dir, f"{component_name}.yaml")
        
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file not found: {config_path}")
            return {}
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.configs[component_name] = config
                logger.info(f"Loaded configuration for {component_name}")
                return config
        except Exception as e:
            logger.error(f"Error loading configuration for {component_name}: {e}")
            return {}
    
    def get_config(self, component_name, section=None, key=None, default=None):
        """Get configuration value.
        
        Args:
            component_name (str): Name of the component
            section (str, optional): Section name within the config
            key (str, optional): Key within the section
            default: Default value if not found
            
        Returns:
            The configuration value or default if not found
        """
        if component_name not in self.configs:
            self.load_config(component_name)
            
        config = self.configs.get(component_name, {})
        
        if section is None:
            return config
            
        section_data = config.get(section, {})
        
        if key is None:
            return section_data
            
        return section_data.get(key, default)
    
    def save_config(self, component_name, config_data):
        """Save configuration for a component.
        
        Args:
            component_name (str): Name of the component
            config_data (dict): Configuration data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        config_path = os.path.join(self.config_dir, f"{component_name}.yaml")
        
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
                
            self.configs[component_name] = config_data
            logger.info(f"Saved configuration for {component_name}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration for {component_name}: {e}")
            return False


# Example usage
if __name__ == "__main__":
    config_manager = ConfigManager()
    
    # Load configuration
    pegasus_config = config_manager.load_config("pegasus")
    
    # Get specific values
    audio_weight = config_manager.get_config("pegasus", "Weights", "audio_rms", 0.5)
    print(f"Audio weight: {audio_weight}")
