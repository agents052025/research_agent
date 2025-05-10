"""
Configuration Manager for Research Agent.
Handles loading, validation, and access to application configuration.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """
    Manages configuration for the Research Agent application.
    Handles loading from files and environment variables.
    """
    
    DEFAULT_CONFIG = {
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7
        },
        "tools": {
            "search": {
                "enabled": True,
                "api_key_env": "SERPAPI_API_KEY",
                "engine": "google",
                "max_results": 10
            },
            "url_fetcher": {
                "enabled": True,
                "timeout": 30,
                "max_size": 1048576  # 1MB
            },
            "pdf_reader": {
                "enabled": True,
                "max_pages": 50
            },
            "data_analysis": {
                "enabled": True,
                "packages": ["pandas", "numpy"]
            },
            "visualization": {
                "enabled": True,
                "max_width": 80,
                "use_unicode": True
            },
            "database": {
                "enabled": True,
                "path": "data/research_data.db",
                "cache_enabled": True,
                "cache_ttl": 86400  # 24 hours
            }
        },
        "planning": {
            "max_steps": 10,
            "step_timeout": 300,  # 5 minutes
            "research_depth": "medium"  # shallow, medium, deep
        },
        "prompts": {
            "system_prompt": """
            You are a research assistant tasked with gathering, analyzing, and synthesizing information.
            Follow these guidelines:
            1. Break down complex research tasks into logical steps
            2. Gather information from credible sources
            3. Analyze data objectively
            4. Synthesize findings into cohesive reports
            5. Provide proper citations for all information
            6. Highlight limitations and uncertainties in your findings
            """
        },
        "output": {
            "format": "markdown",
            "include_citations": True,
            "include_timestamp": True
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ConfigManager.
        
        Args:
            config_path: Path to the configuration file (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path:
            self._load_config_file(config_path)
            
        # Override with environment variables
        self._override_from_env()
        
        # Validate the configuration
        self._validate_config()
        
    def _load_config_file(self, config_path: str) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
        """
        path = Path(config_path)
        
        # If path is relative, look in the config directory
        if not path.is_absolute():
            base_dir = Path(__file__).parent.parent
            path = base_dir / "config" / path
            
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    
                # Deep merge the configurations
                self._deep_merge(self.config, file_config)
                self.logger.info(f"Loaded configuration from {path}")
            else:
                self.logger.warning(f"Configuration file not found at {path}, using defaults")
                
                # If default_config.json doesn't exist, create it
                if path.name == "default_config.json":
                    self._save_default_config(path)
                    
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Error loading configuration from {path}: {str(e)}")
            
    def _save_default_config(self, path: Path) -> None:
        """
        Save the default configuration to a file.
        
        Args:
            path: Path where to save the default configuration
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.DEFAULT_CONFIG, f, indent=2)
            self.logger.info(f"Created default configuration at {path}")
        except IOError as e:
            self.logger.error(f"Error creating default configuration at {path}: {str(e)}")
            
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep merge source dictionary into target dictionary.
        
        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
                
    def _override_from_env(self) -> None:
        """
        Override configuration values from environment variables.
        Environment variables should be in the format RESEARCH_AGENT_SECTION_KEY=value.
        """
        prefix = "RESEARCH_AGENT_"
        
        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                # Remove prefix and split by underscore
                key_parts = env_key[len(prefix):].lower().split('_')
                
                # Navigate to the correct position in the config
                current = self.config
                for part in key_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                    
                # Set the value, with type conversion
                key = key_parts[-1]
                if env_value.lower() in ('true', 'false'):
                    current[key] = env_value.lower() == 'true'
                elif env_value.isdigit():
                    current[key] = int(env_value)
                elif env_value.replace('.', '', 1).isdigit() and env_value.count('.') == 1:
                    current[key] = float(env_value)
                else:
                    current[key] = env_value
                    
                self.logger.debug(f"Overriding config value from environment: {env_key}")
                
    def _validate_config(self) -> None:
        """
        Validate the current configuration.
        Logs warnings for potential issues.
        """
        # Check LLM configuration
        if "llm" not in self.config:
            self.logger.warning("No LLM configuration found, using defaults")
            self.config["llm"] = self.DEFAULT_CONFIG["llm"]
            
        # Check tools configuration
        if "tools" not in self.config:
            self.logger.warning("No tools configuration found, using defaults")
            self.config["tools"] = self.DEFAULT_CONFIG["tools"]
            
        # Validate specific tool configurations
        for tool_name, tool_config in self.config.get("tools", {}).items():
            if not isinstance(tool_config, dict):
                self.logger.warning(f"Invalid configuration for tool {tool_name}, using defaults")
                self.config["tools"][tool_name] = self.DEFAULT_CONFIG["tools"].get(tool_name, {})
                
    def get_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self.config
        
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation path.
        
        Args:
            path: Dot-notation path, e.g., "llm.model"
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        parts = path.split('.')
        current = self.config
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
            
        return current
        
    def save_config(self, path: str) -> bool:
        """
        Save the current configuration to a file.
        
        Args:
            path: Path to save the configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Saved configuration to {path}")
            return True
        except IOError as e:
            self.logger.error(f"Error saving configuration to {path}: {str(e)}")
            return False
