"""
Context Manager for the Research Agent.
Maintains context and state throughout the research process.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path


class ContextManager:
    """
    Manages research context and state throughout the research process.
    Provides methods for storing, retrieving, and persisting context data.
    """
    
    def __init__(self, context_file: Optional[str] = None):
        """
        Initialize the context manager.
        
        Args:
            context_file: Optional file path for persisting context
        """
        self.logger = logging.getLogger(__name__)
        self.context: Dict[str, Any] = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            },
            "research": {},
            "sources": [],
            "cache": {},
            "session_history": []
        }
        
        self.context_file = context_file
        
        # Load existing context if file is provided
        if context_file:
            self._load_context()
            
    def add_to_context(self, key: str, value: Any) -> None:
        """
        Add or update a value in the context.
        
        Args:
            key: Context key
            value: Value to store
        """
        # Handle nested keys using dot notation
        if '.' in key:
            parts = key.split('.')
            current = self.context
            
            # Navigate to the correct position in the context
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
                
            # Set the value
            current[parts[-1]] = value
        else:
            self.context[key] = value
            
        # Update metadata
        self.context["metadata"]["updated_at"] = datetime.now().isoformat()
        
        # Persist context if file is provided
        if self.context_file:
            self._save_context()
            
    def get_from_context(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the context.
        
        Args:
            key: Context key (supports dot notation)
            default: Default value if key is not found
            
        Returns:
            Value from context or default
        """
        # Handle nested keys using dot notation
        if '.' in key:
            parts = key.split('.')
            current = self.context
            
            # Navigate to the correct position in the context
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    return default
                current = current[part]
                
            return current
        else:
            return self.context.get(key, default)
            
    def get_context(self) -> Dict[str, Any]:
        """
        Get the complete context dictionary.
        
        Returns:
            Complete context dictionary
        """
        return self.context
        
    def add_source(self, source: Dict[str, Any]) -> None:
        """
        Add a source to the context.
        
        Args:
            source: Source information dictionary
        """
        # Ensure required fields are present
        required_fields = ["url", "title", "type"]
        if not all(field in source for field in required_fields):
            self.logger.warning(f"Source missing required fields: {required_fields}")
            return
            
        # Add timestamp if not present
        if "timestamp" not in source:
            source["timestamp"] = datetime.now().isoformat()
            
        # Add source ID if not present
        if "id" not in source:
            source["id"] = len(self.context["sources"]) + 1
            
        # Add to sources list
        self.context["sources"].append(source)
        
        # Update metadata
        self.context["metadata"]["updated_at"] = datetime.now().isoformat()
        
        # Persist context if file is provided
        if self.context_file:
            self._save_context()
            
    def get_sources(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get sources from the context, optionally filtered.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            List of sources matching filters
        """
        sources = self.context.get("sources", [])
        
        if not filters:
            return sources
            
        # Apply filters
        filtered_sources = []
        for source in sources:
            if all(source.get(key) == value for key, value in filters.items()):
                filtered_sources.append(source)
                
        return filtered_sources
        
    def add_to_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Add a value to the cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time-to-live in seconds
        """
        cache_entry = {
            "value": value,
            "created_at": datetime.now().isoformat()
        }
        
        if ttl:
            expiry = datetime.now().timestamp() + ttl
            cache_entry["expires_at"] = datetime.fromtimestamp(expiry).isoformat()
            
        self.context["cache"][key] = cache_entry
        
        # Persist context if file is provided
        if self.context_file:
            self._save_context()
            
    def get_from_cache(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value if key is not found or expired
            
        Returns:
            Cached value or default
        """
        if key not in self.context.get("cache", {}):
            return default
            
        cache_entry = self.context["cache"][key]
        
        # Check expiration
        if "expires_at" in cache_entry:
            expires_at = datetime.fromisoformat(cache_entry["expires_at"])
            if datetime.now() > expires_at:
                # Cache entry expired
                del self.context["cache"][key]
                return default
                
        return cache_entry["value"]
        
    def add_to_history(self, entry: Dict[str, Any]) -> None:
        """
        Add an entry to the session history.
        
        Args:
            entry: Session history entry
        """
        # Ensure timestamp is present
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().isoformat()
            
        self.context["session_history"].append(entry)
        
        # Persist context if file is provided
        if self.context_file:
            self._save_context()
            
    def clear_context(self, keep_metadata: bool = True) -> None:
        """
        Clear the context.
        
        Args:
            keep_metadata: Whether to keep metadata
        """
        metadata = self.context["metadata"] if keep_metadata else {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        self.context = {
            "metadata": metadata,
            "research": {},
            "sources": [],
            "cache": {},
            "session_history": []
        }
        
        # Persist context if file is provided
        if self.context_file:
            self._save_context()
            
    def _load_context(self) -> None:
        """Load context from file if it exists."""
        try:
            context_path = Path(self.context_file)
            if context_path.exists():
                with open(context_path, 'r', encoding='utf-8') as f:
                    loaded_context = json.load(f)
                    
                # Ensure basic structure is preserved
                for key in ["metadata", "research", "sources", "cache", "session_history"]:
                    if key not in loaded_context:
                        loaded_context[key] = self.context[key]
                        
                self.context = loaded_context
                self.logger.info(f"Loaded context from {self.context_file}")
            else:
                self.logger.info(f"Context file {self.context_file} not found, using default context")
        except Exception as e:
            self.logger.error(f"Error loading context from {self.context_file}: {str(e)}")
            
    def _save_context(self) -> None:
        """Save context to file."""
        try:
            context_path = Path(self.context_file)
            context_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(context_path, 'w', encoding='utf-8') as f:
                json.dump(self.context, f, indent=2)
                
            self.logger.debug(f"Saved context to {self.context_file}")
        except Exception as e:
            self.logger.error(f"Error saving context to {self.context_file}: {str(e)}")
            
    def export_context(self, file_path: str, include_cache: bool = False) -> bool:
        """
        Export context to a JSON file.
        
        Args:
            file_path: Path to export the context to
            include_cache: Whether to include cache in the export
            
        Returns:
            True if successful, False otherwise
        """
        try:
            export_path = Path(file_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a copy of the context for export
            export_context = self.context.copy()
            
            # Remove cache if not included
            if not include_cache:
                export_context.pop("cache", None)
                
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_context, f, indent=2)
                
            self.logger.info(f"Exported context to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting context to {file_path}: {str(e)}")
            return False
            
    def import_context(self, file_path: str, merge: bool = False) -> bool:
        """
        Import context from a JSON file.
        
        Args:
            file_path: Path to import the context from
            merge: Whether to merge with existing context
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import_path = Path(file_path)
            if not import_path.exists():
                self.logger.error(f"Context file {file_path} not found")
                return False
                
            with open(import_path, 'r', encoding='utf-8') as f:
                imported_context = json.load(f)
                
            if merge:
                # Merge with existing context
                for key, value in imported_context.items():
                    if key in self.context and isinstance(value, dict) and isinstance(self.context[key], dict):
                        self.context[key].update(value)
                    else:
                        self.context[key] = value
            else:
                # Replace existing context
                self.context = imported_context
                
            # Update metadata
            self.context["metadata"]["updated_at"] = datetime.now().isoformat()
            
            # Persist context if file is provided
            if self.context_file:
                self._save_context()
                
            self.logger.info(f"Imported context from {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error importing context from {file_path}: {str(e)}")
            return False
