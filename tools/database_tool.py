"""
Database Tool for the Research Agent.
Provides data storage, retrieval, and caching capabilities.
"""

import os
import json
import sqlite3
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from smolagents import Tool


class DatabaseTool(Tool):
    # Атрибути для smolagents.Tool
    name = "database"
    description = """
    Manages data storage, retrieval, and caching for the research agent.
    Uses SQLite database for local storage.
    """
    inputs = {
        "action": {
            "type": "string",
            "description": "Database action to perform (store, get, find, delete)",
        },
        "key": {
            "type": "string",
            "description": "Key for data storage or retrieval",
            "nullable": True
        },
        "value": {
            "type": "object",
            "description": "Value to store (for 'store' action)",
            "nullable": True
        },
        "filters": {
            "type": "object",
            "description": "Filters for finding data (for 'find' action)",
            "nullable": True
        }
    }
    output_type = "object"
    
    def __init__(self, db_path: str = "data/research_data.db", 
                 cache_enabled: bool = True, cache_ttl: int = 86400):
        """
        Initialize the Database Tool.
        
        Args:
            db_path: Path to SQLite database file
            cache_enabled: Whether to enable caching
            cache_ttl: Cache time-to-live in seconds (default: 24 hours)
        """
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Додаємо атрибут is_initialized для сумісності з smolagents 1.15.0
        self.is_initialized = True
        
    def forward(self, action: str, key: str = None, value: Any = None, filters: Dict[str, Any] = None) -> Any:
        """
        Forward method required by smolagents.Tool.
        Dispatches to appropriate database methods based on the action.
        
        Args:
            action: Database action to perform (store, get, find, delete, list_keys)
            key: Key for data storage or retrieval
            value: Value to store (for 'store' action)
            filters: Filters for finding data (for 'find' action)
            
        Returns:
            Result of the database operation
        """
        self.logger.info("Executing database action: %s", action)
        
        if action == "store" and key is not None and value is not None:
            return self.store_data(key, value)
        elif action == "get" and key is not None:
            return self.get_data(key)
        elif action == "delete" and key is not None:
            return self.delete_data(key)
        elif action == "find":
            if key is not None:
                # If key is provided, find by key pattern
                return self.find_data_by_key_pattern(key)
            elif filters is not None:
                # If filters are provided, first try to find in research_data
                data_results = self.find_data(filters)
                if data_results:
                    return data_results
                # If no results found in research_data, try sources
                return self.find_sources(filters)
            else:
                # If neither key nor filters are provided, return all data
                return self.find_data({})
        elif action == "list_keys":
            return self.list_data_keys(key if key else None)  # key used as prefix if provided
        else:
            self.logger.error("Invalid database action or missing parameters")
            return {"error": "Invalid database action or missing parameters"}
    
    def _init_database(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create sources table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT,
                title TEXT,
                type TEXT,
                content TEXT,
                metadata TEXT,
                timestamp TEXT
            )
            ''')
            
            # Create search_cache table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                engine TEXT,
                results TEXT,
                timestamp TEXT,
                expires_at TEXT
            )
            ''')
            
            # Create research_data table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE,
                value TEXT,
                data_type TEXT,
                timestamp TEXT
            )
            ''')
            
            # Create research_sessions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                results TEXT,
                sources TEXT,
                timestamp TEXT
            )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database initialized at %s", self.db_path)
            
        except sqlite3.Error as e:
            self.logger.error("Database initialization error: %s", str(e))
            raise RuntimeError("Failed to initialize database: %s" % str(e)) from e
            
    def store_source(self, source: Dict[str, Any]) -> int:
        """
        Store a source in the database.
        
        Args:
            source: Source information dictionary
            
        Returns:
            Source ID
            
        Raises:
            RuntimeError: If storage fails
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract source fields
            url = source.get("url", "")
            title = source.get("title", "")
            source_type = source.get("type", "website")
            content = json.dumps(source.get("content", ""))
            metadata = json.dumps(source.get("metadata", {}))
            timestamp = source.get("timestamp", datetime.now().isoformat())
            
            # Insert into database
            cursor.execute('''
            INSERT INTO sources (url, title, type, content, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (url, title, source_type, content, metadata, timestamp))
            
            source_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            self.logger.info("Source stored with ID: %s", source_id)
            return source_id
            
        except sqlite3.Error as e:
            self.logger.error("Error storing source: %s", str(e))
            raise RuntimeError("Failed to store source: %s" % str(e)) from e
            
    def get_source(self, source_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a source from the database by ID.
        
        Args:
            source_id: Source ID
            
        Returns:
            Source information dictionary or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM sources WHERE id = ?
            ''', (source_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
                
            # Convert to dictionary
            source = dict(row)
            
            # Parse JSON fields
            source["content"] = json.loads(source["content"])
            source["metadata"] = json.loads(source["metadata"])
            
            return source
            
        except sqlite3.Error as e:
            self.logger.error("Error retrieving source: %s", str(e))
            return None
            
    def find_data(self, filters: Dict[str, Any] = None, limit: int = 100) -> List[Any]:
        """
        Find data in the research_data table based on filters.
        
        Args:
            filters: Dictionary of field-value pairs to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of matching data items
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM research_data"
            params = []
            
            if filters and len(filters) > 0:
                conditions = []
                for field, value in filters.items():
                    if field in ["key", "data_type", "timestamp"]:
                        conditions.append(f"{field} LIKE ?")
                        params.append(f"%{value}%")
                        
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                    
            query += f" ORDER BY timestamp DESC LIMIT {limit}"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to list of items
            result_data = []
            for row in rows:
                data_dict = dict(row)
                value_str, data_type = data_dict["value"], data_dict["data_type"]
                
                # Convert based on data type
                if data_type == "json":
                    try:
                        result_data.append(json.loads(value_str))
                    except json.JSONDecodeError:
                        result_data.append(value_str)
                elif data_type == "number":
                    try:
                        if "." in value_str:
                            result_data.append(float(value_str))
                        else:
                            result_data.append(int(value_str))
                    except ValueError:
                        result_data.append(value_str)
                elif data_type == "boolean":
                    result_data.append(value_str.lower() == "true")
                else:
                    result_data.append(value_str)
                
            return result_data
            
        except sqlite3.Error as e:
            self.logger.error("Error finding data: %s", str(e))
            return []

    def find_data_by_key_pattern(self, key_pattern: str, limit: int = 100) -> Dict[str, Any]:
        """
        Find data by key pattern and return as a dictionary.
        
        Args:
            key_pattern: Pattern to match keys against
            limit: Maximum number of results
            
        Returns:
            Dictionary of matching key-value pairs
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT key, value, data_type FROM research_data 
            WHERE key LIKE ? ORDER BY timestamp DESC LIMIT ?
            ''', (f"%{key_pattern}%", limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            result = {}
            for key, value_str, data_type in rows:
                # Convert based on data type
                if data_type == "json":
                    try:
                        result[key] = json.loads(value_str)
                    except json.JSONDecodeError:
                        result[key] = value_str
                elif data_type == "number":
                    try:
                        if "." in value_str:
                            result[key] = float(value_str)
                        else:
                            result[key] = int(value_str)
                    except ValueError:
                        result[key] = value_str
                elif data_type == "boolean":
                    result[key] = value_str.lower() == "true"
                else:
                    result[key] = value_str
            
            return result
            
        except sqlite3.Error as e:
            self.logger.error("Error finding data by key pattern: %s", str(e))
            return {}
            
    def find_sources(self, filters: Dict[str, Any] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find sources in the database based on filters.
        
        Args:
            filters: Dictionary of field-value pairs to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of matching sources
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM sources"
            params = []
            
            if filters:
                conditions = []
                for key, value in filters.items():
                    if key in ["id", "url", "title", "type", "timestamp"]:
                        conditions.append(f"{key} = ?")
                        params.append(value)
                    elif key == "content_contains":
                        conditions.append("content LIKE ?")
                        params.append(f"%{value}%")
                    elif key == "metadata_contains":
                        conditions.append("metadata LIKE ?")
                        params.append(f"%{value}%")
                        
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                    
            query += f" ORDER BY timestamp DESC LIMIT {limit}"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to list of dictionaries
            sources = []
            for row in rows:
                source = dict(row)
                source["content"] = json.loads(source["content"])
                source["metadata"] = json.loads(source["metadata"])
                sources.append(source)
                
            return sources
            
        except sqlite3.Error as e:
            self.logger.error("Error finding sources: %s", str(e))
            return []
            
    def cache_search_results(self, query: str, engine: str, results: Dict[str, Any]) -> bool:
        """
        Cache search results.
        
        Args:
            query: Search query
            engine: Search engine used
            results: Search results
            
        Returns:
            True if successful, False otherwise
        """
        if not self.cache_enabled:
            return False
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate expiration time
            timestamp = datetime.now().isoformat()
            expires_at = (datetime.now() + timedelta(seconds=self.cache_ttl)).isoformat()
            
            # Check if query already exists
            cursor.execute('''
            SELECT id FROM search_cache WHERE query = ? AND engine = ?
            ''', (query, engine))
            
            existing_id = cursor.fetchone()
            
            if existing_id:
                # Update existing entry
                cursor.execute('''
                UPDATE search_cache 
                SET results = ?, timestamp = ?, expires_at = ?
                WHERE id = ?
                ''', (json.dumps(results), timestamp, expires_at, existing_id[0]))
            else:
                # Insert new entry
                cursor.execute('''
                INSERT INTO search_cache (query, engine, results, timestamp, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ''', (query, engine, json.dumps(results), timestamp, expires_at))
                
            conn.commit()
            conn.close()
            
            self.logger.info("Search results cached: %s - %s", query, engine)
            return True
            
        except sqlite3.Error as e:
            self.logger.error("Error caching search results: %s", str(e))
            return False
            
    def get_cached_search_results(self, query: str, engine: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached search results.
        
        Args:
            query: Search query
            engine: Search engine
            
        Returns:
            Cached search results or None if not found or expired
        """
        if not self.cache_enabled:
            return None
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT results, expires_at FROM search_cache 
            WHERE query = ? AND engine = ?
            ''', (query, engine))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
                
            results, expires_at = row
            
            # Check if expired
            if datetime.now() > datetime.fromisoformat(expires_at):
                self.logger.info("Cache expired for query: %s", query)
                return None
                
            return json.loads(results)
            
        except sqlite3.Error as e:
            self.logger.error("Error retrieving cached search results: %s", str(e))
            return None
            
    def store_data(self, key: str, value: Any, data_type: str = None) -> bool:
        """
        Store a data value in the database.
        
        Args:
            key: Data key
            value: Data value
            data_type: Optional data type hint
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Determine data type if not provided
            if data_type is None:
                if isinstance(value, (list, dict)):
                    data_type = "json"
                elif isinstance(value, (int, float)):
                    data_type = "number"
                elif isinstance(value, bool):
                    data_type = "boolean"
                # Перевіряємо чи це pandas DataFrame
                elif hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict', None)):
                    data_type = "json"
                    # Якщо DataFrame має метод to_dict, використовуємо його для конвертації
                    value = value.to_dict(orient='records')
                else:
                    data_type = "text"
                    
            # Convert value to string
            if data_type == "json":
                value_str = json.dumps(value)
            else:
                value_str = str(value)
                
            timestamp = datetime.now().isoformat()
            
            # Check if key already exists
            cursor.execute('''
            SELECT id FROM research_data WHERE key = ?
            ''', (key,))
            
            existing_id = cursor.fetchone()
            
            if existing_id:
                # Update existing entry
                cursor.execute('''
                UPDATE research_data 
                SET value = ?, data_type = ?, timestamp = ?
                WHERE id = ?
                ''', (value_str, data_type, timestamp, existing_id[0]))
            else:
                # Insert new entry
                cursor.execute('''
                INSERT INTO research_data (key, value, data_type, timestamp)
                VALUES (?, ?, ?, ?)
                ''', (key, value_str, data_type, timestamp))
                
            conn.commit()
            conn.close()
            
            self.logger.info("Data stored with key: %s", key)
            return True
            
        except sqlite3.Error as e:
            self.logger.error("Error storing data: %s", str(e))
            return False
            
    def get_data(self, key: str) -> Optional[Any]:
        """
        Retrieve data from the database.
        
        Args:
            key: Data key
            
        Returns:
            Data value or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT value, data_type FROM research_data WHERE key = ?
            ''', (key,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
                
            value_str, data_type = row
            
            # Convert based on data type
            if data_type == "json":
                return json.loads(value_str)
            elif data_type == "number":
                try:
                    if "." in value_str:
                        return float(value_str)
                    else:
                        return int(value_str)
                except ValueError:
                    return value_str
            elif data_type == "boolean":
                return value_str.lower() == "true"
            else:
                return value_str
                
        except sqlite3.Error as e:
            self.logger.error("Error retrieving data: %s", str(e))
            return None
            
    def delete_data(self, key: str) -> bool:
        """
        Delete data from the database.
        
        Args:
            key: Data key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            DELETE FROM research_data WHERE key = ?
            ''', (key,))
            
            deleted = cursor.rowcount > 0
            conn.commit()
            conn.close()
            
            if deleted:
                self.logger.info("Data deleted with key: %s", key)
            else:
                self.logger.info("Data not found with key: %s", key)
                
            return deleted
            
        except sqlite3.Error as e:
            self.logger.error("Error deleting data: %s", str(e))
            return False
            
    def list_data_keys(self, prefix: str = None) -> List[str]:
        """
        List all data keys in the database.
        
        Args:
            prefix: Optional key prefix to filter by
            
        Returns:
            List of keys
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if prefix:
                cursor.execute('''
                SELECT key FROM research_data WHERE key LIKE ?
                ''', (f"{prefix}%",))
            else:
                cursor.execute('''
                SELECT key FROM research_data
                ''')
                
            rows = cursor.fetchall()
            conn.close()
            
            # Extract keys
            keys = [row[0] for row in rows]
            return keys
            
        except sqlite3.Error as e:
            self.logger.error("Error listing data keys: %s", str(e))
            return []
            
    def store_research_session(self, query: str, results: Dict[str, Any], sources: List[Dict[str, Any]]) -> int:
        """
        Store a research session.
        
        Args:
            query: Research query
            results: Research results
            sources: Sources used
            
        Returns:
            Session ID or -1 if failed
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            
            cursor.execute('''
            INSERT INTO research_sessions (query, results, sources, timestamp)
            VALUES (?, ?, ?, ?)
            ''', (query, json.dumps(results), json.dumps(sources), timestamp))
            
            session_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            self.logger.info("Research session stored with ID: %s: %s", session_id, query)
            return session_id
            
        except sqlite3.Error as e:
            self.logger.error("Error storing research session: %s", str(e))
            return -1
            
    def get_research_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a research session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session information or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM research_sessions WHERE id = ?
            ''', (session_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
                
            # Convert to dictionary
            session = dict(row)
            
            # Parse JSON fields
            session["results"] = json.loads(session["results"])
            session["sources"] = json.loads(session["sources"])
            
            return session
            
        except sqlite3.Error as e:
            self.logger.error("Error retrieving research session: %s", str(e))
            return None
            
    def find_research_sessions(self, query_contains: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find research sessions.
        
        Args:
            query_contains: Text that the query should contain
            limit: Maximum number of results
            
        Returns:
            List of matching sessions
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            sql_query = "SELECT id, query, timestamp FROM research_sessions"
            params = []
            
            if query_contains:
                sql_query += " WHERE query LIKE ?"
                params.append(f"%{query_contains}%")
                
            sql_query += f" ORDER BY timestamp DESC LIMIT {limit}"
            
            cursor.execute(sql_query, params)
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to list of dictionaries
            sessions = []
            for row in rows:
                session = dict(row)
                sessions.append(session)
                
            return sessions
            
        except sqlite3.Error as e:
            self.logger.error("Error finding research sessions: %s", str(e))
            return []
            
    def export_database(self, export_path: str) -> bool:
        """
        Export the entire database to a file.
        
        Args:
            export_path: Path to export to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure export directory exists
            export_dir = os.path.dirname(export_path)
            if export_dir:
                os.makedirs(export_dir, exist_ok=True)
                
            # Create connection to source database
            conn = sqlite3.connect(self.db_path)
            
            # Export to the specified path
            with open(export_path, 'w', encoding='utf-8') as f:
                for line in conn.iterdump():
                    f.write("%s\n" % line)
                    
            conn.close()
            
            self.logger.info("Database exported to %s", export_path)
            return True
            
        except (sqlite3.Error, IOError) as e:
            self.logger.error("Error exporting database: %s", str(e))
            return False
            
    def save_results(self, results: Dict[str, Any]) -> int:
        """
        Save research results to the database and return the session ID.
        
        Args:
            results: Research results dictionary
            
        Returns:
            Session ID or -1 if failed
        """
        try:
            # Extract data from results
            query = results.get("query", "")
            sources = results.get("sources", [])
            
            # Store the research session
            session_id = self.store_research_session(query, results, sources)
            
            # Store the full report as a data item for easy retrieval
            if "full_report" in results:
                report_key = f"report_{session_id}_{int(time.time())}"
                self.store_data(report_key, results["full_report"], "text")
            
            return session_id
        except Exception as e:
            self.logger.error("Error saving results: %s", str(e))
            return -1
    
    def import_database(self, import_path: str, replace: bool = False) -> bool:
        """
        Import a database from a file.
        
        Args:
            import_path: Path to import from
            replace: Whether to replace the current database
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if import file exists
            if not os.path.exists(import_path):
                self.logger.error("Import file not found: %s", import_path)
                return False
                
            if replace:
                # Back up the current database if it exists
                if os.path.exists(self.db_path):
                    backup_path = f"{self.db_path}.backup-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    import shutil
                    shutil.copy2(self.db_path, backup_path)
                    self.logger.info("Backed up existing database to %s", backup_path)
                    
                # Remove the current database
                if os.path.exists(self.db_path):
                    os.remove(self.db_path)
                    
                # Ensure the directory exists
                os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
                
                # Create a new empty database
                conn = sqlite3.connect(self.db_path)
                
                # Import the SQL
                with open(import_path, 'r') as f:
                    sql_script = f.read()
                    
                conn.executescript(sql_script)
                conn.close()
                
                self.logger.info("Database imported from %s", import_path)
                return True
            else:
                # Import into a temporary database
                temp_db_path = "%s.temp" % self.db_path
                
                # Create a new temporary database
                temp_conn = sqlite3.connect(temp_db_path)
                
                # Import the SQL
                with open(import_path, 'r', encoding='utf-8') as f:
                    sql_script = f.read()
                    
                temp_conn.executescript(sql_script)
                
                # Copy data to the main database
                main_conn = sqlite3.connect(self.db_path)
                
                # Copy tables
                for table in ["sources", "search_cache", "research_data", "research_sessions"]:
                    try:
                        # Get all rows from the temporary database
                        temp_conn.row_factory = sqlite3.Row
                        temp_cursor = temp_conn.cursor()
                        temp_cursor.execute(f"SELECT * FROM {table}")
                        rows = temp_cursor.fetchall()
                        
                        # Insert into main database
                        for row in rows:
                            row_dict = dict(row)
                            
                            # Remove ID field for insertion
                            if "id" in row_dict:
                                del row_dict["id"]
                                
                            # Generate placeholders
                            placeholders = ", ".join(["?"] * len(row_dict))
                            fields = ", ".join(row_dict.keys())
                            
                            main_conn.execute(
                                f"INSERT OR IGNORE INTO {table} ({fields}) VALUES ({placeholders})",
                                list(row_dict.values())
                            )
                    except sqlite3.Error as e:
                        self.logger.warning("Error copying table %s: %s", table, str(e))
                        
                main_conn.commit()
                main_conn.close()
                temp_conn.close()
                
                # Remove temporary database
                os.remove(temp_db_path)
                
                self.logger.info("Database merged from %s", import_path)
                return True
                
        except (sqlite3.Error, IOError) as e:
            self.logger.error("Error importing database: %s", str(e))
            return False
