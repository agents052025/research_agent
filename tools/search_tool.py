"""
Search API Tool for the Research Agent.
Enables the agent to perform web searches via various search engines.
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime
from smolagents import Tool

# API ключі з середовища
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
BRAVESEARCH_API_KEY = os.environ.get("BRAVESEARCH_API_KEY", "")


class SearchAPITool(Tool):
    # Атрибути для smolagents.Tool
    name = "web_search"
    description = """
    Provides web search capabilities via various search APIs.
    Supports multiple search engines including Google (via SerpAPI) and Brave Search.
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "Search query to perform web search",
        },
        "engine": {
            "type": "string",
            "description": "Search engine to use (google, brave, duckduckgo)",
            "nullable": True
        }
    }
    output_type = "object"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, api_key: Optional[str] = None, engine: str = "google", max_results: int = 10):
        """
        Initialize the Search API Tool.
        
        Args:
            config: Configuration dictionary that can include api_key_env, engine, max_results, and fallback settings
            api_key: API key for the search service (overrides config if provided)
            engine: Search engine to use (google, brave, serper) (overrides config if provided)
            max_results: Maximum number of results to return (overrides config if provided)
        """
        self.logger = logging.getLogger(__name__)
        
        # Обробка конфігурації або параметрів конструктора
        if isinstance(config, dict):
            # Отримання API ключа з змінної середовища, якщо вказано
            api_key_env = config.get("api_key_env")
            if api_key_env and not api_key:
                api_key = os.environ.get(api_key_env, "")
                if api_key:
                    self.logger.info(f"Using API key from environment variable {api_key_env}")
            
            # Отримання параметрів з конфігурації, якщо не передані явно
            self.api_key = api_key or config.get("api_key", "")
            self.engine = (engine or config.get("engine", "google")).lower()
            self.max_results = max_results or config.get("max_results", 10)
            
            # Конфігурація fallback пошукової системи
            self.fallback_config = config.get("fallback", {})
            self.fallback_engine = self.fallback_config.get("engine", "google").lower()
            
            # Отримання fallback API ключа з змінної середовища, якщо вказано
            fallback_api_key_env = self.fallback_config.get("api_key_env")
            if fallback_api_key_env:
                self.fallback_api_key = os.environ.get(fallback_api_key_env, "")
                if self.fallback_api_key:
                    self.logger.info(f"Using fallback API key from environment variable {fallback_api_key_env}")
            else:
                self.fallback_api_key = self.fallback_config.get("api_key", "")
        else:
            # Використання параметрів конструктора напряму
            self.api_key = api_key
            self.engine = engine.lower()
            self.max_results = max_results
            self.fallback_engine = "serper"
            self.fallback_api_key = SERPER_API_KEY  # Використовуємо Serper API як fallback за замовчуванням
        
        # Validate engine selection
        if self.engine not in ["google", "brave", "duckduckgo", "serper"]:
            self.logger.warning("Unsupported search engine: %s, falling back to google", engine)
            self.engine = "google"
        
        # Додаємо атрибут is_initialized для сумісності з smolagents 1.15.0
        self.is_initialized = True
        
        # Set up API endpoints based on engine
        self.endpoints = {
            "google": "https://serpapi.com/search",
            "brave": "https://api.search.brave.com/res/v1/web/search",
            "duckduckgo": "https://api.duckduckgo.com/",
            "serper": "https://google.serper.dev/search"
        }
        
        # Validate API key
        if not self.api_key and self.engine in ["google", "brave", "serper"]:
            self.logger.warning("No API key provided for %s search", self.engine)
            
        # Якщо двигун - serper, але API ключ не вказаний, спробуємо взяти з середовища
        if self.engine == "serper" and not self.api_key and SERPER_API_KEY:
            self.api_key = SERPER_API_KEY
            self.logger.info("Using Serper API key from environment variables")
            
        # Видалено згадки про SERPAPI_KEY
            
        # Якщо двигун - brave, але API ключ не вказаний, спробуємо взяти з середовища
        if self.engine == "brave" and not self.api_key and BRAVESEARCH_API_KEY:
            self.api_key = BRAVESEARCH_API_KEY
            self.logger.info("Using BraveSearch API key from environment variables")
            
    def _detect_language(self, text: str) -> str:
        """
        Визначає мову тексту на основі специфічних символів та лексичних маркерів.
        
        Args:
            text: Текст для аналізу
            
        Returns:
            Код мови (uk, ru, en тощо)
        """
        # Унікальні символи української мови
        ukrainian_chars = "їєіґщийегфвапролджзхкҗ"
        ukrainian_words = ["та", "й", "з", "в", "у", "це", "що", "як"]
        
        # Перевіряємо українську мову
        for char in ukrainian_chars:
            if char in text.lower():
                return "uk"
                
        for word in ukrainian_words:
            if re.search(r'\b' + word + r'\b', text.lower()):
                return "uk"
        
        # Отримуємо англійську за замовчуванням
        return "en"
    
    def forward(self, query: str, engine: str = None) -> Dict[str, Any]:
        """
        Perform a web search with the specified query. This is the required method for smolagents.Tool.
        
        Args:
            query: Search query
            engine: Search engine to use (google, brave, duckduckgo, serper)
        
        Returns:
            Dictionary with search results
        """
        self.logger.info("Performing web search for: %s", query)
        
        # Додаємо спеціальну обробку українських запитів
        query_lang = self._detect_language(query)
        self.logger.info("Detected query language: %s", query_lang)
        
        # Пріоритет пошукових систем для українських запитів: 
        # 1. BraveSearch API (якщо явно вказано або доступний для української мови)
        # 2. Serper API (як fallback)
        use_fallback = False
        
        # Для українських запитів завжди використовуємо пріоритетний API, незалежно від того, який параметр engine передано
        if query_lang == "uk":
            # Перевизначаємо двигун для українських запитів, навіть якщо його явно вказано
            self.logger.info("Ukrainian query detected, overriding engine parameter")
            if BRAVESEARCH_API_KEY:  # Спочатку пробуємо BraveSearch API
                self.logger.info("Using BraveSearch API")
                engine = "brave"
            elif SERPER_API_KEY:  # Якщо BraveSearch недоступний, використовуємо Serper API
                self.logger.info("Using Serper API (fallback)")
                engine = "serper"
                use_fallback = True
            else:  # Якщо жоден ключ недоступний
                self.logger.warning("No API keys available for search engines")
                return {"error": "No API keys available", "results": []}
        
        # Створюємо додаткові параметри для українських запитів
        kwargs = {}
        if query_lang == "uk":
            # Додаємо параметри для кращого пошуку українською
            kwargs = {
                "gl": "ua",      # Георегіон - Україна
                "hl": "uk",      # Мова інтерфейсу - українська
                "lr": "lang_uk"  # Обмеження результатів українською мовою
            }
            # Якщо запит містить слово 'дослідження' або 'досліди', додаємо додаткові параметри для наукового пошуку
            if "дослід" in query.lower():
                kwargs["as_sitesearch"] = ".ua,.org,.edu"
                
            self.logger.info("Added Ukrainian search parameters: %s", kwargs)
        
        # Використовуємо вказаний двигун, якщо він був переданий у параметрах
        original_engine = self.engine
        original_api_key = self.api_key
        
        # Якщо ми використовуємо fallback для українського запиту
        if use_fallback and engine == "google" and hasattr(self, 'fallback_api_key') and self.fallback_api_key:
            self.api_key = self.fallback_api_key
            self.logger.info("Using fallback API key for %s", engine)
        
        # Встановлюємо пошуковий двигун
        if engine is not None:
            self.engine = engine.lower()
        
        try:
            return self.search(query, **kwargs)
        except Exception as e:
            self.logger.error(f"Search failed with engine {self.engine}: {str(e)}")
            
            # Якщо сталася помилка і ми ще не використовували fallback
            if not use_fallback and hasattr(self, 'fallback_engine') and self.fallback_engine != self.engine:
                self.logger.info(f"Trying fallback search with {self.fallback_engine}")
                return self._search_with_fallback(query, **kwargs)
            else:
                # Повертаємо порожній результат з повідомленням про помилку
                return {
                    "error": str(e),
                    "results": [],
                    "metadata": {
                        "query": query,
                        "engine": self.engine,
                        "timestamp": datetime.now().isoformat(),
                        "error": True
                    }
                }
        finally:
            # Відновлюємо оригінальні налаштування
            if engine is not None:
                self.engine = original_engine
                self.api_key = original_api_key
    
    def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Perform a web search with the specified query.
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
        
        Returns:
            Dictionary containing search results
        
        Raises:
            RuntimeError: If search fails and no fallback is available
        """
        search_method = getattr(self, f"_search_{self.engine}", None)
        if not search_method:
            self.logger.warning(f"Search method not implemented for engine: {self.engine}")
            # Перевіряємо, чи можемо використати fallback
            if hasattr(self, 'fallback_engine') and self.fallback_engine != self.engine:
                self.logger.info(f"Falling back to {self.fallback_engine} search engine")
                return self._search_with_fallback(query, **kwargs)
            else:
                raise RuntimeError(f"Search method not implemented for engine: {self.engine}")
        
        try:
            results = search_method(query, **kwargs)
            
            # Add metadata
            results["metadata"] = {
                "query": query,
                "engine": self.engine,
                "timestamp": datetime.now().isoformat(),
                "result_count": len(results.get("results", [])),
                "parameters": kwargs
            }
            
            self.logger.info("Search returned %s results", len(results.get('results', [])))
            return results
        except Exception as e:
            self.logger.error(f"Search error with {self.engine}: {str(e)}")
            
            # Спроба використання fallback пошукової системи
            if hasattr(self, 'fallback_engine') and self.fallback_engine != self.engine:
                self.logger.info(f"Trying fallback search engine: {self.fallback_engine}")
                return self._search_with_fallback(query, **kwargs)
            else:
                raise RuntimeError(f"Search failed: {str(e)}")
    
    def _search_with_fallback(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Perform a search using the fallback search engine.
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
        
        Returns:
            Dictionary containing search results
        
        Raises:
            RuntimeError: If fallback search also fails
        """
        # Зберігаємо оригінальний API ключ та двигун
        original_engine = self.engine
        original_api_key = self.api_key
        
        try:
            # Встановлюємо параметри fallback пошукової системи
            self.engine = self.fallback_engine
            if hasattr(self, 'fallback_api_key') and self.fallback_api_key:
                self.api_key = self.fallback_api_key
            
            # Виконуємо пошук через fallback систему
            search_method = getattr(self, f"_search_{self.fallback_engine}", None)
            if not search_method:
                raise RuntimeError(f"Fallback search method not implemented for engine: {self.fallback_engine}")
            
            results = search_method(query, **kwargs)
            
            # Add metadata with fallback flag
            results["metadata"] = {
                "query": query,
                "engine": self.fallback_engine,
                "timestamp": datetime.now().isoformat(),
                "result_count": len(results.get("results", [])),
                "parameters": kwargs,
                "fallback_used": True
            }
            
            self.logger.info(f"Fallback search with {self.fallback_engine} returned %s results", 
                             len(results.get('results', [])))
            return results
        except Exception as e:
            self.logger.error(f"Fallback search with {self.fallback_engine} also failed: {str(e)}")
            raise RuntimeError(f"All search methods failed. Last error: {str(e)}")
        finally:
            # Відновлюємо оригінальні налаштування
            self.engine = original_engine
            self.api_key = original_api_key
    
    def _search_google(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Perform a Google search via SerpAPI.
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
            
        Returns:
            Dictionary containing search results
        """
        if not self.api_key:
            raise RuntimeError("SerpAPI key is required for Google search")
            
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": min(self.max_results, 100)  # SerpAPI limit
        }
        
        # Add additional parameters
        for key, value in kwargs.items():
            # Запобігаємо дублюванню ключа API
            if key != 'api_key':
                params[key] = value
            
        try:
            response = requests.get(self.endpoints["google"], params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract and format organic results
            results = []
            if "organic_results" in data:
                for item in data["organic_results"][:self.max_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "position": item.get("position", 0),
                        "source": "google"
                    })
                    
            # Include knowledge graph if available
            knowledge_graph = None
            if "knowledge_graph" in data:
                knowledge_graph = {
                    "title": data["knowledge_graph"].get("title", ""),
                    "type": data["knowledge_graph"].get("type", ""),
                    "description": data["knowledge_graph"].get("description", ""),
                    "source": "google_knowledge_graph"
                }
                
            return {
                "results": results,
                "knowledge_graph": knowledge_graph,
                "total_results": data.get("search_information", {}).get("total_results", 0)
            }
        except requests.RequestException as e:
            self.logger.error("SerpAPI request error: %s", str(e))
            raise RuntimeError(f"Google search failed: {str(e)}")
            
    def _search_serper(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Perform a Google search via Serper API.
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
            
        Returns:
            Dictionary containing search results
        """
        if not self.api_key:
            raise RuntimeError("Serper API key is required for search")
            
        # Підготовка запиту до Serper API
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Створення тіла запиту з параметрами пошуку
        payload = {
            "q": query
        }
        
        # Додавання додаткових параметрів для українською мови, якщо потрібно
        language_params = kwargs.get("gl", None) 
        if language_params:
            payload["gl"] = kwargs.get("gl")
            payload["hl"] = kwargs.get("hl")
        
        try:
            # Виконання POST запиту до Serper API
            response = requests.post(
                self.endpoints["serper"],
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Обробка та форматування результатів
            results = []
            
            # Органічні результати пошуку
            if "organic" in data:
                for i, item in enumerate(data["organic"][:self.max_results]):
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "position": i + 1,
                        "source": "serper_google"
                    })
            
            # Knowledge Graph (якщо є)
            knowledge_graph = None
            if "knowledgeGraph" in data:
                knowledge_graph = {
                    "title": data["knowledgeGraph"].get("title", ""),
                    "type": data["knowledgeGraph"].get("type", ""),
                    "description": data["knowledgeGraph"].get("description", ""),
                    "source": "serper_knowledge_graph"
                }
            
            return {
                "results": results,
                "knowledge_graph": knowledge_graph,
                "total_results": data.get("searchParameters", {}).get("totalResults", 0)
            }
            
        except requests.RequestException as e:
            self.logger.error("Serper API request error: %s", str(e))
            raise RuntimeError(f"Serper search failed: {str(e)}")
    
    def _search_brave(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Perform a search via Brave Search API.
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
            
        Returns:
            Dictionary containing search results
        """
        if not self.api_key:
            raise RuntimeError("Brave Search API key is required")
            
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }
        
        params = {
            "q": query,
            "count": min(self.max_results, 20)  # Brave API limit
        }
        
        # Add additional parameters
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value
                
        try:
            response = requests.get(
                self.endpoints["brave"],
                headers=headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract and format results
            results = []
            if "web" in data and "results" in data["web"]:
                for i, item in enumerate(data["web"]["results"][:self.max_results]):
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("url", ""),
                        "snippet": item.get("description", ""),
                        "position": i + 1,
                        "source": "brave"
                    })
                    
            return {
                "results": results,
                "total_results": data.get("web", {}).get("totalResults", 0)
            }
        except requests.RequestException as e:
            self.logger.error("Brave request error: %s", str(e))
            raise RuntimeError("Brave search failed: " + str(e))
        
    def _search_duckduckgo(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Perform a search via DuckDuckGo.
        Note: DuckDuckGo doesn't have an official search API, this uses their instant answer API.
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
            
        Returns:
            Dictionary containing search results
        """
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        
        try:
            response = requests.get(
                self.endpoints["duckduckgo"],
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract and format results
            results = []
            
            # Add instant answer if available
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading", ""),
                    "link": data.get("AbstractURL", ""),
                    "snippet": data.get("AbstractText", ""),
                    "position": 1,
                    "source": "duckduckgo_abstract"
                })
                
            # Add related topics
            if "RelatedTopics" in data:
                for i, topic in enumerate(data["RelatedTopics"][:self.max_results-len(results)]):
                    # Skip topics without Text
                    if "Text" not in topic:
                        continue
                        
                    link = topic.get("FirstURL", "")
                    title = topic.get("Text", "").split(" - ")[0] if " - " in topic.get("Text", "") else topic.get("Text", "")
                    snippet = topic.get("Text", "").split(" - ")[1] if " - " in topic.get("Text", "") else ""
                    
                    results.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet,
                        "position": len(results) + 1,
                        "source": "duckduckgo_related"
                    })
                    
            return {
                "results": results,
                "instant_answer": {
                    "title": data.get("Heading", ""),
                    "text": data.get("AbstractText", ""),
                    "source": data.get("AbstractSource", "")
                } if data.get("AbstractText") else None
            }
        except requests.RequestException as e:
            self.logger.error("DuckDuckGo request error: %s", str(e))
            raise RuntimeError(f"DuckDuckGo search failed: {str(e)}")
            
    def search_multi_engine(self, query: str, engines: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Perform searches across multiple engines and combine results.
        
        Args:
            query: Search query
            engines: List of engines to use
            **kwargs: Additional search parameters
            
        Returns:
            Dictionary containing combined search results
        """
        if not engines:
            engines = ["google"]  # Default to Google only
            
        all_results = []
        errors = []
        
        # Restore original engine
        self.engine = original_engine
        
        # Return combined results
        return {
            "results": all_results[:self.max_results],
            "metadata": {
                "query": query,
                "engines": engines,
                "timestamp": datetime.now().isoformat(),
                "result_count": len(all_results),
                "errors": errors
            }
        }
