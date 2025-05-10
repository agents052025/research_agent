"""
Search API Tool for the Research Agent.
Enables the agent to perform web searches via various search engines.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime


class SearchAPITool:
    """
    Provides web search capabilities via various search APIs.
    Supports multiple search engines including Google (via SerpAPI) and Brave Search.
    """
    
    def __init__(self, api_key: Optional[str] = None, engine: str = "google", max_results: int = 10):
        """
        Initialize the Search API Tool.
        
        Args:
            api_key: API key for the search service
            engine: Search engine to use (google, brave)
            max_results: Maximum number of results to return
        """
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.engine = engine.lower()
        self.max_results = max_results
        
        # Validate engine selection
        if self.engine not in ["google", "brave", "duckduckgo"]:
            self.logger.warning(f"Unsupported search engine: {engine}, falling back to google")
            self.engine = "google"
            
        # Set up API endpoints based on engine
        self.endpoints = {
            "google": "https://serpapi.com/search",
            "brave": "https://api.search.brave.com/res/v1/web/search",
            "duckduckgo": "https://api.duckduckgo.com/"
        }
        
        # Validate API key
        if not self.api_key and self.engine in ["google", "brave"]:
            self.logger.warning(f"No API key provided for {self.engine} search")
            
    def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Perform a web search with the specified query.
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
        
        Returns:
            Dictionary containing search results
        
        Raises:
            RuntimeError: If search fails
        """
        self.logger.info(f"Performing {self.engine} search for: {query}")
        
        search_method = getattr(self, f"_search_{self.engine}", None)
        if not search_method:
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
            
            self.logger.info(f"Search returned {len(results.get('results', []))} results")
            return results
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            raise RuntimeError(f"Search failed: {str(e)}")
    
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
            self.logger.error(f"SerpAPI request error: {str(e)}")
            raise RuntimeError(f"Google search failed: {str(e)}")
            
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
            if key in ["country", "search_lang", "text_search_strategy"]:
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
                for item in data["web"]["results"][:self.max_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("url", ""),
                        "snippet": item.get("description", ""),
                        "position": len(results) + 1,
                        "source": "brave"
                    })
                    
            return {
                "results": results,
                "total_results": data.get("web", {}).get("totalResults", 0)
            }
        except requests.RequestException as e:
            self.logger.error(f"Brave Search API request error: {str(e)}")
            raise RuntimeError(f"Brave search failed: {str(e)}")
            
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
            self.logger.error(f"DuckDuckGo request error: {str(e)}")
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
        
        # Save current engine
        original_engine = self.engine
        
        for engine in engines:
            try:
                # Temporarily switch engine
                self.engine = engine
                
                # Perform search
                results = self.search(query, **kwargs)
                
                # Add engine results
                if "results" in results:
                    for result in results["results"]:
                        # Avoid duplicates by checking URLs
                        if not any(r["link"] == result["link"] for r in all_results):
                            all_results.append(result)
            except Exception as e:
                errors.append({"engine": engine, "error": str(e)})
                
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
