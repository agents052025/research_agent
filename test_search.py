#!/usr/bin/env python
"""
Test script for testing the search tool with BraveSearch API as the primary search engine.
"""

import json
import os
import logging
from tools.search_tool import SearchAPITool

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_config(config_path: str):
    """Завантаження конфігурації з файлу."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Помилка завантаження конфігурації: {str(e)}")
        return {}

def main():
    """Основна функція для тестування пошукового інструменту."""
    # Завантаження конфігурації
    config = load_config('config/config/default_config.json')
    search_config = config.get("tools", {}).get("search", {})
    
    # Ініціалізація пошукового інструменту
    search_tool = SearchAPITool(search_config)
    
    # Тестові запити (українською та англійською)
    test_queries = [
        "ринок смартфонів в Україні 2025",
        "дослідження впливу штучного інтелекту на економіку",
        "latest smartphone market trends 2025"
    ]
    
    # Перевірка поточних ключів API
    logger.info(f"Using BraveSearch API key: {'Yes' if os.environ.get('BRAVESEARCH_API_KEY') else 'No'}")
    logger.info(f"Using Serper API key: {'Yes' if os.environ.get('SERPER_API_KEY') else 'No'}")
    logger.info(f"Using SerpAPI key: {'Yes' if os.environ.get('SERPAPI_API_KEY') else 'No'}")
    
    # Виконання тестових запитів
    for query in test_queries:
        logger.info(f"\n\n=== Testing query: {query} ===")
        try:
            results = search_tool.forward(query)
            num_results = len(results.get("results", []))
            logger.info(f"Search successful: {num_results} results found")
            logger.info(f"Using engine: {results.get('metadata', {}).get('engine', 'unknown')}")
            
            # Виведення перших трьох результатів (якщо доступні)
            if num_results > 0:
                logger.info("First results:")
                for i, result in enumerate(results.get("results", [])[:3]):
                    logger.info(f"  {i+1}. {result.get('title', 'No title')} - {result.get('link', 'No link')}")
            
            # Перевірка, чи використовувався fallback
            if results.get('metadata', {}).get('fallback_used'):
                logger.info("Fallback search was used")
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")

if __name__ == "__main__":
    main()
