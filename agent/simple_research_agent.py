"""
Simple Research Agent - A lightweight implementation using OpenAI API directly.
"""

import os
import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
import requests  # Використання requests замість бібліотеки anthropic
from rich.console import Console

from core.research_planner import ResearchPlanner
from core.context_manager import ContextManager
from utils.language_strings import get_string

class SimpleResearchAgent:
    """
    A simple research agent that uses OpenAI API directly without smolagents dependency.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the research agent with configuration.
        
        Args:
            config: Dictionary containing agent configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.console = Console()
        
        # Set the language from config
        self.language = self.config.get("language", {}).get("primary", "uk")
        self.console.print(get_string("initializing", self.language))
        
        # Initialize context manager for maintaining state
        self.context = ContextManager()
        
        # Get LLM configuration
        self.llm_config = self.config.get("llm", {})
        
        # Initialize the research planner
        self.planner = ResearchPlanner(self.llm_config)
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        
        Args:
            text: Input text to detect language for
            
        Returns:
            Language code (uk or en)
        """
        # Simple language detection based on common Ukrainian characters
        ukrainian_chars = re.compile(r'[іїєґ]', re.IGNORECASE)
        if ukrainian_chars.search(text):
            return "uk"
        return "en"
    
    def process_query(self, query: str, progress_callback=None) -> Dict[str, Any]:
        """
        Process a research query and return results using Anthropic API.
        
        Args:
            query: Research query to process
            progress_callback: Optional callback function to report progress
            
        Returns:
            Dictionary containing research results
        """
        # Перевірка типу запиту
        if isinstance(query, dict):
            query = str(query)
        elif not isinstance(query, str):
            query = str(query) if query is not None else ""
            
        self.logger.info("Starting research on query: %s", query)
        self.console.print(get_string("processing_query", self.language))
        
        # Detect language if not specified
        detected_lang = self.detect_language(query)
        if detected_lang != self.language:
            self.logger.info("Detected language: %s, switching from %s", detected_lang, self.language)
            self.language = detected_lang
        
        # Create a research plan
        plan = self.planner.create_plan(query)
        
        # Create task data for the execution
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # Use Anthropic API directly
            api_key = self.llm_config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("No Anthropic API key found. Please set ANTHROPIC_API_KEY in your .env file")
                
            # Обов'язково використовуємо модель Claude, а не GPT
            model = "claude-3-haiku-20240307"  # Фіксоване значення моделі Anthropic
            temperature = self.llm_config.get("temperature", 0.7)
            
            # Prepare the system and user messages
            system_message = """
            You are a research assistant that helps users find information on various topics.
            For the given query, provide a comprehensive research report following these steps:
            
            1. Analyze the query and break it down into key components
            2. Provide relevant information for each component
            3. Include likely sources of information
            4. Summarize your findings
            
            Format your response as a Markdown document with the following sections:
            - Summary
            - Key Findings
            - Details
            - Sources
            """
            
            user_message = f"Please research the following query: {query}\n\nUse this research plan: {json.dumps(plan, ensure_ascii=False)}"
            
            # Track progress
            if progress_callback:
                progress_callback(20)  # Initial progress
            
            # Використаємо прямий HTTP запит через requests для більшого контролю
            self.logger.info(f"Sending request to Anthropic API using model {model}")
            
            # Підготовка даних для API запиту
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            # Дані запиту у форматі, якого очікує Anthropic API
            payload = {
                "model": model,
                "max_tokens": 4000,
                "temperature": temperature,
                "system": system_message,
                "messages": [
                    {"role": "user", "content": user_message}
                ]
            }
            
            # Журналювання деталей запиту для діагностики
            # Логування заголовків (з маскуванням самого ключа)
            safe_headers = headers.copy()
            if "x-api-key" in safe_headers:
                # Залишаємо тільки перші і останні 4 символи ключа для безпеки
                api_key_value = safe_headers["x-api-key"]
                if len(api_key_value) > 8:
                    masked_key = api_key_value[:4] + "*" * (len(api_key_value) - 8) + api_key_value[-4:]
                    safe_headers["x-api-key"] = masked_key
            
            self.logger.info(f"Request headers: {safe_headers}")
            self.logger.info(f"Request endpoint: https://api.anthropic.com/v1/messages")
            
            # Виконання запиту
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=60  # Додаємо таймаут для уникнення зависання запиту
            )
            
            if progress_callback:
                progress_callback(50)  # Progress after API request
            
            # Обробка відповіді
            if response.status_code == 200:
                result = response.json()
                content = result["content"][0]["text"]
            else:
                raise Exception(f"Error code: {response.status_code} - {response.text}")
            
            if progress_callback:
                progress_callback(80)  # Progress after receiving response
            
            # Process the response content
            research_results = self._format_results(content, query, timestamp)
            
            if progress_callback:
                progress_callback(100)  # Complete progress
            
            # Store results in context
            if hasattr(self.context, 'add_results'):
                self.context.add_results(research_results)
            else:
                # Fallback for older versions of ContextManager
                if not hasattr(self.context, 'results'):
                    self.context.results = []
                self.context.results.append(research_results)
            
            return research_results
                
        except Exception as e:
            self.logger.error("Error processing research query: %s", str(e))
            return {
                "query": query,
                "timestamp": timestamp,
                "error": str(e),
                "summary": get_string("error_occurred", self.language),
                "full_report": get_string("error_details", self.language).format(error=str(e))
            }
    
    def _format_results(self, content: str, query: str, timestamp: str) -> Dict[str, Any]:
        """
        Format the raw response content into a structured output.
        
        Args:
            content: Raw text response from the API
            query: Original research query
            timestamp: Timestamp when research was conducted
            
        Returns:
            Formatted research results dictionary
        """
        self.logger.info("Formatting results...")
        
        # Extracting sections from the response
        summary = ""
        sources = []
        findings = []
        
        # Try to extract summary
        summary_match = re.search(r'(?i)#*\s*Summary\s*[:\n]\s*(.+?)(?=\n\s*#|$)', content, re.DOTALL)
        if not summary_match:
            summary_match = re.search(r'(?i)#*\s*Резюме\s*[:\n]\s*(.+?)(?=\n\s*#|$)', content, re.DOTALL)
        
        if summary_match:
            summary = summary_match.group(1).strip()
        else:
            # If no summary is found, use the first paragraph
            first_paragraph = content.split('\n\n', 1)[0].strip()
            if len(first_paragraph) > 50:
                summary = first_paragraph
            else:
                summary = f"Дослідження за запитом '{query}' завершено."
        
        # Extract sources
        sources_section = re.search(r'(?i)#*\s*Sources|References|Джерела[:\n]\s*(.+?)(?=\n\s*#|$)', content, re.DOTALL)
        if sources_section:
            sources_text = sources_section.group(1).strip()
            # Extract URLs from text
            urls = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)|(?:https?://[^\s,"]+)', sources_text)
            for i, url_match in enumerate(urls):
                if isinstance(url_match, tuple):
                    title, url = url_match
                else:
                    title = f"Джерело {i+1}"
                    url = url_match
                
                sources.append({
                    "title": title,
                    "url": url,
                    "type": "web"
                })
        
        # Extract findings/insights
        findings_section = re.search(r'(?i)#*\s*(?:Findings|Key\s+Insights|Висновки)[:\n]\s*(.+?)(?=\n\s*#|$)', content, re.DOTALL)
        if findings_section:
            findings_text = findings_section.group(1).strip()
            # Split by bullet points or numbered items
            finding_points = re.split(r'\n\s*(?:•|\*|\d+\.\s+|-)\s+', findings_text)
            for i, point in enumerate(finding_points):
                if point.strip():
                    findings.append({
                        "topic": f"Висновок {i+1}",
                        "content": point.strip()
                    })
        
        # Create the final result structure
        research_results = {
            "query": query,
            "timestamp": timestamp,
            "summary": summary,
            "full_report": content,
            "sources": sources,
            "findings": findings,
            "section_titles": {
                "summary": get_string("results_summary", self.language),
                "full_report": get_string("results_full", self.language),
                "sources": get_string("results_sources", self.language),
                "findings": get_string("results_findings", self.language),
                "analysis": get_string("results_analysis", self.language),
                "limitations": get_string("results_limitations", self.language)
            }
        }
        
        return research_results
