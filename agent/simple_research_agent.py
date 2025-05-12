"""
Simple Research Agent - Implementation using smolagents framework.
"""

import os
import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from rich.console import Console

# Імпорт smolagents
from smolagents import LiteLLMModel, CodeAgent, Tool, PythonInterpreterTool

from core.research_planner import ResearchPlanner
from core.context_manager import ContextManager
from tools.search_tool import SearchAPITool
from tools.url_fetcher_tool import URLFetcherTool
from tools.pdf_reader_tool import PDFReaderTool
from utils.language_strings import get_string

class SimpleResearchAgent:
    """
    A simple research agent that uses smolagents framework.
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
        
        # Initialize tools
        self._initialize_tools()
        
        # Initialize the agent
        self.agent = self._initialize_agent()
    
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
    
    def _initialize_tools(self):
        """
        Initialize all the tools needed for research.
        """
        self.logger.info("Initializing research tools")
        
        # Initialize search tool
        try:
            search_config = self.config.get("tools", {}).get("search", {})
            self.search_tool = SearchAPITool(search_config)
            self.logger.info("Search tool initialized")
        except Exception as e:
            self.logger.error("Error initializing search tool: %s", str(e))
            self.search_tool = None
            
        # Initialize URL fetcher tool
        try:
            url_fetcher_config = self.config.get("tools", {}).get("url_fetcher", {})
            self.url_fetcher_tool = URLFetcherTool(url_fetcher_config)
            self.logger.info("URL fetcher tool initialized")
        except Exception as e:
            self.logger.error("Error initializing URL fetcher tool: %s", str(e))
            self.url_fetcher_tool = None
            
        # Initialize PDF reader tool
        try:
            pdf_reader_config = self.config.get("tools", {}).get("pdf_reader", {})
            self.pdf_reader_tool = PDFReaderTool(pdf_reader_config)
            self.logger.info("PDF reader tool initialized")
        except Exception as e:
            self.logger.error("Error initializing PDF reader tool: %s", str(e))
            self.pdf_reader_tool = None
            
        # Створення списку інструментів з правильною перевіркою типів
        self.tools = []
        
        # Додаємо інструменти тільки якщо вони є екземплярами Tool
        for tool in [self.search_tool, self.url_fetcher_tool, self.pdf_reader_tool]:
            if tool is not None and (isinstance(tool, Tool) or issubclass(type(tool), Tool)):
                self.tools.append(tool)
            elif tool is not None:
                self.logger.warning("Інструмент %s не є екземпляром Tool і не буде доданий", type(tool).__name__)
                
        # Додаємо PythonInterpreterTool
        try:
            python_tool = PythonInterpreterTool()
            if isinstance(python_tool, Tool) or issubclass(type(python_tool), Tool):
                self.tools.append(python_tool)
            else:
                self.logger.warning("PythonInterpreterTool не є екземпляром Tool і не буде доданий")
        except Exception as e:
            self.logger.error("Помилка ініціалізації PythonInterpreterTool: %s", str(e))
            
    def _initialize_agent(self):
        """
        Initialize the agent with the necessary tools and configuration.
        
        Returns:
            An instance of the research agent
        """
        self.logger.info("Initializing research agent")
        
        # Встановлюємо змінні середовища для API ключів
        # Згідно з документацією LiteLLM, це правильний спосіб налаштування
        # Визначаємо модель для використання
        model_name = self.llm_config.get("model", "claude-3-haiku-20240307")
        
        # Встановлюємо відповідні змінні середовища для API ключів
        if "claude" in model_name.lower():
            # Для моделей Anthropic
            anthropic_api_key = self.llm_config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
            if not anthropic_api_key:
                raise ValueError("No Anthropic API key found. Please set ANTHROPIC_API_KEY in your .env file")
            
            # Встановлюємо змінну середовища для Anthropic
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
            
            # Використовуємо правильний формат моделі
            model_name = model_name  # Не потрібно додавати префікс "anthropic/"
        else:
            # Для моделей OpenAI
            openai_api_key = self.llm_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in your .env file")
            
            # Встановлюємо змінну середовища для OpenAI
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Ініціалізуємо LLM модель з правильними параметрами
        self.logger.info(f"Initializing LiteLLM model: {model_name}")
        llm = LiteLLMModel(
            model=model_name,
            temperature=self.llm_config.get("temperature", 0.7)
        )
        
        # Create the agent with the tools
        agent = CodeAgent(
            model=llm,  # CodeAgent очікує параметр 'model', а не 'llm'
            tools=self.tools
            # Параметр verbose не підтримується в CodeAgent
        )
        
        return agent
        
    def process_query(self, query: str, progress_callback=None) -> Dict[str, Any]:
        """
        Process a research query and return results using smolagents framework.
        
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
            # Prepare the system and user messages for the agent
            system_message = """
            You are a research assistant that helps users find information on various topics.
            Your task is to conduct thorough research on the given query and provide a comprehensive report.
            
            Follow these guidelines:
            1. Break down the query into key components
            2. Search for relevant information from reliable sources
            3. Analyze and synthesize the information
            4. Provide a well-structured report with the following sections:
               - Summary: A concise overview of your findings
               - Findings: Detailed information organized by subtopics
               - Sources: List of sources used in your research
            
            Be objective, thorough, and provide accurate information with proper citations.
            """
            
            user_message = f"""
            Research query: {query}
            
            Research plan:
            {json.dumps(plan, indent=2, ensure_ascii=False)}
            
            Please conduct thorough research on this topic and provide a comprehensive report.
            Use the research plan as a guide but feel free to adjust it as needed based on your findings.
            
            For Ukrainian queries, please provide the response in Ukrainian.
            For English queries, please provide the response in English.
            """
            
            # Report progress
            if progress_callback:
                progress_callback(0.1)  # Research started
                
            # Run the agent with the query
            self.logger.info("Running research agent")
            # CodeAgent.run() не приймає аргументи system_message та user_message
            # Замість цього передаємо запит напряму
            prompt = f"{system_message}\n\n{user_message}"
            agent_response = self.agent.run(prompt)
            
            # Report progress
            if progress_callback:
                progress_callback(0.7)  # Research completed, formatting results
            
            # Отримуємо результат від агента
            content = agent_response
            
            if progress_callback:
                progress_callback(0.8)  # Progress after receiving response
            
            # Process the response content
            research_results = self._format_results(content, query, timestamp)
            
            if progress_callback:
                progress_callback(1.0)  # Complete progress
            
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
