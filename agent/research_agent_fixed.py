"""
Research Agent - Core agent implementation based on SmolaGents framework.
"""

import os
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional, Union

from smolagents import CodeAgent
from smolagents.models import OpenAIServerModel
from smolagents.tools import PythonInterpreterTool
from rich.console import Console

from core.research_planner import ResearchPlanner
from core.context_manager import ContextManager
from tools.search_tool import SearchAPITool
from tools.url_fetcher_tool import URLFetcherTool
from tools.pdf_reader_tool import PDFReaderTool
from tools.data_analysis_tool import DataAnalysisTool
from tools.visualization_tool import VisualizationTool
from tools.database_tool import DatabaseTool
from utils.language_strings import get_string


class ResearchAgent:
    """
    Main research agent class responsible for conducting end-to-end research
    based on user queries using SmolaGents framework.
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
        
        # Initialize tools
        self._initialize_tools()
        
        # Initialize the agent
        self.agent = self._initialize_agent()
    
    def _initialize_tools(self):
        """
        Initialize all the tools needed for research.
        """
        self.logger.info("Initializing research tools")
        
        # Initialize search tool
        search_config = self.config.get("tools", {}).get("search", {})
        self.search_tool = SearchAPITool(search_config)
        
        # Initialize URL fetcher tool
        url_config = self.config.get("tools", {}).get("url_fetcher", {})
        self.url_fetcher_tool = URLFetcherTool(url_config)
        
        # Initialize PDF reader tool
        pdf_config = self.config.get("tools", {}).get("pdf_reader", {})
        self.pdf_reader_tool = PDFReaderTool(pdf_config)
        
        # Initialize data analysis tool
        data_config = self.config.get("tools", {}).get("data_analysis", {})
        self.data_analysis_tool = DataAnalysisTool(data_config)
        
        # Initialize visualization tool
        viz_config = self.config.get("tools", {}).get("visualization", {})
        self.visualization_tool = VisualizationTool(viz_config)
        
        # Initialize database tool
        db_config = self.config.get("tools", {}).get("database", {})
        self.database_tool = DatabaseTool(db_config)
        
        # Create a list of all tools
        self.tools = [
            self.search_tool,
            self.url_fetcher_tool,
            self.pdf_reader_tool,
            self.data_analysis_tool,
            self.visualization_tool,
            self.database_tool,
            PythonInterpreterTool()
        ]
    
    def _initialize_agent(self):
        """
        Initialize the agent with the necessary tools and configuration.
        
        Returns:
            An instance of the research agent
        """
        self.logger.info("Initializing research agent")
        
        # Create an OpenAI model instance
        model = OpenAIServerModel(
            model_id=self.llm_config["model"],
            api_key=self.llm_config["api_key"],
            temperature=self.llm_config.get("temperature", 0.7)
        )
        
        # Create a simple agent for testing
        agent = CodeAgent(
            tools=self.tools,
            model=model
        )
        
        return agent
    
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
    
    def _generate_sample_research_code(self) -> str:
        """
        Генерує приклад коду для дослідження, який буде виконано агентом.
        
        Returns:
            Рядок з Python кодом для дослідження
        """
        code = '''
# Приклад коду для дослідження ринку смартфонів в Україні
import json

# Ініціалізація результатів дослідження
research_results = {
    "query": "Дослідження ринку смартфонів в Україні",
    "timestamp": "2023-06-15T10:30:00",
    "success": True
}

# Крок 1: Збір інформації
def gather_information():
    print("Збір інформації з різних джерел...")
    
    # Тут буде код для пошуку в Інтернеті, читання PDF, аналізу даних тощо
    
    # Приклад результатів збору інформації
    return {
        "success": True,
        "sources": [
            {
                "title": "Ринок смартфонів в Україні",
                "url": "https://example.com/smartphones-ukraine",
                "content": "Ринок смартфонів в Україні оцінюється приблизно в $500 млн з річним темпом зростання близько 15%."
            }
        ]
    }

# Крок 2: Аналіз даних
def analyze_data(gathered_info):
    print("Аналіз зібраних даних...")
    
    # Тут буде код для аналізу зібраної інформації
    
    # Приклад результатів аналізу
    return {
        "success": True,
        "insights": [
            "Ринок смартфонів в Україні зростає",
            "Найпопулярніші бренди: Samsung, Apple, Xiaomi"
        ],
        "trends": [
            "Зростання попиту на смартфони середнього цінового сегменту",
            "Збільшення частки китайських виробників"
        ],
        "statistics": {
            "market_size": "$500 млн",
            "growth_rate": "15%",
            "top_brands": ["Samsung", "Apple", "Xiaomi"]
        }
    }

# Крок 3: Синтез та генерація звіту
def generate_report(analysis_results, gathered_info):
    print("Синтез результатів та створення звіту...")
    
    # Створення комплексного звіту на основі плану дослідження
    # Включення всіх необхідних цитат та посилань
    
    # Приклад звіту
    summary = """Ринок смартфонів в Україні демонструє стабільне зростання з річним темпом близько 15%. 
    Найпопулярнішими брендами є Samsung, Apple та Xiaomi. Спостерігається тенденція до збільшення 
    попиту на смартфони середнього цінового сегменту та зростання частки китайських виробників."""
    
    full_report = """# Дослідження ринку смартфонів в Україні

## Вступ
Це дослідження аналізує поточний стан та тенденції розвитку ринку смартфонів в Україні.

## Основні висновки
- Ринок смартфонів в Україні оцінюється приблизно в $500 млн.
- Щорічний темп зростання ринку становить близько 15%.
- Найпопулярнішими брендами є Samsung, Apple та Xiaomi.
- Спостерігається зростання попиту на смартфони середнього цінового сегменту.
- Частка китайських виробників на ринку збільшується.

## Детальний аналіз
### Розмір та зростання ринку
Ринок смартфонів в Україні демонструє стабільне зростання протягом останніх років. 
За оцінками експертів, розмір ринку становить приблизно $500 млн з річним темпом зростання близько 15%.

### Популярні бренди
Найпопулярнішими брендами на українському ринку є:
1. Samsung
2. Apple
3. Xiaomi

### Тенденції
- Зростання попиту на смартфони середнього цінового сегменту
- Збільшення частки китайських виробників
- Підвищення інтересу до смартфонів з підтримкою 5G

## Висновки
Ринок смартфонів в Україні є динамічним та перспективним. Очікується, що тенденції зростання 
збережуться в найближчі роки, особливо з розвитком 5G-мереж та появою нових технологій.

## Джерела
- Дослідження ринку смартфонів в Україні (https://example.com/smartphones-ukraine)
"""
    
    return {
        "summary": summary,
        "full_report": full_report,
        "bibliography": [
            {
                "title": "Ринок смартфонів в Україні",
                "url": "https://example.com/smartphones-ukraine"
            }
        ]
    }

# Головне виконання
try:
    # Збір інформації
    gathered_info = gather_information()
    print(f"Зібрано джерел: {len(gathered_info.get('sources', []))}")
    
    # Аналіз даних
    if gathered_info["success"]:
        analysis_results = analyze_data(gathered_info)
        print(f"Знайдено інсайтів: {len(analysis_results.get('insights', []))}")
        
        # Генерація звіту
        if analysis_results["success"]:
            report = generate_report(analysis_results, gathered_info)
            
            # Оновлення результатів дослідження
            research_results["summary"] = report["summary"]
            research_results["full_report"] = report["full_report"]
            research_results["analysis"] = analysis_results
            research_results["sources"] = gathered_info["sources"]
            research_results["findings"] = analysis_results["insights"]
            
            print("Дослідження завершено успішно.")
    
    # Повернення результатів дослідження
    print(json.dumps(research_results, indent=2))
except Exception as e:
    print(f"Помилка під час дослідження: {str(e)}")
    research_results["error"] = str(e)

# Повернення результатів
research_results
'''
        return code
    
    def _format_results(self, agent_results: Dict[str, Any], query: str, timestamp: str) -> Dict[str, Any]:
        """
        Format the raw agent results into a structured output.
        
        Args:
            agent_results: Raw output from the agent
            query: Original research query
            timestamp: Timestamp when research was conducted
            
        Returns:
            Formatted research results dictionary
        """
        # У новій версії smolagents, результати повертаються в іншому форматі
        # Спробуємо витягнути результати з різних можливих місць
        research_results = {}
        
        # Перевіряємо, чи є 'final_answer' в результатах
        if agent_results and isinstance(agent_results, dict):
            # Спочатку перевіряємо, чи є 'final_answer' в результатах
            if 'final_answer' in agent_results:
                # Спробуємо розпарсити JSON з final_answer
                try:
                    import json
                    final_answer = agent_results['final_answer']
                    if isinstance(final_answer, str) and '{' in final_answer and '}' in final_answer:
                        # Витягуємо JSON з тексту
                        json_start = final_answer.find('{')
                        json_end = final_answer.rfind('}') + 1
                        json_str = final_answer[json_start:json_end]
                        research_results = json.loads(json_str)
                    elif isinstance(final_answer, dict):
                        research_results = final_answer
                except Exception as e:
                    self.logger.error(f"Error parsing final_answer as JSON: {str(e)}")
            # Якщо не вдалося отримати результати з final_answer, шукаємо в інших місцях
            if not research_results and 'logs' in agent_results:
                # Шукаємо в логах рядки, що містять JSON
                logs = agent_results['logs']
                if isinstance(logs, list):
                    for log in logs:
                        if isinstance(log, str) and '{' in log and '}' in log:
                            try:
                                import json
                                # Витягуємо JSON з тексту
                                json_start = log.find('{')
                                json_end = log.rfind('}') + 1
                                json_str = log[json_start:json_end]
                                parsed = json.loads(json_str)
                                if parsed and isinstance(parsed, dict) and 'summary' in parsed and 'full_report' in parsed:
                                    research_results = parsed
                                    break
                            except Exception as e:
                                continue
        
        # Якщо все ще немає результатів, створюємо порожній результат
        if not research_results:
            research_results = {
                "query": query,
                "timestamp": timestamp,
                "summary": get_string("no_results", self.language),
                "full_report": get_string("no_results", self.language)
            }
        
        # Переконуємося, що всі очікувані поля присутні
        if "query" not in research_results:
            research_results["query"] = query
            
        if "timestamp" not in research_results:
            research_results["timestamp"] = timestamp
            
        if "summary" not in research_results or not research_results["summary"]:
            research_results["summary"] = get_string("no_results", self.language)
            
        if "full_report" not in research_results or not research_results["full_report"]:
            research_results["full_report"] = get_string("no_results", self.language)
            
        # Додаємо локалізовані заголовки розділів
        research_results["section_titles"] = {
            "summary": get_string("results_summary", self.language),
            "full_report": get_string("results_full", self.language),
            "sources": get_string("results_sources", self.language),
            "findings": get_string("results_findings", self.language),
            "analysis": get_string("results_analysis", self.language),
            "limitations": get_string("results_limitations", self.language)
        }
        
        return research_results
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a research query and return results.
        
        Args:
            query: Research query to process
            
        Returns:
            Dictionary containing research results
        """
        self.logger.info(f"Processing research query: {query}")
        self.console.print(get_string("processing_query", self.language))
        
        # Detect language if not specified
        detected_lang = self.detect_language(query)
        if detected_lang != self.language:
            self.logger.info(f"Detected language: {detected_lang}, switching from {self.language}")
            self.language = detected_lang
        
        # Generate timestamp
        timestamp = datetime.now().isoformat()
        
        # Store query in context
        self.context.add_query(query, timestamp)
        
        # Generate research code
        research_code = self._generate_sample_research_code()
        
        # Create a task for the agent
        task = {
            "query": query,
            "language": self.language,
            "timestamp": timestamp,
            "code": research_code
        }
        
        # Execute the research task
        try:
            # Execute the research task using the agent
            agent_results = self.agent.execute_task(task)
            
            # Format the results
            research_results = self._format_results(agent_results, query, timestamp)
            
            # Store results in context
            self.context.add_results(research_results)
            
            # Return formatted results
            return research_results
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                "query": query,
                "timestamp": timestamp,
                "error": str(e),
                "summary": get_string("error_occurred", self.language),
                "full_report": get_string("error_details", self.language).format(error=str(e))
            }
