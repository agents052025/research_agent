"""
Research Agent - Core agent implementation based on SmolaGents framework.
"""

import os
import json
import logging
import re
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from uuid import uuid4
from rich.console import Console

# Підключення smolagents
from smolagents import LiteLLMModel, CodeAgent, Tool, PythonInterpreterTool

# Імпорт внутрішніх модулів
from core.research_planner import ResearchPlanner
from core.context_manager import ContextManager
from agent.simple_research_agent import SimpleResearchAgent
import agent.improved_prompt as prompt_utils

# We don't need any custom model class since we're using the standard OpenAI library 0.28.0 
# which is compatible with smolagents
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
        
        # Перевірка, чи всі інструменти є підкласами Tool
        # Створення списку інструментів з правильною перевіркою типів
        valid_tools = []
        
        # Додаємо інструменти тільки якщо вони є екземплярами Tool
        for tool in [self.search_tool, self.url_fetcher_tool, self.pdf_reader_tool, 
                    self.data_analysis_tool, self.visualization_tool, self.database_tool]:
            if isinstance(tool, Tool) or issubclass(type(tool), Tool):
                valid_tools.append(tool)
            else:
                self.logger.warning("Інструмент %s не є екземпляром Tool і не буде доданий", type(tool).__name__)
                
        # Додаємо PythonInterpreterTool
        try:
            python_tool = PythonInterpreterTool()
            if isinstance(python_tool, Tool) or issubclass(type(python_tool), Tool):
                valid_tools.append(python_tool)
            else:
                self.logger.warning("PythonInterpreterTool не є екземпляром Tool і не буде доданий")
        except Exception as e:
            self.logger.error("Помилка ініціалізації PythonInterpreterTool: %s", str(e))
            
        self.tools = valid_tools
        
        # Initialize the research planner
        self.planner = ResearchPlanner(self.llm_config)
        
        # Initialize the agent
        self.agent = self._initialize_agent()
    
    def _initialize_tools(self):
        """
        Initialize all the tools needed for research.
        """
        self.logger.info("Initializing research tools")
        valid_tools = []
        
        # Initialize search tool
        try:
            search_config = self.config.get("tools", {}).get("search", {})
            self.search_tool = SearchAPITool(search_config)
            valid_tools.append(self.search_tool)
        except Exception as e:
            self.logger.error(f"Failed to initialize SearchAPITool: {str(e)}")
        
        # Initialize URL fetcher tool
        try:
            url_config = self.config.get("tools", {}).get("url_fetcher", {})
            self.url_fetcher_tool = URLFetcherTool(url_config)
            valid_tools.append(self.url_fetcher_tool)
        except Exception as e:
            self.logger.error(f"Failed to initialize URLFetcherTool: {str(e)}")
        
        # Initialize PDF reader tool
        try:
            pdf_config = self.config.get("tools", {}).get("pdf_reader", {})
            self.pdf_reader_tool = PDFReaderTool(pdf_config)
            valid_tools.append(self.pdf_reader_tool)
        except Exception as e:
            self.logger.error(f"Failed to initialize PDFReaderTool: {str(e)}")
        
        # Initialize data analysis tool
        try:
            data_config = self.config.get("tools", {}).get("data_analysis", {})
            self.data_analysis_tool = DataAnalysisTool(data_config)
            valid_tools.append(self.data_analysis_tool)
        except Exception as e:
            self.logger.error(f"Failed to initialize DataAnalysisTool: {str(e)}")
        
        # Initialize visualization tool
        try:
            viz_config = self.config.get("tools", {}).get("visualization", {})
            self.visualization_tool = VisualizationTool(viz_config)
            valid_tools.append(self.visualization_tool)
        except Exception as e:
            self.logger.error(f"Failed to initialize VisualizationTool: {str(e)}")
        
        # Initialize database tool
        try:
            db_config = self.config.get("tools", {}).get("database", {})
            db_path = db_config.get("path", "data/research_data.db")
            cache_enabled = db_config.get("cache_enabled", True)
            cache_ttl = db_config.get("cache_ttl", 86400)
            self.database_tool = DatabaseTool(
                db_path=db_path,
                cache_enabled=cache_enabled,
                cache_ttl=cache_ttl
            )
            valid_tools.append(self.database_tool)
        except Exception as e:
            self.logger.error(f"Failed to initialize DatabaseTool: {str(e)}")
        
        # Initialize Python interpreter tool
        try:
            python_tool = PythonInterpreterTool()
            valid_tools.append(python_tool)
        except Exception as e:
            self.logger.error(f"Failed to initialize PythonInterpreterTool: {str(e)}")
            
        # Filter only tools that inherit from smolagents.Tool
        self.tools = [tool for tool in valid_tools if isinstance(tool, Tool)]
        
    def _setup_tools(self):
        """
        Alias for _initialize_tools for compatibility with tests.
        Returns the list of tools for testing.
        """
        self._initialize_tools()
        return self.tools
        
    def _initialize_agent(self):
        """
        Initialize the agent with the necessary tools and configuration.
        
        Returns:
            An instance of the research agent
        """
        self.logger.info("Initializing research agent")
        
        # Отримуємо OpenAI API ключ
        api_key = self.llm_config.get("api_key") or os.environ.get("OPENAI_API_KEY", "test_openai_key")
        
        # Зберігаємо ключ API в конфігурації для подальшого використання
        if "api_key" not in self.llm_config:
            self.llm_config["api_key"] = api_key
        
        # Отримуємо ID моделі
        model_id = self.llm_config.get("model", "gpt-4")
        
        # Якщо проектний ключ OpenAI, додаємо префікс 'openai/'
        if api_key.startswith("sk-proj-"):
            prefixed_model_id = f"openai/{model_id}"
            self.logger.info(f"Using OpenAI model via LiteLLM with project key: {prefixed_model_id}")
            model = LiteLLMModel(
                model_id=prefixed_model_id,  # Додаємо префікс 'openai/' до назви моделі
                api_key=api_key,
                temperature=self.llm_config.get("temperature", 0.7)
            )
        else:
            # Для звичайних ключів також використовуємо LiteLLM для єдності коду
            self.logger.info(f"Using OpenAI model via LiteLLM: {model_id}")
            model = LiteLLMModel(
                model_id=model_id,
                api_key=api_key,
                temperature=self.llm_config.get("temperature", 0.7)
            )
        
        # Create a simple agent with improved prompting
        agent = prompt_utils.create_agent_with_improved_prompt(
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
    
    # Alias for compatibility with tests
    _detect_language = detect_language
    
    def _generate_sample_research_code(self) -> str:
        """
        Генерує приклад коду для дослідження, який буде виконано агентом.
        
        Returns:
            Рядок з Python кодом для дослідження
        """
        code = '''
# Приклад коду для дослідження ринку смартфонів в Україні
import json
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ініціалізація результатів дослідження
research_results = {
    "query": "Дослідження ринку смартфонів в Україні",
    "timestamp": "2023-06-15T10:30:00",
    "success": True
}

# Крок 1: Збір інформації
def gather_information():
    logger.info("Збір інформації з різних джерел...")
    
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
    logger.info("Аналіз зібраних даних...")
    
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
    logger.info("Синтез результатів та створення звіту...")
    
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
    logger.info("Зібрано джерел: %s", len(gathered_info.get('sources', [])))
    
    # Аналіз даних
    if gathered_info["success"]:
        analysis_results = analyze_data(gathered_info)
        logger.info("Знайдено інсайтів: %s", len(analysis_results.get('insights', [])))
        
        # Генерація звіту
        if analysis_results["success"]:
            report = generate_report(analysis_results, gathered_info)
            
            # Оновлення результатів дослідження
            research_results["summary"] = report["summary"]
            research_results["full_report"] = report["full_report"]
            research_results["analysis"] = analysis_results
            research_results["sources"] = gathered_info["sources"]
            research_results["findings"] = analysis_results["insights"]
            
            logger.info("Дослідження завершено успішно.")
    
    # Повернення результатів дослідження
    logger.info("Результати дослідження: %s", json.dumps(research_results, indent=2))
except Exception as e:
    logger.error("Помилка під час дослідження: %s", str(e))
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
                except (ValueError, KeyError, TypeError, AttributeError) as e:
                    self.logger.warning("Error enhancing non-dict report: %s", str(e))
            
            research_results = {
                "query": query,
                "timestamp": timestamp,
                "summary": f"Дослідження за запитом '{query}' завершено.",
                "full_report": report,
                "section_titles": {
                    "summary": get_string("results_summary", self.language),
                    "full_report": get_string("results_full", self.language),
                    "sources": get_string("results_sources", self.language),
                    "findings": get_string("results_findings", self.language),
                    "analysis": get_string("results_analysis", self.language),
                    "limitations": get_string("results_limitations", self.language)
                }
            }
        
        # Store results in context
        # Add compatibility with tests
        if hasattr(self.context, 'add_results'):
            self.context.add_results(research_results)
        else:
            # Fallback for older versions of ContextManager
            if not hasattr(self.context, 'results'):
                self.context.results = []
            self.context.results.append(research_results)
        
        # Return formatted results
        return research_results
    
    except Exception as e:
        self.logger.error("Error formatting results: %s", str(e))
        # Fallback для безпечного повернення
        return {
            "query": query,
        
        # Detect language if not specified
        detected_lang = self.detect_language(query)
        if detected_lang != self.language:
            self.logger.info("Detected language: %s, switching from %s", detected_lang, self.language)
            self.language = detected_lang
        
        # Create a research plan
        # Зауваження: метод create_plan не приймає аргумент language
        plan = self.planner.create_plan(query)
        
        # Create task data for the execution
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get research code (default or custom)
        research_code = self._generate_sample_research_code()
        
        # Create task data - ensure all values are proper types
        task = {
            "id": timestamp.replace(" ", "_").replace(":", ""),
            "query": str(query) if query else "",  # Ensure query is a string
            "language": str(self.language) if self.language else "uk",  # Ensure language is a string
            "timestamp": timestamp,
            "code": research_code,
            "plan": plan
        }
        
        # Debug output for task data
        self.logger.info("Task data types: id=%s, query=%s, language=%s, timestamp=%s, code=%s, plan=%s", 
                       type(task["id"]), type(task["query"]), type(task["language"]), 
                       type(task["timestamp"]), type(task["code"]), type(task["plan"]))
        
        # Execute the research task
        try:
            # Використовуємо реальний агент для виконання завдань
            self.logger.info("Запуск реального дослідницького процесу з використанням smolagents 1.15.0 API")
            
            # Перевіряємо доступні інструменти
            available_tools = [t.__class__.__name__ for t in self.tools]
            self.logger.info(f"Available tools: {', '.join(available_tools)}")
            
            # Підготовка задачі для виконання
            research_instructions = f"""
            You are a research assistant conducting a thorough investigation on: '{query}'.
            Follow this research plan: {plan}
            
            Provide comprehensive results including:
            1. Summary of findings
            2. Key insights 
            3. Sources used (with URLs)
            4. Limitations of the research
            
            Be detailed, factual, and cite your sources.
            """
            
            # Підготовка загальних даних запиту
            total_steps = len(plan["steps"])
            current_step = 1
            
            # Дамо можливість агенту виконати дослідження з використанням доступних інструментів
            # Тут можна було б виконати по-крокове виконання з викликами окремих інструментів,
            # але ми дамо агенту можливість вибрати потрібні інструменти самостійно
            
            # Агент з підтримкою прогресу
            def progress_tracker(percent):
                # Оновлюємо прогрес виконання
                if progress_callback:
                    # Прогрес може бути невідомим, тому ми оцінюємо його на основі кроків
                    # Якщо відсоток прогресу не приходить, обчислюємо його
                    if percent > 0:
                        progress_callback(percent)
                    else:
                        # Поступове відображення прогресу
                        nonlocal current_step
                        progress = (current_step / total_steps) * 100
                        progress_callback(progress)
                        current_step += 1
            
            # Виконуємо запит через агент
            try:
                # Спробуємо з використанням progress_callback
                agent_results = self.agent.run(
                    research_instructions,
                    progress_callback=progress_tracker
                )
            except TypeError:
                # Якщо метод не підтримує progress_callback
                self.logger.info("Run не підтримує progress_callback, виконуємо без нього")
                # Запускаємо без progress_callback
                agent_results = self.agent.run(research_instructions)
                
            # Форматуємо результати дослідження
            formatted_results = self._format_results(agent_results, query, timestamp)
            return formatted_results
            
        except Exception as e:
            self.logger.error("Error processing research query: %s", str(e))
            return {
                "query": query,
                "timestamp": timestamp,
                "error": str(e),
                "summary": get_string("error_occurred", self.language),
                "full_report": get_string("error_details", self.language).format(error=str(e))
            }
