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
from utils.results_enhancer import ResultsEnhancer


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
        try:
            self.logger.info("\n")
            self.logger.info("Result type: %s", type(agent_results))
            self.logger.info("")
            
            # Спроба імпортувати ResultsEnhancer для покращення результатів
            try:
                # from utils.results_enhancer import ResultsEnhancer # Already imported at the top
                enhancer = ResultsEnhancer()
                has_enhancer = True
                self.logger.info("Using ResultsEnhancer to improve results quality")
            except ImportError:
                has_enhancer = False
                self.logger.warning("ResultsEnhancer not available, using basic formatting")
            
            if isinstance(agent_results, dict):
                self.logger.info("Result keys: %s", agent_results.keys())
                self.logger.info("")
                
                # For legacy reasons/compatibility with older models
                if 'response' in agent_results:
                    content = agent_results["response"]
                    
                    # Використовуємо enhancer для покращення вмісту, якщо він доступний
                    if has_enhancer and isinstance(content, str) and len(content) > 0:
                        try:
                            # Створюємо структуру для аналізу
                            analysis_content = {"full_report": content}
                            enhanced_content = enhancer.enhance_results(analysis_content)
                            content = enhanced_content.get("full_report", content)
                            self.logger.info("Enhanced raw response content")
                        except (ValueError, KeyError, TypeError, AttributeError) as e:
                            self.logger.warning("Error enhancing response content: %s", str(e))
                    
                    research_results = {
                        "query": query,
                        "timestamp": timestamp,
                        "summary": get_string("results_completed", self.language).format(query=query),
                        "full_report": content,
                        "section_titles": {
                            "summary": get_string("results_summary", self.language),
                            "full_report": get_string("results_full", self.language),
                            "sources": get_string("results_sources", self.language),
                            "findings": get_string("results_findings", self.language),
                            "analysis": get_string("results_analysis", self.language),
                            "limitations": get_string("results_limitations", self.language)
                        }
                    }
                # For smolagent version ~1.15.0
                elif 'Summary of Findings' in agent_results:
                    # Already a results dict - покращуємо результати, якщо доступно
                    if has_enhancer:
                        try:
                            enhanced_results = enhancer.enhance_results(agent_results)
                            self.logger.info("Enhanced results with more detailed content")
                            full_report = enhanced_results
                        except (ValueError, KeyError, TypeError, AttributeError) as e:
                            self.logger.warning("Error enhancing structured results: %s", str(e))
                            full_report = agent_results
                    else:
                        full_report = agent_results
                        
                    research_results = {
                        "query": query,
                        "timestamp": timestamp,
                        "summary": get_string("results_completed", self.language).format(query=query),
                        "full_report": full_report,
                        "section_titles": {
                            "summary": get_string("results_summary", self.language),
                            "full_report": get_string("results_full", self.language),
                            "sources": get_string("results_sources", self.language),
                            "findings": get_string("results_findings", self.language),
                            "analysis": get_string("results_analysis", self.language),
                            "limitations": get_string("results_limitations", self.language)
                        }
                    }
                # For smolagent version ~1.15.0 but different format
                elif 'content' in agent_results:
                    content = agent_results["content"]
                    # Спроба покращити контент, якщо він є словником
                    if has_enhancer and isinstance(content, dict):
                        try:
                            enhanced_content = enhancer.enhance_results(content)
                            content = enhanced_content
                            self.logger.info("Enhanced dictionary content")
                        except (ValueError, KeyError, TypeError, AttributeError) as e:
                            self.logger.warning("Error enhancing dictionary content: %s", str(e))
                    elif has_enhancer and isinstance(content, str) and len(content) > 0:
                        try:
                            # Створюємо структуру для аналізу, якщо це просто рядок
                            analysis_content = {"full_report": content}
                            enhanced_content = enhancer.enhance_results(analysis_content)
                            content = enhanced_content.get("full_report", content)
                            self.logger.info("Enhanced string content from 'content' key")
                        except (ValueError, KeyError, TypeError, AttributeError) as e:
                            self.logger.warning("Error enhancing string content from 'content' key: %s", str(e))
                    
                    research_results = {
                        "query": query,
                        "timestamp": timestamp,
                        "summary": get_string("results_completed", self.language).format(query=query),
                        "full_report": content,
                        "section_titles": {
                            "summary": get_string("results_summary", self.language),
                            "full_report": get_string("results_full", self.language),
                            "sources": get_string("results_sources", self.language),
                            "findings": get_string("results_findings", self.language),
                            "analysis": get_string("results_analysis", self.language),
                            "limitations": get_string("results_limitations", self.language)
                        }
                    }
                else:
                    self.logger.warning("Unrecognized dictionary structure for agent_results. Using raw dictionary.")
                    # Якщо структура невідома, намагаємося покращити її, якщо це можливо
                    if has_enhancer:
                        try:
                            enhanced_results = enhancer.enhance_results(agent_results)
                            self.logger.info("Enhanced unrecognized dictionary structure")
                            full_report_content = enhanced_results
                        except (ValueError, KeyError, TypeError, AttributeError) as e:
                            self.logger.warning("Error enhancing unrecognized dictionary structure: %s", str(e))
                            full_report_content = agent_results
                    else:
                        full_report_content = agent_results
                        
                    research_results = {
                        "query": query,
                        "timestamp": timestamp,
                        "summary": get_string("results_completed", self.language).format(query=query),
                        "full_report": full_report_content,
                        "section_titles": {
                            "summary": get_string("results_summary", self.language),
                            "full_report": get_string("results_full", self.language),
                            "sources": get_string("results_sources", self.language),
                            "findings": get_string("results_findings", self.language),
                            "analysis": get_string("results_analysis", self.language),
                            "limitations": get_string("results_limitations", self.language)
                        }
                    }
            elif isinstance(agent_results, str):
                self.logger.info("Agent results is a string. Length: %d", len(agent_results))
                content = agent_results
                # Покращуємо рядок, якщо enhancer доступний
                if has_enhancer and len(content) > 0:
                    try:
                        analysis_content = {"full_report": content}
                        enhanced_content = enhancer.enhance_results(analysis_content)
                        content = enhanced_content.get("full_report", content)
                        self.logger.info("Enhanced string results")
                    except (ValueError, KeyError, TypeError, AttributeError) as e:
                        self.logger.warning("Error enhancing string results: %s", str(e))
                
                research_results = {
                    "query": query,
                    "timestamp": timestamp,
                    "summary": get_string("results_completed", self.language).format(query=query),
                    "full_report": content,
                    "section_titles": {
                        "summary": get_string("results_summary", self.language),
                        "full_report": get_string("results_full", self.language),
                        "sources": get_string("results_sources", self.language),
                        "findings": get_string("results_findings", self.language),
                        "analysis": get_string("results_analysis", self.language),
                        "limitations": get_string("results_limitations", self.language)
                    }
                }
            else:
                self.logger.error("Unsupported agent_results type: %s. Returning raw results.", type(agent_results))
                research_results = {
                    "query": query,
                    "timestamp": timestamp,
                    "summary": get_string("results_error", self.language),
                    "full_report": str(agent_results),  # Перетворюємо на рядок для безпеки
                    "section_titles": {
                        "summary": get_string("results_summary", self.language),
                        "full_report": get_string("results_full", self.language),
                        "sources": get_string("results_sources", self.language),
                        "findings": get_string("results_findings", self.language),
                        "analysis": get_string("results_analysis", self.language),
                        "limitations": get_string("results_limitations", self.language)
                    }
                }
        except Exception as e:
            self.logger.error("Critical error in _format_results: %s", str(e), exc_info=True)
            research_results = {
                "query": query,
                "timestamp": timestamp,
                "summary": get_string("results_critical_error", self.language),
                "full_report": f"Error: {str(e)}",
                "section_titles": {
                    "summary": get_string("results_summary", self.language),
                    "full_report": get_string("results_full", self.language),
                    "sources": get_string("results_sources", self.language),
                    "findings": get_string("results_findings", self.language),
                    "analysis": get_string("results_analysis", self.language),
                    "limitations": get_string("results_limitations", self.language)
                }
            }
            
        self.logger.debug("Formatted results: %s", json.dumps(research_results, indent=2, ensure_ascii=False))
        return research_results

    def _save_results(self, results: Dict[str, Any]):
        """
        Save the research results to the database.
        
        Args:
            results: Research results to save
        """
        try:
            # Store results in context
            # Add compatibility with tests
            if hasattr(self.context, 'add_results'):
                self.context.add_results(results)
            else:
                # Fallback for older versions of ContextManager
                if not hasattr(self.context, 'results'):
                    self.context.results = []
                self.context.results.append(results)
            
            # Save results to database
            self.database_tool.save_results(results)
        except Exception as e:
            self.logger.error("Error saving results: %s", str(e))
            
    def process_query(self, query: str, progress_callback: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
        """
        Process a research query and return the results.
        
        Args:
            query: Research query to process
            progress_callback: Optional callback function to report progress
            
        Returns:
            Research results dictionary
        """
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
        
        # Create task data - ensure all values are proper types
        task = {
            "id": timestamp.replace(" ", "_").replace(":", ""),
            "query": str(query) if query else "",  # Ensure query is a string
            "language": str(self.language) if self.language else "uk",  # Ensure language is a string
            "timestamp": timestamp,
            "plan": plan
        }
        
        # Debug output for task data
        self.logger.info("Task data types: id=%s, query=%s, language=%s, timestamp=%s, plan=%s", 
                       type(task["id"]), type(task["query"]), type(task["language"]), 
                       type(task["timestamp"]), type(task["plan"]))
        
        # Execute the research task
        try:
            # Використовуємо реальний агент для виконання завдань
            self.logger.info("Запуск реального дослідницького процесу з використанням smolagents 1.15.0 API")
            
            # Перевіряємо доступні інструменти
            available_tools = [t.__class__.__name__ for t in self.tools]
            self.logger.info(f"Available tools: {', '.join(available_tools)}")
            
            # Підготовка задачі для виконання залежно від встановленої мови
            if self.language == "uk":
                research_instructions = f"""
                Ви - дослідницький асистент, який проводить ґрунтовне дослідження на тему: '{query}'.
                Дотримуйтесь цього плану дослідження: {plan}
                
                Надайте вичерпні результати, які включають:
                1. Узагальнення отриманої інформації
                2. Ключові висновки
                3. Використані джерела (з URL-посиланнями)
                4. Обмеження дослідження
                
                Будьте детальними, об'єктивними та цитуйте ваші джерела.
                Обов'язково надайте фінальний результат українською мовою.
                """
            else:
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
            
            # Зберігаємо результати у базу даних
            self._save_results(formatted_results)
            
            # Зберігаємо результати у файл
            if "full_report" in formatted_results:
                # Створюємо унікальне ім'я файлу на основі запиту та часової мітки
                safe_query = re.sub(r'[^\w\s]', '', query[:30]).strip().replace(' ', '_').lower()
                filename = f"res_{safe_query}_{timestamp.replace(' ', '_').replace(':', '')}.txt"
                
                try:
                    # Підготовка вмісту для збереження у файл
                    if "full_report" in formatted_results:
                        # Перевіряємо тип даних full_report та відповідно його обробляємо
                        if isinstance(formatted_results["full_report"], str):
                            report_content = formatted_results["full_report"]
                        elif isinstance(formatted_results["full_report"], dict):
                            try:
                                # Якщо це словник, конвертуємо його в JSON рядок
                                report_content = json.dumps(formatted_results["full_report"], ensure_ascii=False, indent=2)
                            except (TypeError, ValueError) as e:
                                self.logger.warning(f"Помилка при конвертації словника у JSON: {str(e)}")
                                # Якщо не вдалося конвертувати в JSON, використовуємо стрінгове представлення
                                report_content = str(formatted_results["full_report"])
                        elif isinstance(formatted_results["full_report"], list):
                            try:
                                # Якщо це список, спробуємо конвертувати в JSON
                                report_content = json.dumps(formatted_results["full_report"], ensure_ascii=False, indent=2)
                            except (TypeError, ValueError) as e:
                                self.logger.warning(f"Помилка при конвертації списку у JSON: {str(e)}")
                                # Якщо не вдалося, з'єднуємо елементи списку в текст
                                report_content = "\n".join([str(item) for item in formatted_results["full_report"]])
                        else:
                            # Для будь-яких інших типів даних, конвертуємо в рядок
                            report_content = str(formatted_results["full_report"])
                    else:
                        # Якщо full_report відсутній, спробуємо використати інші ключі, які можуть містити вміст звіту
                        if "summary" in formatted_results and formatted_results["summary"]:
                            report_parts = []
                            if formatted_results["summary"]:
                                report_parts.append(f"Summary: {formatted_results['summary']}\n")
                            if "key_insights" in formatted_results and formatted_results["key_insights"]:
                                report_parts.append(f"Key Insights: {formatted_results['key_insights']}\n")
                            if "sources" in formatted_results and formatted_results["sources"]:
                                report_parts.append(f"Sources: {formatted_results['sources']}\n")
                            report_content = "\n".join(report_parts)
                        else:
                            # Якщо немає основних ключів для структурованого звіту, використовуємо весь результат
                            try:
                                report_content = json.dumps(formatted_results, ensure_ascii=False, indent=2)
                            except (TypeError, ValueError) as e:
                                self.logger.warning(f"Помилка при конвертації результатів у JSON: {str(e)}")
                                # Якщо не вдалося конвертувати в JSON, використовуємо стрінгове представлення
                                report_content = str(formatted_results)
                    
                    # Переконуємося, що report_content є рядком перед записом у файл
                    if not isinstance(report_content, str):
                        report_content = str(report_content)
                        
                    # Записуємо вміст у файл
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(report_content)
                    
                    self.logger.info(f"Results saved to file: {filename}")
                    formatted_results["report_file"] = filename
                except Exception as e:
                    self.logger.error(f"Error saving results to file: {str(e)}")
            
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
