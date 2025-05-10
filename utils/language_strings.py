"""
Language strings for the Research Agent.
Contains translations for all user-facing text.
"""

# Dictionary of all interface strings
STRINGS = {
    "en": {
        # General messages
        "initializing": "Initializing research agent...",
        "ready": "Research agent ready.",
        "processing_query": "Processing research query: '{query}'",
        "research_complete": "Research complete.",
        "error_occurred": "An error occurred: {error}",
        
        # Progress messages
        "planning_research": "Planning research...",
        "gathering_information": "Gathering information from various sources...",
        "analyzing_data": "Analyzing collected information...",
        "generating_report": "Synthesizing findings and generating report...",
        "progress_update": "Progress: {progress}%",
        
        # Result messages
        "results_summary": "Research Summary",
        "results_full": "Full Research Report",
        "results_sources": "Sources",
        "results_findings": "Key Findings",
        "results_analysis": "Analysis",
        "results_limitations": "Limitations and Uncertainties",
        "no_results": "No results were found for this query.",
        "update_smolagents": "The agent requires tools compatible with smolagents 1.15.0.",
        "update_smolagents_detailed": "The Research Agent is using smolagents 1.15.0, but the current tools are not compatible with this version. The tools need to be updated to inherit from the smolagents.Tool class. Please update the tools implementation to make them compatible with the new API.",
        
        # System prompts
        "system_prompt": """
You are a research assistant tasked with gathering, analyzing, and synthesizing information.
Follow these guidelines:
1. Break down complex research tasks into logical steps
2. Gather information from credible sources
3. Analyze data objectively
4. Synthesize findings into cohesive reports
5. Provide proper citations for all information
6. Highlight limitations and uncertainties in your findings
""",
        
        # Research instructions
        "research_instructions": """
1. Break down complex research tasks into logical steps
2. Gather information from credible sources
3. Analyze data objectively
4. Synthesize findings into cohesive reports
5. Provide proper citations for all information
6. Highlight limitations and uncertainties in your findings
"""
    },
    
    "uk": {
        # General messages
        "initializing": "Ініціалізація дослідницького агента...",
        "ready": "Дослідницький агент готовий до роботи.",
        "processing_query": "Обробка дослідницького запиту: '{query}'",
        "research_complete": "Дослідження завершено.",
        "error_occurred": "Сталася помилка: {error}",
        
        # Progress messages
        "planning_research": "Планування дослідження...",
        "gathering_information": "Збір інформації з різних джерел...",
        "analyzing_data": "Аналіз зібраної інформації...",
        "generating_report": "Синтез результатів та створення звіту...",
        "progress_update": "Прогрес: {progress}%",
        
        # Result messages
        "results_summary": "Резюме дослідження",
        "results_full": "Повний звіт дослідження",
        "results_sources": "Джерела",
        "results_findings": "Ключові висновки",
        "results_analysis": "Аналіз",
        "results_limitations": "Обмеження та невизначеності",
        "no_results": "За цим запитом не знайдено результатів.",
        "update_smolagents": "Агент потребує інструменти, сумісні з smolagents 1.15.0.",
        "update_smolagents_detailed": "Дослідницький агент використовує smolagents 1.15.0, але поточні інструменти не сумісні з цією версією. Інструменти повинні бути оновлені, щоб успадковуватися від класу smolagents.Tool. Будь ласка, оновіть реалізацію інструментів, щоб вони були сумісні з новим API.",
        
        # System prompts
        "system_prompt": """
Ви — дослідницький асистент, завданням якого є збір, аналіз та синтез інформації.
Дотримуйтесь цих рекомендацій:
1. Розбивайте складні дослідницькі завдання на логічні кроки
2. Збирайте інформацію з надійних джерел
3. Аналізуйте дані об'єктивно
4. Синтезуйте результати у зв'язні звіти
5. Надавайте належні посилання на всю інформацію
6. Виділяйте обмеження та невизначеності у ваших висновках
""",
        
        # Research instructions
        "research_instructions": """
1. Розбивайте складні дослідницькі завдання на логічні кроки
2. Збирайте інформацію з надійних джерел
3. Аналізуйте дані об'єктивно
4. Синтезуйте результати у зв'язні звіти
5. Надавайте належні посилання на всю інформацію
6. Виділяйте обмеження та невизначеності у ваших висновках
"""
    }
}

def get_string(key: str, language: str = "en", **kwargs) -> str:
    """
    Get a string in the specified language.
    
    Args:
        key: The string key to retrieve
        language: The language code (default: "en")
        **kwargs: Format parameters for the string
        
    Returns:
        The translated string
    """
    # Default to English if language not supported
    if language not in STRINGS:
        language = "en"
        
    # Get the string or return a placeholder if key not found
    string = STRINGS[language].get(key, f"[Missing string: {key}]")
    
    # Format the string with any provided parameters
    if kwargs:
        try:
            string = string.format(**kwargs)
        except KeyError as e:
            return f"[Format error for {key}: {e}]"
        
    return string
