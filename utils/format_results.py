"""
Тимчасовий файл з імплементацією оновленого методу _format_results для класу ResearchAgent.
Цей файл містить метод, який використовує утиліту ResultsEnhancer для покращення результатів дослідження.
"""

def _format_results(self, agent_results, query, timestamp):
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
            from utils.results_enhancer import ResultsEnhancer
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
                        self.logger.info("Enhanced content with more detailed structure")
                    except (ValueError, KeyError, TypeError, AttributeError) as e:
                        self.logger.warning("Error enhancing content: %s", str(e))
                        
                research_results = {
                    "query": query,
                    "timestamp": timestamp,
                    "summary": agent_results.get("summary", ""),
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
                # Зберігаємо базову структуру
                original_report = agent_results.get("full_report", json.dumps(agent_results, indent=2, ensure_ascii=False))
                
                # Спроба покращити звіт
                if has_enhancer and isinstance(agent_results, dict):
                    try:
                        enhanced_results = enhancer.enhance_results(agent_results)
                        self.logger.info("Enhanced basic structure with more details")
                        agent_results = enhanced_results
                    except (ValueError, KeyError, TypeError, AttributeError) as e:
                        self.logger.warning("Error enhancing basic structure: %s", str(e))
                
                research_results = {
                    "query": query,
                    "timestamp": timestamp,
                    "summary": agent_results.get("summary", f"Дослідження за запитом '{query}' завершено."),
                    "full_report": agent_results.get("full_report", original_report),
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
            # Якщо результати у вигляді строки (для версії smolagents 1.15.0)
            response_text = agent_results
            
            # Виділяємо розділи з відповіді
            summary = ""
            sources = []
            findings = []
            
            # Спробуємо виділити резюме (Summary)
            summary_match = re.search(r'(?i)#*\s*Summary\s*[:\n]\s*(.+?)(?=\n\s*#|$)', response_text, re.DOTALL)
            if not summary_match:
                summary_match = re.search(r'(?i)#*\s*Резюме\s*[:\n]\s*(.+?)(?=\n\s*#|$)', response_text, re.DOTALL)
            
            if summary_match and summary_match.group(1) is not None:
                summary = summary_match.group(1).strip()
            else:
                # Якщо не знайдено резюме, використовуємо перший абзац
                first_paragraph = response_text.split('\n\n', 1)[0].strip()
                if len(first_paragraph) > 50:
                    summary = first_paragraph
                else:
                    summary = f"Дослідження за запитом '{query}' завершено."
            
            # Покращення текстових результатів за допомогою enhancer
            if has_enhancer and len(response_text) > 0:
                try:
                    # Створюємо словник для аналізу
                    text_analysis = {
                        "full_report": response_text,
                        "summary": summary
                    }
                    enhanced_text = enhancer.enhance_results(text_analysis)
                    if isinstance(enhanced_text, dict) and "full_report" in enhanced_text:
                        response_text = enhanced_text["full_report"]
                        if "summary" in enhanced_text and enhanced_text["summary"]:
                            summary = enhanced_text["summary"]
                        self.logger.info("Enhanced text-based results")
                except (ValueError, KeyError, TypeError, AttributeError) as e:
                    self.logger.warning("Error enhancing text results: %s", str(e))
            
            # Виділяємо джерела
            sources_match = re.search(r'(?i)#*\s*Sources\s*[:\n]\s*(.+?)(?=\n\s*#|$)', response_text, re.DOTALL)
            if sources_match and sources_match.group(1) is not None:
                sources = sources_match.group(1).strip().split('\n')
            
            # Виділяємо висновки
            findings_match = re.search(r'(?i)#*\s*Findings\s*[:\n]\s*(.+?)(?=\n\s*#|$)', response_text, re.DOTALL)
            if findings_match and findings_match.group(1) is not None:
                findings = findings_match.group(1).strip().split('\n')
            
            # Об'єднання результатів
            research_results = {
                "query": query,
                "timestamp": timestamp,
                "summary": summary,
                "full_report": response_text,
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
        else:
            # Якщо тип результатів невідомий, застосовуємо дружній формат
            self.logger.warning("Unknown result type: %s, converting to string", type(agent_results))
            report = str(agent_results)
            
            # Спроба покращити текстовий звіт
            if has_enhancer and len(report) > 100:
                try:
                    analysis_content = {"full_report": report}
                    enhanced_content = enhancer.enhance_results(analysis_content)
                    if "full_report" in enhanced_content:
                        report = enhanced_content["full_report"]
                    self.logger.info("Enhanced non-dict report")
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
            "timestamp": timestamp,
            "summary": f"Дослідження за запитом '{query}' завершено з помилкою: {str(e)}",
            "full_report": str(agent_results),
            "error": str(e),
            "section_titles": {
                "summary": get_string("results_summary", self.language),
                "full_report": get_string("results_full", self.language),
                "error": "Помилка"
            }
        }
