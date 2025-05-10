"""
Research Agent - Core agent implementation based on SmolaGents framework.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional, Union

from smolagents import CodeAgent
from rich.console import Console

from core.research_planner import ResearchPlanner
from core.context_manager import ContextManager
from tools.search_tool import SearchAPITool
from tools.url_fetcher_tool import URLFetcherTool
from tools.pdf_reader_tool import PDFReaderTool
from tools.data_analysis_tool import DataAnalysisTool
from tools.visualization_tool import VisualizationTool
from tools.database_tool import DatabaseTool


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
        
        # Initialize context manager for maintaining state
        self.context = ContextManager()
        
        # Initialize research planner
        self.planner = ResearchPlanner(config.get("planning", {}))
        
        # Load LLM API keys from environment
        self._setup_llm_credentials()
        
        # Initialize the agent with tools
        self._initialize_agent()

    def _setup_llm_credentials(self):
        """Set up LLM credentials from environment variables."""
        self.llm_config = {
            "provider": self.config.get("llm", {}).get("provider", "openai"),
            "model": self.config.get("llm", {}).get("model", "gpt-4"),
            "api_key": os.environ.get(f"{self.config.get('llm', {}).get('provider', 'openai').upper()}_API_KEY"),
            "temperature": self.config.get("llm", {}).get("temperature", 0.7),
        }
        
        if not self.llm_config["api_key"]:
            self.logger.warning(f"No API key found for {self.llm_config['provider']}")
            
    def _initialize_agent(self):
        """Initialize the SmolaGents agent with appropriate tools."""
        # Initialize tools based on configuration
        self.tools = self._setup_tools()
        
        # System prompt for the agent
        system_prompt = self.config.get("prompts", {}).get("system_prompt", """
        You are a research assistant tasked with gathering, analyzing, and synthesizing information.
        Follow these guidelines:
        1. Break down complex research tasks into logical steps
        2. Gather information from credible sources
        3. Analyze data objectively
        4. Synthesize findings into cohesive reports
        5. Provide proper citations for all information
        6. Highlight limitations and uncertainties in your findings
        """)
        
        # Initialize the CodeAgent
        self.agent = CodeAgent(
            tools=self.tools,
            system_prompt=system_prompt,
            model_name=self.llm_config["model"],
            provider=self.llm_config["provider"],
            temperature=self.llm_config["temperature"],
            api_key=self.llm_config["api_key"],
        )
        
    def _setup_tools(self) -> List[Any]:
        """
        Set up and configure the tools used by the agent.
        
        Returns:
            List of initialized tools
        """
        tools = []
        
        # Search tool
        search_config = self.config.get("tools", {}).get("search", {})
        if search_config.get("enabled", True):
            tools.append(SearchAPITool(
                api_key=os.environ.get(search_config.get("api_key_env", "SERPAPI_API_KEY")),
                engine=search_config.get("engine", "google"),
                max_results=search_config.get("max_results", 10)
            ))
            
        # URL Fetcher tool
        url_config = self.config.get("tools", {}).get("url_fetcher", {})
        if url_config.get("enabled", True):
            tools.append(URLFetcherTool(
                timeout=url_config.get("timeout", 30),
                max_size=url_config.get("max_size", 1024 * 1024)  # 1MB default
            ))
            
        # PDF Reader tool
        pdf_config = self.config.get("tools", {}).get("pdf_reader", {})
        if pdf_config.get("enabled", True):
            tools.append(PDFReaderTool(
                max_pages=pdf_config.get("max_pages", 50)
            ))
            
        # Data Analysis tool
        analysis_config = self.config.get("tools", {}).get("data_analysis", {})
        if analysis_config.get("enabled", True):
            tools.append(DataAnalysisTool(
                analysis_packages=analysis_config.get("packages", ["pandas", "numpy"])
            ))
            
        # Visualization tool
        viz_config = self.config.get("tools", {}).get("visualization", {})
        if viz_config.get("enabled", True):
            tools.append(VisualizationTool(
                max_width=viz_config.get("max_width", 80),
                use_unicode=viz_config.get("use_unicode", True)
            ))
            
        # Database tool
        db_config = self.config.get("tools", {}).get("database", {})
        if db_config.get("enabled", True):
            tools.append(DatabaseTool(
                db_path=db_config.get("path", "data/research_data.db"),
                cache_enabled=db_config.get("cache_enabled", True),
                cache_ttl=db_config.get("cache_ttl", 86400)  # 24 hours default
            ))
            
        return tools
        
    def process_query(self, query: str, progress_callback: Optional[Callable[[int], None]] = None) -> Dict[str, Any]:
        """
        Process a research query and return results.
        
        Args:
            query: The research query to process
            progress_callback: Optional callback function to report progress (0-100)
            
        Returns:
            Dictionary containing research results
        """
        try:
            self.logger.info(f"Processing research query: {query}")
            
            # Report initial progress
            if progress_callback:
                progress_callback(5)
                
            # Step 1: Create a research plan
            research_plan = self.planner.create_plan(query)
            self.logger.info(f"Research plan created with {len(research_plan['steps'])} steps")
            self.context.add_to_context("research_plan", research_plan)
            
            if progress_callback:
                progress_callback(15)
                
            # Step 2: Execute the research plan using the agent
            self.logger.info("Executing research plan")
            execution_code = self._generate_execution_code(research_plan, query)
            
            # Record time for citation purposes
            research_timestamp = datetime.now().isoformat()
            self.context.add_to_context("timestamp", research_timestamp)
            
            # Execute the code agent with progress updates
            results = {}
            if progress_callback:
                # Report progress from 20% to 90% during execution
                def code_progress_callback(step_index, total_steps):
                    progress = 20 + int(70 * (step_index / total_steps))
                    progress_callback(progress)
                
                results = self.agent.run(execution_code, 
                                         {"context": self.context.get_context()},
                                         progress_callback=code_progress_callback)
            else:
                results = self.agent.run(execution_code, 
                                         {"context": self.context.get_context()})
                
            # Step 3: Format and return results
            self.logger.info("Research completed, formatting results")
            
            # Format the output
            formatted_results = self._format_results(results, query, research_timestamp)
            
            if progress_callback:
                progress_callback(100)
                
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise RuntimeError(f"Research failed: {str(e)}")
            
    def _generate_execution_code(self, research_plan: Dict[str, Any], query: str) -> str:
        """
        Generate the execution code for the agent based on the research plan.
        
        Args:
            research_plan: Dictionary containing the research plan
            query: Original research query
            
        Returns:
            String containing Python code for execution
        """
        # Basic code template that the agent will expand and modify
        code = f'''
# Research Query: {query}
# Generated Research Plan
"""
{json.dumps(research_plan, indent=2)}
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

# Research Results Dictionary
research_results = {{
    "query": "{query}",
    "timestamp": context.get("timestamp"),
    "sources": [],
    "findings": [],
    "analysis": {{}},
    "summary": "",
    "full_report": ""
}}

# Step 1: Information Gathering
def gather_information():
    print("Gathering information from various sources...")
    
    # Implement search strategy based on the research plan
    # Use the available tools to gather information
    
    return {{
        "success": True,
        "sources_collected": 0,
        "data_points": 0
    }}

# Step 2: Data Analysis
def analyze_data(gathered_info):
    print("Analyzing collected information...")
    
    # Implement analysis logic based on the research plan
    # Use data analysis tools to extract insights
    
    return {{
        "success": True,
        "insights": [],
        "trends": [],
        "statistics": {{}}
    }}

# Step 3: Synthesis and Report Generation
def generate_report(analysis_results):
    print("Synthesizing findings and generating report...")
    
    # Create a comprehensive report based on the research plan
    # Include all necessary citations and references
    
    return {{
        "summary": "",
        "full_report": "",
        "bibliography": []
    }}

# Main Execution
gathered_info = gather_information()
if gathered_info["success"]:
    analysis_results = analyze_data(gathered_info)
    if analysis_results["success"]:
        report = generate_report(analysis_results)
        
        # Update the research results
        research_results["summary"] = report["summary"]
        research_results["full_report"] = report["full_report"]
        research_results["analysis"] = analysis_results
        
        # Return the complete research results
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
        # Extract the research_results from agent output
        research_results = agent_results.get("result", {})
        
        if not research_results:
            research_results = {
                "query": query,
                "timestamp": timestamp,
                "summary": "Research could not be completed.",
                "full_report": "The research agent encountered an error during execution."
            }
        
        # Ensure all expected fields are present
        if "query" not in research_results:
            research_results["query"] = query
            
        if "timestamp" not in research_results:
            research_results["timestamp"] = timestamp
            
        if "summary" not in research_results or not research_results["summary"]:
            research_results["summary"] = "No summary was generated."
            
        if "full_report" not in research_results or not research_results["full_report"]:
            research_results["full_report"] = "No detailed report was generated."
        
        return research_results
