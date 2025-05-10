"""
Research Planner for the Research Agent.
Responsible for creating structured research plans based on user queries.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class ResearchPlanner:
    """
    Creates and manages research plans based on user queries.
    Breaks down complex research tasks into structured steps.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the research planner.
        
        Args:
            config: Configuration for the planner
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Default configuration values
        self.max_steps = self.config.get("max_steps", 10)
        self.step_timeout = self.config.get("step_timeout", 300)  # 5 minutes
        self.research_depth = self.config.get("research_depth", "medium")
        
    def create_plan(self, query: str) -> Dict[str, Any]:
        """
        Create a research plan based on the user query.
        
        Args:
            query: The research query from the user
            
        Returns:
            Dictionary containing the structured research plan
        """
        self.logger.info(f"Creating research plan for query: {query}")
        
        # Set depth factors based on research depth
        depth_factors = self._get_depth_factors()
        
        # Parse the query to extract key aspects
        query_aspects = self._parse_query(query)
        
        # Generate research steps
        steps = self._generate_research_steps(query, query_aspects, depth_factors)
        
        # Create the research plan
        plan = {
            "query": query,
            "aspects": query_aspects,
            "created_at": datetime.now().isoformat(),
            "depth": self.research_depth,
            "steps": steps,
            "estimated_completion_time": self._estimate_completion_time(steps)
        }
        
        self.logger.info(f"Research plan created with {len(steps)} steps")
        return plan
        
    def _parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse the query to extract key aspects for research.
        
        Args:
            query: The research query
            
        Returns:
            Dictionary with extracted query aspects
        """
        # In a real implementation, this would use LLM to extract key aspects
        # For now, we'll use a simple dictionary structure
        
        aspects = {
            "main_topic": query,
            "subtopics": [],
            "time_frame": None,
            "perspective": "neutral",
            "required_depth": self.research_depth
        }
        
        # Simple heuristics to extract time frame (could be replaced with LLM)
        time_indicators = [
            "last year", "past year", "last month", "past month",
            "recent", "latest", "current", "past decade", "historical",
            "future", "upcoming", "next year", "next decade"
        ]
        
        for indicator in time_indicators:
            if indicator in query.lower():
                aspects["time_frame"] = indicator
                break
                
        return aspects
        
    def _get_depth_factors(self) -> Dict[str, Any]:
        """
        Get depth factors based on the configured research depth.
        
        Returns:
            Dictionary with depth-related parameters
        """
        depth_map = {
            "shallow": {
                "sources_per_topic": 3,
                "analysis_detail": "basic",
                "citation_requirement": "minimal"
            },
            "medium": {
                "sources_per_topic": 5,
                "analysis_detail": "moderate",
                "citation_requirement": "standard"
            },
            "deep": {
                "sources_per_topic": 10,
                "analysis_detail": "comprehensive",
                "citation_requirement": "extensive"
            }
        }
        
        return depth_map.get(self.research_depth, depth_map["medium"])
        
    def _generate_research_steps(self, query: str, query_aspects: Dict[str, Any], 
                                 depth_factors: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate research steps based on the query and depth factors.
        
        Args:
            query: The research query
            query_aspects: Extracted aspects of the query
            depth_factors: Depth-related parameters
            
        Returns:
            List of research steps
        """
        # In a complete implementation, this would use LLM to generate steps
        # For now, we'll create a standard research workflow
        
        steps = [
            {
                "id": 1,
                "name": "Initial Query Analysis",
                "description": "Analyze the research query and break it down into components",
                "tools": ["data_analysis"],
                "expected_output": "Structured query components",
                "estimated_time": 30  # seconds
            },
            {
                "id": 2,
                "name": "Information Gathering",
                "description": f"Search for relevant information across multiple sources, collecting {depth_factors['sources_per_topic']} sources per topic",
                "tools": ["search", "url_fetcher", "pdf_reader"],
                "expected_output": "Collection of relevant information",
                "estimated_time": 120  # seconds
            },
            {
                "id": 3,
                "name": "Data Organization",
                "description": "Organize and structure collected information",
                "tools": ["data_analysis", "database"],
                "expected_output": "Structured dataset of research information",
                "estimated_time": 60  # seconds
            },
            {
                "id": 4,
                "name": "Analysis and Synthesis",
                "description": f"Analyze information with {depth_factors['analysis_detail']} detail and synthesize findings",
                "tools": ["data_analysis", "visualization"],
                "expected_output": "Key findings and insights",
                "estimated_time": 90  # seconds
            },
            {
                "id": 5,
                "name": "Report Generation",
                "description": f"Generate comprehensive report with {depth_factors['citation_requirement']} citations",
                "tools": ["database", "visualization"],
                "expected_output": "Complete research report",
                "estimated_time": 60  # seconds
            }
        ]
        
        # If query requires deeper research, add more specialized steps
        if self.research_depth == "deep":
            steps.insert(3, {
                "id": 3.5,
                "name": "Secondary Source Verification",
                "description": "Verify information from secondary sources for accuracy",
                "tools": ["search", "url_fetcher", "database"],
                "expected_output": "Verified information with confidence scores",
                "estimated_time": 90  # seconds
            })
            
        return steps
        
    def _estimate_completion_time(self, steps: List[Dict[str, Any]]) -> int:
        """
        Estimate total completion time for the research plan.
        
        Args:
            steps: List of research steps
            
        Returns:
            Estimated completion time in seconds
        """
        total_time = sum(step.get("estimated_time", 60) for step in steps)
        
        # Add overhead factor based on research depth
        overhead_factors = {"shallow": 1.2, "medium": 1.5, "deep": 2.0}
        overhead = overhead_factors.get(self.research_depth, 1.5)
        
        return int(total_time * overhead)
