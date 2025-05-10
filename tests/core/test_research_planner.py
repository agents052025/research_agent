"""
Unit tests for the ResearchPlanner class.
"""

import pytest
from unittest.mock import patch
from datetime import datetime

from core.research_planner import ResearchPlanner


class TestResearchPlanner:
    """Test cases for the ResearchPlanner class."""

    def test_initialization(self):
        """Test that the planner initializes with default values."""
        planner = ResearchPlanner()
        
        # Check default values
        assert planner.max_steps == 10
        assert planner.step_timeout == 300
        assert planner.research_depth == "medium"
        
        # Test with custom config
        custom_config = {
            "max_steps": 5,
            "step_timeout": 120,
            "research_depth": "deep"
        }
        
        planner = ResearchPlanner(custom_config)
        
        # Check custom values
        assert planner.max_steps == 5
        assert planner.step_timeout == 120
        assert planner.research_depth == "deep"
    
    def test_create_plan(self):
        """Test creating a research plan from a query."""
        planner = ResearchPlanner()
        
        # Create a plan for an English query
        query = "Analyze recent trends in artificial intelligence"
        plan = planner.create_plan(query)
        
        # Check that the plan has the expected structure
        assert "query" in plan
        assert "aspects" in plan
        assert "created_at" in plan
        assert "depth" in plan
        assert "steps" in plan
        assert "estimated_completion_time" in plan
        
        # Check that the query was stored
        assert plan["query"] == query
        
        # Check that the plan has steps
        assert len(plan["steps"]) > 0
        
        # Check that each step has required fields
        for step in plan["steps"]:
            assert "id" in step
            assert "name" in step
            assert "description" in step
            assert "tools" in step
            assert "expected_output" in step
            assert "estimated_time" in step
    
    def test_create_plan_ukrainian(self):
        """Test creating a research plan from a Ukrainian query."""
        planner = ResearchPlanner()
        
        # Create a plan for a Ukrainian query
        query = "Проаналізуй останні тенденції у штучному інтелекті"
        plan = planner.create_plan(query)
        
        # Check that the plan has the expected structure
        assert "query" in plan
        assert "aspects" in plan
        assert "created_at" in plan
        assert "depth" in plan
        assert "steps" in plan
        assert "estimated_completion_time" in plan
        
        # Check that the query was stored
        assert plan["query"] == query
        
        # Check that the plan has steps
        assert len(plan["steps"]) > 0
    
    def test_parse_query(self):
        """Test parsing a query to extract key aspects."""
        planner = ResearchPlanner()
        
        # Test with a query containing time frame
        query = "Analyze recent trends in AI over the past year"
        aspects = planner._parse_query(query)
        
        assert aspects["main_topic"] == query
        assert aspects["time_frame"] == "past year"
        
        # Test with a query without time frame
        query = "Compare different machine learning algorithms"
        aspects = planner._parse_query(query)
        
        assert aspects["main_topic"] == query
        assert aspects["time_frame"] is None
    
    def test_get_depth_factors(self):
        """Test getting depth factors based on research depth."""
        # Test shallow depth
        planner = ResearchPlanner({"research_depth": "shallow"})
        shallow_factors = planner._get_depth_factors()
        
        assert shallow_factors["sources_per_topic"] == 3
        assert shallow_factors["analysis_detail"] == "basic"
        
        # Test medium depth
        planner = ResearchPlanner({"research_depth": "medium"})
        medium_factors = planner._get_depth_factors()
        
        assert medium_factors["sources_per_topic"] == 5
        assert medium_factors["analysis_detail"] == "moderate"
        
        # Test deep depth
        planner = ResearchPlanner({"research_depth": "deep"})
        deep_factors = planner._get_depth_factors()
        
        assert deep_factors["sources_per_topic"] == 10
        assert deep_factors["analysis_detail"] == "comprehensive"
    
    def test_generate_research_steps(self):
        """Test generating research steps based on query and depth factors."""
        planner = ResearchPlanner()
        
        query = "Analyze AI trends"
        query_aspects = planner._parse_query(query)
        depth_factors = planner._get_depth_factors()
        
        steps = planner._generate_research_steps(query, query_aspects, depth_factors)
        
        # Check that steps were generated
        assert len(steps) > 0
        
        # Check that each step has required fields
        for step in steps:
            assert "id" in step
            assert "name" in step
            assert "description" in step
            assert "tools" in step
            assert "expected_output" in step
            assert "estimated_time" in step
        
        # Test with deep research depth
        planner = ResearchPlanner({"research_depth": "deep"})
        depth_factors = planner._get_depth_factors()
        
        steps = planner._generate_research_steps(query, query_aspects, depth_factors)
        
        # Deep research should have more steps
        assert len(steps) > 5
    
    def test_estimate_completion_time(self):
        """Test estimating completion time based on steps."""
        planner = ResearchPlanner()
        
        # Create some test steps
        steps = [
            {"estimated_time": 30},
            {"estimated_time": 60},
            {"estimated_time": 45}
        ]
        
        # Estimate completion time
        time = planner._estimate_completion_time(steps)
        
        # Check that the time is calculated correctly
        # For medium depth, the overhead factor is 1.5
        expected_time = (30 + 60 + 45) * 1.5
        assert time == int(expected_time)
        
        # Test with different research depth
        planner = ResearchPlanner({"research_depth": "deep"})
        time = planner._estimate_completion_time(steps)
        
        # For deep depth, the overhead factor is 2.0
        expected_time = (30 + 60 + 45) * 2.0
        assert time == int(expected_time)
