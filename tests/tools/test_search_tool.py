"""
Unit tests for the SearchAPITool class.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from datetime import datetime

from tools.search_tool import SearchAPITool


class TestSearchAPITool:
    """Test cases for the SearchAPITool class."""

    def test_initialization(self):
        """Test that the search tool initializes with default and custom values."""
        # Test with default values
        search_tool = SearchAPITool(api_key="test_key")
        
        assert search_tool.api_key == "test_key"
        assert search_tool.engine == "google"
        assert search_tool.max_results == 10
        
        # Test with custom values
        search_tool = SearchAPITool(
            api_key="test_key",
            engine="duckduckgo",
            max_results=5
        )
        
        assert search_tool.api_key == "test_key"
        assert search_tool.engine == "duckduckgo"
        assert search_tool.max_results == 5
        
        # Test with unsupported engine
        search_tool = SearchAPITool(
            api_key="test_key",
            engine="unsupported_engine"
        )
        
        # Should fall back to google
        assert search_tool.engine == "google"
    
    @patch('requests.get')
    def test_search_google(self, mock_get):
        """Test Google search via SerpAPI."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "organic_results": [
                {
                    "title": "Test Result 1",
                    "link": "https://example1.com",
                    "snippet": "This is a test result 1",
                    "position": 1
                },
                {
                    "title": "Test Result 2",
                    "link": "https://example2.com",
                    "snippet": "This is a test result 2",
                    "position": 2
                }
            ],
            "knowledge_graph": {
                "title": "Test Knowledge Graph",
                "type": "Test Type",
                "description": "This is a test knowledge graph"
            },
            "search_information": {
                "total_results": 2
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        # Create search tool and perform search
        search_tool = SearchAPITool(api_key="test_key", engine="google")
        results = search_tool.search("test query")
        
        # Check that the request was made with correct parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["q"] == "test query"
        assert kwargs["params"]["api_key"] == "test_key"
        assert kwargs["params"]["engine"] == "google"
        
        # Check the results
        assert len(results["results"]) == 2
        assert results["results"][0]["title"] == "Test Result 1"
        assert results["results"][1]["title"] == "Test Result 2"
        assert results["knowledge_graph"]["title"] == "Test Knowledge Graph"
        assert results["total_results"] == 2
        assert "metadata" in results
        assert results["metadata"]["query"] == "test query"
        assert results["metadata"]["engine"] == "google"
    
    @patch('requests.get')
    def test_search_brave(self, mock_get):
        """Test Brave search via Brave Search API."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Test Result 1",
                        "url": "https://example1.com",
                        "description": "This is a test result 1"
                    },
                    {
                        "title": "Test Result 2",
                        "url": "https://example2.com",
                        "description": "This is a test result 2"
                    }
                ],
                "totalResults": 2
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        # Create search tool and perform search
        search_tool = SearchAPITool(api_key="test_key", engine="brave")
        results = search_tool.search("test query")
        
        # Check that the request was made with correct parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["q"] == "test query"
        assert kwargs["headers"]["X-Subscription-Token"] == "test_key"
        
        # Check the results
        assert len(results["results"]) == 2
        assert results["results"][0]["title"] == "Test Result 1"
        assert results["results"][1]["title"] == "Test Result 2"
        assert results["total_results"] == 2
        assert "metadata" in results
        assert results["metadata"]["query"] == "test query"
        assert results["metadata"]["engine"] == "brave"
    
    @patch('requests.get')
    def test_search_duckduckgo(self, mock_get):
        """Test DuckDuckGo search via Instant Answer API."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "Heading": "Test Heading",
            "AbstractText": "This is a test abstract",
            "AbstractURL": "https://example.com",
            "RelatedTopics": [
                {
                    "Text": "Topic 1 - Description 1",
                    "FirstURL": "https://example1.com"
                },
                {
                    "Text": "Topic 2 - Description 2",
                    "FirstURL": "https://example2.com"
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        # Create search tool and perform search
        search_tool = SearchAPITool(api_key="test_key", engine="duckduckgo")
        results = search_tool.search("test query")
        
        # Check that the request was made with correct parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["q"] == "test query"
        assert kwargs["params"]["format"] == "json"
        
        # Check the results
        assert len(results["results"]) == 3  # Abstract + 2 related topics
        assert results["results"][0]["title"] == "Test Heading"
        assert results["results"][0]["link"] == "https://example.com"
        assert results["results"][0]["snippet"] == "This is a test abstract"
        assert results["instant_answer"]["title"] == "Test Heading"
        assert results["instant_answer"]["text"] == "This is a test abstract"
        assert "metadata" in results
        assert results["metadata"]["query"] == "test query"
        assert results["metadata"]["engine"] == "duckduckgo"
    
    @patch('requests.get')
    def test_search_error_handling(self, mock_get):
        """Test error handling during search."""
        # Mock a request exception
        mock_get.side_effect = Exception("Test error")
        
        # Create search tool
        search_tool = SearchAPITool(api_key="test_key")
        
        # Search should raise a RuntimeError
        with pytest.raises(RuntimeError) as excinfo:
            search_tool.search("test query")
        
        # Check the error message
        assert "Search failed" in str(excinfo.value)
    
    @patch.object(SearchAPITool, 'search')
    def test_search_multi_engine(self, mock_search):
        """Test searching across multiple engines."""
        # Mock the search method to return different results for different engines
        def mock_search_impl(query, **kwargs):
            engine = self.engine
            if engine == "google":
                return {
                    "results": [{"title": "Google Result", "link": "https://example.com"}],
                    "metadata": {"engine": "google"}
                }
            elif engine == "duckduckgo":
                return {
                    "results": [{"title": "DuckDuckGo Result", "link": "https://example.org"}],
                    "metadata": {"engine": "duckduckgo"}
                }
            return {"results": [], "metadata": {"engine": engine}}
        
        mock_search.side_effect = mock_search_impl
        
        # Create search tool
        search_tool = SearchAPITool(api_key="test_key")
        
        # Test multi-engine search
        results = search_tool.search_multi_engine(
            "test query",
            engines=["google", "duckduckgo"]
        )
        
        # Check that search was called for each engine
        assert mock_search.call_count == 2
        
        # Check the combined results
        assert "results" in results
        assert "metadata" in results
        assert "engines" in results["metadata"]
        assert "google" in results["metadata"]["engines"]
        assert "duckduckgo" in results["metadata"]["engines"]
