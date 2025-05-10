"""
Unit tests for the URLFetcherTool class.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import requests

from tools.url_fetcher_tool import URLFetcherTool


class TestURLFetcherTool:
    """Test cases for the URLFetcherTool class."""

    def test_initialization(self):
        """Test that the URL fetcher tool initializes with default and custom values."""
        # Test with default values
        url_fetcher = URLFetcherTool()
        
        assert url_fetcher.timeout == 30
        assert url_fetcher.max_size == 1048576  # 1MB
        assert "Research Agent" in url_fetcher.user_agent
        
        # Test with custom values
        url_fetcher = URLFetcherTool(
            timeout=10,
            max_size=524288,  # 512KB
            user_agent="Custom User Agent"
        )
        
        assert url_fetcher.timeout == 10
        assert url_fetcher.max_size == 524288
        assert url_fetcher.user_agent == "Custom User Agent"
    
    def test_validate_url(self):
        """Test URL validation."""
        url_fetcher = URLFetcherTool()
        
        # Valid URLs should not raise an exception
        url_fetcher.fetch("https://example.com")  # This will be mocked
        
        # Invalid URLs should raise a ValueError
        with pytest.raises(ValueError):
            url_fetcher.fetch("not-a-url")
    
    @patch('requests.get')
    def test_fetch_html(self, mock_get):
        """Test fetching HTML content."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.headers = {
            "Content-Type": "text/html; charset=utf-8"
        }
        mock_response.status_code = 200
        
        # Create a simple HTML content
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="This is a test page">
        </head>
        <body>
            <main>
                <h1>Test Content</h1>
                <p>This is a test paragraph.</p>
                <a href="https://example.com/link1">Link 1</a>
                <a href="/link2">Link 2</a>
            </main>
        </body>
        </html>
        """
        
        # Set up the mock to return the HTML content
        mock_response.iter_content.return_value = [html_content]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        # Create URL fetcher and fetch the content
        url_fetcher = URLFetcherTool()
        result = url_fetcher.fetch("https://example.com")
        
        # Check that the request was made with correct parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs["timeout"] == 30
        assert kwargs["stream"] is True
        assert "User-Agent" in kwargs["headers"]
        assert "Accept-Language" in kwargs["headers"]
        
        # Check the result
        assert result["url"] == "https://example.com"
        assert result["status_code"] == 200
        assert result["content_type"] == "text/html; charset=utf-8"
        assert result["is_html"] is True
        assert result["title"] == "Test Page"
        assert result["description"] == "This is a test page"
        assert "Test Content" in result["content"]
        assert "This is a test paragraph" in result["content"]
        
        # Check that links were extracted
        assert len(result["links"]) == 2
        assert result["links"][0]["href"] == "https://example.com/link1"
        assert result["links"][0]["text"] == "Link 1"
        assert result["links"][1]["href"] == "https://example.com/link2"  # Relative link should be resolved
        assert result["links"][1]["text"] == "Link 2"
    
    @patch('requests.get')
    def test_fetch_non_html(self, mock_get):
        """Test fetching non-HTML content."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.headers = {
            "Content-Type": "application/json"
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        # Create URL fetcher and fetch the content
        url_fetcher = URLFetcherTool()
        result = url_fetcher.fetch("https://example.com/api")
        
        # Check the result
        assert result["url"] == "https://example.com/api"
        assert result["status_code"] == 200
        assert result["content_type"] == "application/json"
        assert result["is_html"] is False
        assert "title" not in result
        assert "content" not in result
    
    @patch('requests.get')
    def test_fetch_with_size_limit(self, mock_get):
        """Test fetching content with size limit."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.headers = {
            "Content-Type": "text/html; charset=utf-8"
        }
        mock_response.status_code = 200
        
        # Create a large HTML content that exceeds the size limit
        large_content = "a" * 1000000  # 1MB of content
        
        # Set up the mock to return chunks of content
        mock_response.iter_content.return_value = [large_content[:500000], large_content[500000:]]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        # Create URL fetcher with a small size limit
        url_fetcher = URLFetcherTool(max_size=600000)  # 600KB
        result = url_fetcher.fetch("https://example.com")
        
        # The content should be truncated
        assert len(result["content"]) < len(large_content)
    
    @patch('requests.get')
    def test_fetch_error_handling(self, mock_get):
        """Test error handling during fetch."""
        # Mock a request exception
        mock_get.side_effect = requests.RequestException("Test error")
        
        # Create URL fetcher
        url_fetcher = URLFetcherTool()
        
        # Fetch should raise a RuntimeError
        with pytest.raises(RuntimeError) as excinfo:
            url_fetcher.fetch("https://example.com")
        
        # Check the error message
        assert "Failed to fetch URL" in str(excinfo.value)
    
    @patch.object(URLFetcherTool, 'fetch')
    def test_fetch_multiple(self, mock_fetch):
        """Test fetching content from multiple URLs."""
        # Mock the fetch method to return different results for different URLs
        def mock_fetch_impl(url, extract_content=True):
            if url == "https://example1.com":
                return {
                    "url": url,
                    "title": "Example 1",
                    "content": "Content 1"
                }
            elif url == "https://example2.com":
                return {
                    "url": url,
                    "title": "Example 2",
                    "content": "Content 2"
                }
            elif url == "https://error.com":
                raise RuntimeError("Test error")
            return {}
        
        mock_fetch.side_effect = mock_fetch_impl
        
        # Create URL fetcher
        url_fetcher = URLFetcherTool()
        
        # Test fetching multiple URLs
        urls = ["https://example1.com", "https://example2.com", "https://error.com"]
        results = url_fetcher.fetch_multiple(urls)
        
        # Check that fetch was called for each URL
        assert mock_fetch.call_count == 3
        
        # Check the results
        assert "successful" in results
        assert "failed" in results
        assert "summary" in results
        
        assert "https://example1.com" in results["successful"]
        assert "https://example2.com" in results["successful"]
        assert "https://error.com" in results["failed"]
        
        assert results["summary"]["total"] == 3
        assert results["summary"]["successful"] == 2
        assert results["summary"]["failed"] == 1
        
        # Test with ignore_errors=False
        mock_fetch.reset_mock()
        mock_fetch.side_effect = mock_fetch_impl
        
        # Should raise an exception on the first error
        with pytest.raises(RuntimeError):
            url_fetcher.fetch_multiple(urls, ignore_errors=False)
            
        # Should have called fetch only twice (stops at the error)
        assert mock_fetch.call_count == 3
