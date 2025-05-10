"""
URL Fetcher Tool for the Research Agent.
Fetches and processes content from web URLs.
"""

import logging
import requests
from typing import Dict, Any, Optional
from datetime import datetime
from bs4 import BeautifulSoup
import validators


class URLFetcherTool:
    """
    Fetches and processes content from web URLs.
    Extracts main content, title, and metadata from web pages.
    """
    
    def __init__(self, timeout: int = 30, max_size: int = 1048576, user_agent: Optional[str] = None):
        """
        Initialize the URL Fetcher Tool.
        
        Args:
            timeout: Request timeout in seconds
            max_size: Maximum content size to process in bytes
            user_agent: Custom user agent string
        """
        self.logger = logging.getLogger(__name__)
        self.timeout = timeout
        self.max_size = max_size
        self.user_agent = user_agent or "Research Agent/1.0"
        
    def fetch(self, url: str, extract_content: bool = True) -> Dict[str, Any]:
        """
        Fetch content from a URL.
        
        Args:
            url: URL to fetch
            extract_content: Whether to extract and process the main content
            
        Returns:
            Dictionary containing fetched content and metadata
            
        Raises:
            ValueError: If URL is invalid
            RuntimeError: If fetch fails
        """
        # Validate URL
        if not validators.url(url):
            raise ValueError(f"Invalid URL: {url}")
            
        self.logger.info(f"Fetching URL: {url}")
        
        try:
            # Prepare headers
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml",
                "Accept-Language": "uk-UA,uk;q=0.9,en-US;q=0.8,en;q=0.7"
            }
            
            # Stream the request to handle large files
            response = requests.get(
                url,
                headers=headers,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get("Content-Type", "").lower()
            if not content_type.startswith("text/html") and not "application/xhtml+xml" in content_type:
                self.logger.warning(f"URL {url} is not HTML content: {content_type}")
                
                # For non-HTML content, return basic metadata
                return {
                    "url": url,
                    "status_code": response.status_code,
                    "content_type": content_type,
                    "headers": dict(response.headers),
                    "timestamp": datetime.now().isoformat(),
                    "is_html": False
                }
                
            # Read content with size limit
            content = ""
            content_size = 0
            for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                content_size += len(chunk)
                if content_size > self.max_size:
                    self.logger.warning(f"Content size exceeds limit ({self.max_size} bytes)")
                    break
                content += chunk
                
            # Extract and process content
            result = {
                "url": url,
                "status_code": response.status_code,
                "content_type": content_type,
                "headers": dict(response.headers),
                "timestamp": datetime.now().isoformat(),
                "is_html": True
            }
            
            if extract_content:
                processed_content = self._process_html(content, url)
                result.update(processed_content)
            else:
                result["raw_content"] = content
                
            self.logger.info(f"Successfully fetched URL: {url}")
            return result
            
        except requests.RequestException as e:
            self.logger.error(f"Error fetching URL {url}: {str(e)}")
            raise RuntimeError(f"Failed to fetch URL: {str(e)}")
            
    def _process_html(self, html_content: str, url: str) -> Dict[str, Any]:
        """
        Process HTML content to extract useful information.
        
        Args:
            html_content: Raw HTML content
            url: Source URL
            
        Returns:
            Dictionary with processed content
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.text.strip()
                
            # Extract meta description
            description = ""
            meta_desc = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
            if meta_desc and meta_desc.get('content'):
                description = meta_desc['content'].strip()
                
            # Extract main content
            # First try article or main tags
            main_content = soup.find('article') or soup.find('main')
            
            # If not found, look for common content containers
            if not main_content:
                for container in ['#content', '#main', '.content', '.main', '.article', '.post']:
                    main_content = soup.select_one(container)
                    if main_content:
                        break
                        
            # If still not found, use body
            if not main_content:
                main_content = soup.body
                
            # Extract text content
            content_text = ""
            if main_content:
                # Remove script and style tags
                for script in main_content.find_all(['script', 'style']):
                    script.decompose()
                    
                # Get text content
                content_text = main_content.get_text(separator='\n').strip()
                
                # Clean up: remove excessive newlines and whitespace
                import re
                content_text = re.sub(r'\n\s*\n', '\n\n', content_text)
                content_text = re.sub(r' +', ' ', content_text)
                
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Handle relative URLs
                if href.startswith('/') and not href.startswith('//'):
                    from urllib.parse import urlparse
                    base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
                    href = base_url + href
                    
                links.append({
                    "text": link.text.strip(),
                    "href": href
                })
                
            return {
                "title": title,
                "description": description,
                "content": content_text,
                "links": links,
                "word_count": len(content_text.split()) if content_text else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error processing HTML content: {str(e)}")
            return {
                "title": "",
                "description": "",
                "content": "",
                "links": [],
                "error": str(e)
            }
            
    def fetch_multiple(self, urls: list, extract_content: bool = True, ignore_errors: bool = True) -> Dict[str, Any]:
        """
        Fetch content from multiple URLs.
        
        Args:
            urls: List of URLs to fetch
            extract_content: Whether to extract and process the main content
            ignore_errors: Whether to continue on error
            
        Returns:
            Dictionary containing results for each URL
        """
        results = {
            "successful": {},
            "failed": {},
            "summary": {
                "total": len(urls),
                "successful": 0,
                "failed": 0,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        for url in urls:
            try:
                result = self.fetch(url, extract_content)
                results["successful"][url] = result
                results["summary"]["successful"] += 1
            except Exception as e:
                if not ignore_errors:
                    raise
                    
                results["failed"][url] = str(e)
                results["summary"]["failed"] += 1
                
        return results
