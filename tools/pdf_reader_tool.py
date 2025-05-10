"""
PDF Reader Tool for the Research Agent.
Extracts and processes content from PDF documents.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import PyPDF2
import requests
from io import BytesIO


class PDFReaderTool:
    """
    Extracts and processes content from PDF documents.
    Supports reading from local files and URLs.
    """
    
    def __init__(self, max_pages: int = 50, extract_images: bool = False):
        """
        Initialize the PDF Reader Tool.
        
        Args:
            max_pages: Maximum number of pages to process
            extract_images: Whether to extract images (not implemented yet)
        """
        self.logger = logging.getLogger(__name__)
        self.max_pages = max_pages
        self.extract_images = extract_images
        
    def read_pdf(self, source: str, pages: Optional[Union[int, List[int], str]] = None) -> Dict[str, Any]:
        """
        Read and extract content from a PDF.
        
        Args:
            source: Path or URL to the PDF
            pages: Specific pages to extract (int, list, or range string like "1-5")
                  If None, extracts all pages up to max_pages
                  
        Returns:
            Dictionary containing extracted content and metadata
            
        Raises:
            ValueError: If source is invalid
            RuntimeError: If PDF processing fails
        """
        self.logger.info(f"Reading PDF from: {source}")
        
        try:
            # Check if source is a URL or local file
            if source.startswith(('http://', 'https://')):
                pdf_file = self._fetch_pdf_from_url(source)
            else:
                # Ensure the file exists
                if not os.path.exists(source):
                    raise ValueError(f"PDF file not found: {source}")
                pdf_file = source
                
            # Open and read the PDF
            return self._process_pdf(pdf_file, pages)
            
        except Exception as e:
            self.logger.error(f"Error reading PDF: {str(e)}")
            raise RuntimeError(f"Failed to read PDF: {str(e)}")
            
    def _fetch_pdf_from_url(self, url: str) -> BytesIO:
        """
        Fetch a PDF from a URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            BytesIO object containing the PDF
            
        Raises:
            RuntimeError: If fetch fails
        """
        try:
            self.logger.info(f"Fetching PDF from URL: {url}")
            
            # Request the PDF
            headers = {
                "User-Agent": "Research Agent/1.0",
                "Accept": "application/pdf"
            }
            
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get("Content-Type", "").lower()
            if "application/pdf" not in content_type:
                self.logger.warning(f"URL {url} may not be a PDF: {content_type}")
                
            # Create BytesIO object from content
            pdf_io = BytesIO(response.content)
            return pdf_io
            
        except requests.RequestException as e:
            self.logger.error(f"Error fetching PDF from URL {url}: {str(e)}")
            raise RuntimeError(f"Failed to fetch PDF from URL: {str(e)}")
            
    def _process_pdf(self, pdf_source: Union[str, BytesIO], pages: Optional[Union[int, List[int], str]] = None) -> Dict[str, Any]:
        """
        Process a PDF file and extract content.
        
        Args:
            pdf_source: PDF file path or BytesIO object
            pages: Specific pages to extract
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        # Parse pages parameter
        page_numbers = self._parse_pages_parameter(pages)
        
        try:
            # Open the PDF file
            if isinstance(pdf_source, BytesIO):
                pdf = PyPDF2.PdfReader(pdf_source)
                source_type = "url"
            else:
                pdf = PyPDF2.PdfReader(open(pdf_source, "rb"))
                source_type = "file"
                
            # Get basic metadata
            num_pages = len(pdf.pages)
            
            # Limit to max_pages if needed
            if not page_numbers and num_pages > self.max_pages:
                self.logger.warning(f"PDF has {num_pages} pages, limiting to first {self.max_pages}")
                page_numbers = list(range(self.max_pages))
                
            # If still no page numbers, extract all pages
            if not page_numbers:
                page_numbers = list(range(num_pages))
                
            # Validate page numbers
            page_numbers = [p for p in page_numbers if 0 <= p < num_pages]
            
            # Extract content from specified pages
            page_contents = {}
            for page_num in page_numbers:
                page = pdf.pages[page_num]
                text = page.extract_text()
                
                if text:
                    page_contents[str(page_num + 1)] = text
                    
            # Extract document metadata
            metadata = {}
            if pdf.metadata:
                for key, value in pdf.metadata.items():
                    if key.startswith('/'):
                        key = key[1:]
                    metadata[key] = str(value)
                    
            # Prepare result
            result = {
                "source_type": source_type,
                "total_pages": num_pages,
                "extracted_pages": len(page_contents),
                "metadata": metadata,
                "content": page_contents,
                "timestamp": datetime.now().isoformat()
            }
            
            if isinstance(pdf_source, str):
                result["source"] = pdf_source
                
            self.logger.info(f"Successfully extracted content from {len(page_contents)} pages")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise RuntimeError(f"Failed to process PDF: {str(e)}")
            
    def _parse_pages_parameter(self, pages: Optional[Union[int, List[int], str]] = None) -> List[int]:
        """
        Parse the pages parameter into a list of page numbers.
        
        Args:
            pages: Pages specification
            
        Returns:
            List of page numbers (0-indexed)
        """
        if pages is None:
            return []
            
        if isinstance(pages, int):
            return [max(0, pages - 1)]  # Convert to 0-indexed
            
        if isinstance(pages, list):
            return [max(0, p - 1) for p in pages]  # Convert to 0-indexed
            
        if isinstance(pages, str):
            result = []
            
            # Handle comma-separated values and ranges
            parts = pages.split(',')
            for part in parts:
                part = part.strip()
                
                if '-' in part:
                    # Handle range like "1-5"
                    try:
                        start, end = part.split('-')
                        start = max(0, int(start.strip()) - 1)  # Convert to 0-indexed
                        end = int(end.strip())
                        result.extend(range(start, end))
                    except ValueError:
                        self.logger.warning(f"Invalid page range: {part}")
                else:
                    # Handle single page
                    try:
                        page = max(0, int(part) - 1)  # Convert to 0-indexed
                        result.append(page)
                    except ValueError:
                        self.logger.warning(f"Invalid page number: {part}")
                        
            return result
            
        self.logger.warning(f"Unrecognized pages parameter: {pages}")
        return []
        
    def extract_text_by_sections(self, source: str) -> Dict[str, Any]:
        """
        Extract text from a PDF and attempt to organize by sections.
        This is a simple heuristic approach to identify sections.
        
        Args:
            source: Path or URL to the PDF
            
        Returns:
            Dictionary containing extracted sections
        """
        # First get the regular content
        pdf_content = self.read_pdf(source)
        
        # Process the content to identify sections
        sections = {}
        current_section = "Introduction"
        section_content = []
        
        for page_num, text in pdf_content["content"].items():
            lines = text.split('\n')
            
            for line in lines:
                # Simple heuristic for section headers: all caps, short, ends with colon
                line_stripped = line.strip()
                if (line_stripped.isupper() and len(line_stripped) < 50) or \
                   (line_stripped.endswith(':') and len(line_stripped) < 50):
                    # Save previous section
                    if section_content:
                        sections[current_section] = '\n'.join(section_content)
                        section_content = []
                        
                    # Start new section
                    current_section = line_stripped
                else:
                    section_content.append(line_stripped)
                    
        # Save the last section
        if section_content:
            sections[current_section] = '\n'.join(section_content)
            
        # Add metadata
        result = {
            "source": source if isinstance(source, str) else "url",
            "total_pages": pdf_content["total_pages"],
            "sections": sections,
            "section_count": len(sections),
            "metadata": pdf_content["metadata"],
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    def summarize_pdf(self, source: str, max_length: int = 1000) -> Dict[str, Any]:
        """
        Provide a summary of PDF content.
        This is a simple approach that returns the beginning of each page.
        
        Args:
            source: Path or URL to the PDF
            max_length: Maximum length of summary per page
            
        Returns:
            Dictionary with PDF summary
        """
        # Get the full content
        pdf_content = self.read_pdf(source)
        
        # Create summaries for each page
        page_summaries = {}
        
        for page_num, text in pdf_content["content"].items():
            # Simple approach: take first part of each page
            if text:
                summary = text[:max_length]
                if len(text) > max_length:
                    summary += "..."
                page_summaries[page_num] = summary
                
        # Calculate total word count
        total_words = sum(len(text.split()) for text in pdf_content["content"].values())
        
        # Create result
        result = {
            "source": source if isinstance(source, str) else "url",
            "total_pages": pdf_content["total_pages"],
            "page_summaries": page_summaries,
            "total_word_count": total_words,
            "metadata": pdf_content["metadata"],
            "timestamp": datetime.now().isoformat()
        }
        
        return result
