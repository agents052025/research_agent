"""
PDF Reader Tool for the Research Agent.
Extracts and processes content from PDF documents.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import fitz  # PyMuPDF
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
                # PyMuPDF can open from bytes
                pdf_bytes = pdf_source.getvalue()
                pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                source_type = "url"
            else:
                # Open from file path
                pdf_doc = fitz.open(pdf_source)
                source_type = "file"
                
            # Get basic metadata
            total_pages = len(pdf_doc)
            
            # Limit to max_pages if no specific pages requested
            if not page_numbers:
                page_numbers = list(range(min(total_pages, self.max_pages)))
            
            # Extract content from specified pages
            page_contents = {}
            for page_num in page_numbers:
                if page_num < total_pages:
                    try:
                        page = pdf_doc[page_num]
                        text = page.get_text()
                        
                        # Store page content (1-indexed for output)
                        page_contents[str(page_num + 1)] = text
                    except Exception as e:
                        self.logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                        page_contents[str(page_num + 1)] = f"[Error extracting text: {str(e)}]"
            
            # Extract metadata if available
            metadata = {}
            if pdf_doc.metadata:
                for key, value in pdf_doc.metadata.items():
                    if value:  # Only add non-empty values
                        metadata[key.lower()] = str(value)
            
            # Create result
            result = {
                "total_pages": total_pages,
                "processed_pages": len(page_contents),
                "source_type": source_type,
                "metadata": metadata,
                "content": page_contents,
                "timestamp": datetime.now().isoformat()
            }
            
            if isinstance(pdf_source, str):
                result["source"] = pdf_source
                
            # Close the document
            pdf_doc.close()
                
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
        Uses PyMuPDF's text extraction with more advanced section detection.
        
        Args:
            source: Path or URL to the PDF
            
        Returns:
            Dictionary containing extracted sections
        """
        try:
            # Open the PDF document
            if source.startswith(('http://', 'https://')):
                pdf_file = self._fetch_pdf_from_url(source)
                pdf_bytes = pdf_file.getvalue()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            else:
                # Ensure the file exists
                if not os.path.exists(source):
                    raise ValueError(f"PDF file not found: {source}")
                doc = fitz.open(source)
            
            # Initialize sections dictionary
            sections = {}
            total_pages = len(doc)
            
            # Process each page to identify sections
            for page_idx in range(min(total_pages, self.max_pages)):
                page = doc[page_idx]
                
                # Extract text blocks with their formatting information
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            if "spans" in line:
                                for span in line["spans"]:
                                    # Check if this is likely a header (larger font, bold, etc.)
                                    text = span.get("text", "").strip()
                                    font_size = span.get("size", 0)
                                    font_flags = span.get("flags", 0)
                                    
                                    # Font flags: 1=superscript, 2=italic, 4=serifed, 8=monospaced, 16=bold
                                    is_bold = (font_flags & 16) != 0
                                    
                                    # Heuristic for headers: larger font or bold text
                                    if (font_size > 12 or is_bold) and text and len(text) < 100:
                                        # Likely a section header
                                        if text not in sections:
                                            sections[text] = []
                                    elif text:
                                        # Add to the current section if one exists
                                        current_sections = list(sections.keys())
                                        if current_sections:
                                            latest_section = current_sections[-1]
                                            sections[latest_section].append(text)
                                        else:
                                            # No section found yet, create a default one
                                            sections["Introduction"] = [text]
            
            # Join section content into strings
            for section, content_list in sections.items():
                sections[section] = "\n".join(content_list)
            
            # Close the document
            doc.close()
            
            # Add metadata
            result = {
                "source": source,
                "total_pages": total_pages,
                "sections": sections,
                "section_count": len(sections),
                "metadata": doc.metadata if hasattr(doc, "metadata") else {},
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting sections from PDF: {str(e)}")
            raise RuntimeError(f"Failed to extract sections: {str(e)}")
        
    def summarize_pdf(self, source: str, max_length: int = 1000) -> Dict[str, Any]:
        """
        Provide a summary of PDF content using PyMuPDF.
        Extracts key information from each page with improved formatting.
        
        Args:
            source: Path or URL to the PDF
            max_length: Maximum length of summary per page
            
        Returns:
            Dictionary with PDF summary
        """
        try:
            # Open the PDF document
            if source.startswith(('http://', 'https://')):
                pdf_file = self._fetch_pdf_from_url(source)
                pdf_bytes = pdf_file.getvalue()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            else:
                # Ensure the file exists
                if not os.path.exists(source):
                    raise ValueError(f"PDF file not found: {source}")
                doc = fitz.open(source)
            
            total_pages = len(doc)
            page_summaries = {}
            total_words = 0
            
            # Process each page
            for page_idx in range(min(total_pages, self.max_pages)):
                page = doc[page_idx]
                
                # Extract text with better formatting
                text = page.get_text("text")
                words_on_page = len(text.split())
                total_words += words_on_page
                
                # Create summary for this page
                if text:
                    # Take the beginning of the page text up to max_length
                    if len(text) > max_length:
                        # Try to cut at a sentence boundary
                        cut_point = text[:max_length].rfind('.')
                        if cut_point > max_length // 2:  # Only use sentence boundary if it's not too short
                            summary = text[:cut_point+1]
                        else:
                            summary = text[:max_length]
                        summary += "..."
                    else:
                        summary = text
                    
                    # Store summary (1-indexed for output)
                    page_summaries[str(page_idx + 1)] = summary
            
            # Extract TOC (Table of Contents) if available
            toc = doc.get_toc()
            toc_entries = []
            if toc:
                for t in toc:
                    level, title, page = t
                    toc_entries.append({
                        "level": level,
                        "title": title,
                        "page": page
                    })
            
            # Close the document
            doc.close()
            
            # Create result
            result = {
                "source": source,
                "total_pages": total_pages,
                "page_summaries": page_summaries,
                "total_word_count": total_words,
                "toc": toc_entries,
                "metadata": doc.metadata if hasattr(doc, "metadata") else {},
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error summarizing PDF: {str(e)}")
            raise RuntimeError(f"Failed to summarize PDF: {str(e)}")
