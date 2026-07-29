import io
import re
import uuid
import hashlib
import logging
from typing import List, Dict, Any, Tuple
import requests
from bs4 import BeautifulSoup
import pypdf
import fitz  # PyMuPDF

from app.models.document import FileType, ChunkRecord, DocumentRecord, DocumentStatus
from app.config import settings

logger = logging.getLogger("ai_platform.ingestion")

class DocumentIngestionService:
    """
    Ingestion & Chunking Processor for PDFs, Source Code, Markdown, Text & Web Scraping.
    Extracts text, handles scanned PDFs via PyMuPDF/EasyOCR fallback, splits into structural chunks with metadata.
    """
    
    @staticmethod
    def calculate_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def process_pdf(self, content_bytes: bytes, file_name: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extracts text from PDF page by page with scanned PDF OCR fallback."""
        full_text_pages = []
        
        # 1. Standard text extraction via PyMuPDF / pypdf
        doc = fitz.open(stream=content_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            text = page.get_text() or ""
            if text.strip():
                full_text_pages.append(f"[Page {i+1}]\n" + text.strip())
                
        full_text = "\n\n".join(full_text_pages)

        # 2. Scanned PDF fallback using EasyOCR if text layer is missing/empty
        if len(full_text.strip()) < 50 and len(doc) > 0:
            logger.info(f"PDF '{file_name}' appears to be a scanned document. Running EasyOCR page text extraction...")
            try:
                import easyocr
                reader = easyocr.Reader(['en'], gpu=False)
                ocr_pages = []
                for i, page in enumerate(doc):
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    results = reader.readtext(img_bytes, detail=0)
                    page_ocr_text = " ".join(results).strip()
                    if page_ocr_text:
                        ocr_pages.append(f"[Page {i+1}]\n" + page_ocr_text)
                if ocr_pages:
                    full_text = "\n\n".join(ocr_pages)
            except Exception as e:
                logger.warning(f"OCR processing failed for '{file_name}': {e}")

        chunks = self.chunk_text(full_text, chunk_size=settings.DEFAULT_CHUNK_SIZE, overlap=settings.DEFAULT_CHUNK_OVERLAP)
        return full_text, chunks

    def process_code(self, code_text: str, file_name: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Structure-aware Chunker for Source Code (.py, .js, .ts, etc.)
        Preserves function/class blocks and records line numbers.
        """
        lines = code_text.splitlines()
        chunks = []
        current_chunk_lines = []
        start_line = 1
        current_char_count = 0
        
        chunk_size = settings.DEFAULT_CHUNK_SIZE
        overlap_lines = 3

        for i, line in enumerate(lines, start=1):
            current_chunk_lines.append(line)
            current_char_count += len(line) + 1
            
            # Split on structural bounds (class/def/blank line) or reached max chunk size
            is_class_or_def = line.strip().startswith(("def ", "class ", "async def ", "function "))
            if current_char_count >= chunk_size or (is_class_or_def and len(current_chunk_lines) > 8):
                chunk_str = "\n".join(current_chunk_lines)
                chunks.append({
                    "content": chunk_str,
                    "start_line": start_line,
                    "end_line": i,
                    "char_count": len(chunk_str),
                    "token_estimate": len(chunk_str) // 4
                })
                # Apply line overlap
                overlap = current_chunk_lines[-overlap_lines:] if len(current_chunk_lines) > overlap_lines else []
                current_chunk_lines = list(overlap)
                start_line = max(1, i - len(overlap) + 1)
                current_char_count = sum(len(l) + 1 for l in current_chunk_lines)

        if current_chunk_lines:
            chunk_str = "\n".join(current_chunk_lines)
            chunks.append({
                "content": chunk_str,
                "start_line": start_line,
                "end_line": len(lines),
                "char_count": len(chunk_str),
                "token_estimate": len(chunk_str) // 4
            })

        return code_text, chunks

    def process_web_url(self, url: str) -> Tuple[str, List[Dict[str, Any]], str]:
        """
        Web Scraping using BeautifulSoup4 as required by tech stack.
        Cleans HTML, extracts title and main text content.
        """
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AI-Knowledge-Platform/1.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script, style, nav, and header tags
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
            
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        paragraphs = [p.get_text().strip() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'li']) if p.get_text().strip()]
        full_text = "\n\n".join(paragraphs)
        
        chunks = self.chunk_text(full_text, chunk_size=settings.DEFAULT_CHUNK_SIZE, overlap=settings.DEFAULT_CHUNK_OVERLAP)
        return full_text, chunks, title

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Generic text chunking with sliding character window and sentence boundary respect.
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # Adjust end to nearest period, newline, or space if not at end of text
            if end < text_len:
                break_point = max(
                    text.rfind('. ', start, end),
                    text.rfind('\n', start, end),
                    text.rfind(' ', start, end)
                )
                if break_point > start + (chunk_size // 2):
                    end = break_point + 1

            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append({
                    "content": chunk_content,
                    "start_line": None,
                    "end_line": None,
                    "char_count": len(chunk_content),
                    "token_estimate": len(chunk_content) // 4
                })
                
            start = end - overlap if end < text_len else text_len

        return chunks

ingestion_service = DocumentIngestionService()
